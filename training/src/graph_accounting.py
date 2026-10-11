"""Source-bound FLOPs ledgers for Hybrid and nonstandard computation graphs.

No kernel forward/backward multiplier is guessed. --auto-hybrid generates the
versioned complete reference graph; custom pending components still prevent
planning or certifying a budget. Counts describe one micro-batch, including
every recurrence iteration and repeated invocation; DP and accumulation are
applied exactly once afterwards. The original dense Megatron CLI stays separate.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys


SCHEMA = 'scicomp26-graph-flops-v1'
HYBRID_TYPES = {'olmo_hybrid_1b', 'olmo_hybrid_scaling', 'olmo_hybrid_1b_with_signals'}
EXCLUSIONS = {'optimizer_update', 'activation_recomputation', 'communication'}


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8_388_608), b''):
            digest.update(block)
    return digest.hexdigest()


def gemm_operations(m, n, k, *, calls=1, input_gradient, weight_gradient):
    """2mnk forward, plus only the input/weight gradients actually computed."""
    for name, value in [('m', m), ('n', n), ('k', k), ('calls', calls)]:
        integer(value, name)
    if not isinstance(input_gradient, bool) or not isinstance(weight_gradient, bool):
        raise ValueError('gradient requirements must be explicit booleans')
    operations = [dict(kind='matmul', phase='forward', m=m, n=n, k=k, calls=calls)]
    if input_gradient:
        operations.append(dict(kind='matmul', phase='backward', m=m, n=k, k=n, calls=calls))
    if weight_gradient:
        operations.append(dict(kind='matmul', phase='backward', m=n, n=m, k=k, calls=calls))
    return operations


def operation_flops(operation):
    """Count logical operations, with no precision-dependent scaling."""
    kind = operation.get('kind')
    calls = integer(operation.get('calls'), 'operation.calls')
    if kind == 'matmul':
        cost = calls * 2 * integer(operation.get('m'), 'm') * integer(operation.get('n'), 'n') * integer(operation.get('k'), 'k')
        causal_half = operation.get('causal_reference_half')
        if causal_half is not None and not isinstance(causal_half, bool):
            raise ValueError('causal_reference_half must be an explicit boolean')
        if causal_half is True:
            if cost % 2:
                raise ValueError('Megatron causal half must yield an integer count')
            cost //= 2
        return cost
    if kind in {'add', 'subtract', 'multiply', 'fma'}:
        return calls * integer(operation.get('elements'), 'elements') * (2 if kind == 'fma' else 1)
    if kind == 'reduce_sum':
        groups = integer(operation.get('groups'), 'groups')
        width = integer(operation.get('width'), 'width')
        return calls * groups * (width - 1)
    if kind == 'audited_scalar':
        # The supplied rules do not assign a cost to exp/division/sqrt etc.
        # Such costs need an explicit convention and source, never a default.
        if not operation.get('operation') or not operation.get('counting_basis'):
            raise ValueError('audited_scalar requires operation and counting_basis')
        return calls * integer(operation.get('elements'), 'elements') * integer(operation.get('flops_per_element'), 'flops_per_element', minimum=0)
    raise ValueError(f'unsupported operation {kind!r}; decompose it into reviewed logical operations')


def required_components(model):
    if model.get('type') not in HYBRID_TYPES:
        raise ValueError('graph templates currently cover the reviewed Hybrid architectures')
    if model.get('context_length') != 8192 or model.get('activation') != 'swiglu':
        raise ValueError('review the graph for a changed sequence length or activation first')
    layers = integer(model.get('num_layers'), 'num_layers')
    pattern = model.get('block_pattern')
    if pattern != ['gdn', 'gdn', 'gdn', 'attn'] or layers % len(pattern):
        raise ValueError('review component coverage for the changed block pattern first')
    signals = model['type'] == 'olmo_hybrid_1b_with_signals'
    components = {'embedding', 'output.gemm', 'output.elementwise'}
    for layer in range(layers):
        prefix = f'layer.{layer}'
        components.update({f'{prefix}.ffn.gemm', f'{prefix}.ffn.elementwise', f'{prefix}.norm_residual'})
        mixer = pattern[layer % len(pattern)]
        if mixer == 'gdn':
            components.update({f'{prefix}.gdn.gemm', f'{prefix}.gdn.convolution', f'{prefix}.gdn.delta_rule', f'{prefix}.gdn.elementwise'})
            if signals:
                components.update({f'{prefix}.signal.scan_arithmetic', f'{prefix}.signal.scan_other'})
        else:
            components.update({f'{prefix}.attn.gemm', f'{prefix}.attn.core', f'{prefix}.attn.elementwise'})
            if signals:
                components.update({f'{prefix}.signal.gate_forward', f'{prefix}.signal.gate_backward', f'{prefix}.signal.group_mean'})
    return components


def make_template(model_config, settings, micro_batch_size, vocab_size):
    model = settings['model']
    required_components(model)
    sequence = integer(model.get('context_length'), 'context_length')
    micro = integer(micro_batch_size, 'micro_batch_size')
    vocab = integer(vocab_size, 'padded_vocab_size')
    hidden = integer(model.get('hidden_size'), 'hidden_size')
    intermediate = integer(model.get('intermediate_size'), 'intermediate_size')
    heads = integer(model.get('num_attention_heads'), 'num_attention_heads')
    kv_heads = integer(model.get('num_key_value_heads'), 'num_key_value_heads')
    head_dim = integer(model.get('attention_head_dim'), 'attention_head_dim')
    gdn_heads = integer(model.get('gdn_num_heads'), 'gdn_num_heads')
    value_heads = integer(model.get('gdn_num_value_heads'), 'gdn_num_value_heads')
    key_dim = integer(model.get('gdn_head_dim'), 'gdn_head_dim')
    expanded = Decimal(str(model.get('gdn_value_expand'))) * key_dim
    if not expanded.is_finite() or expanded != expanded.to_integral_value() or expanded <= 0:
        raise ValueError('GDN value dimension must be a positive integer')
    value_dim = int(expanded)
    tokens = micro * sequence
    signals = model['type'] == 'olmo_hybrid_1b_with_signals'
    if signals and (settings.get('signal', {}).get('detach') is not True or value_heads != gdn_heads):
        raise ValueError('review the signal graph for detach=False or unequal GDN head counts')
    root = Path(__file__).resolve().parents[2]
    source_names = ['training/src/model_config.py', 'training/src/train_ddp.py']
    if signals:
        source_names.append('training/src/signal_hybrid.py')
    sources = {name: {'path': name, 'sha256': file_hash(root / name)} for name in source_names}
    components = []

    def add(identity, operations=None, *, forward_only=False, note=''):
        components.append(dict(
            id=identity, status='pending', forward_only=forward_only,
            operations=operations or [], evidence=[], counting_basis=note,
        ))

    def gemms(identity, dimensions):
        operations = []
        for input_width, output_width in dimensions:
            operations.extend(gemm_operations(tokens, input_width, output_width,
                                               input_gradient=True, weight_gradient=True))
        add(identity, operations, note='Candidate dimensions only; verify actual module widths, biases and gradient requirements against the installed source.')

    add('embedding', note='Lookup has no floating-point GEMM; account for actual backward scatter additions and shared-weight gradient accumulation.')
    gemms('output.gemm', [(hidden, vocab)])
    add('output.elementwise', note='Output RMSNorm and any output bias; keep loss-only bookkeeping separate from the prediction graph.')
    pattern = model['block_pattern']
    for layer in range(model['num_layers']):
        prefix = f'layer.{layer}'
        gemms(f'{prefix}.ffn.gemm', [(hidden, intermediate), (hidden, intermediate), (intermediate, hidden)])
        add(f'{prefix}.ffn.elementwise', note='SwiGLU activations, multiplication and any biases, forward and actual backward.')
        add(f'{prefix}.norm_residual', note='Both reordered RMSNorms and residual additions, forward and actual backward.')
        if pattern[layer % len(pattern)] == 'attn':
            query = heads * head_dim
            kv = kv_heads * head_dim
            dims = [(hidden, query), (hidden, kv), (hidden, kv), (query, hidden)]
            if signals:
                dims.append((hidden, heads))
            gemms(f'{prefix}.attn.gemm', dims)
            add(f'{prefix}.attn.core', note='QK^T and AV, forward and actual backward; explicitly record causal/window semantics and the chosen Megatron reference convention.')
            add(f'{prefix}.attn.elementwise', note='QK norms, RoPE, softmax, scaling; signals also add sigmoid and head-output multiplication, with actual backward.')
            if signals:
                gate_elements = tokens * heads
                add(f'{prefix}.signal.gate_forward', [
                    dict(kind='multiply', phase='forward', elements=gate_elements, calls=1),
                    dict(kind='add', phase='forward', elements=gate_elements, calls=2),
                ], forward_only=True, note='Only Wx + rho * signal_scale + signal_bias arithmetic; projection and sigmoid are covered separately. Verify broadcast shapes.')
                add(f'{prefix}.signal.gate_backward', note='Actual signal_scale/signal_bias gradients and reductions. Detached rho does not receive a gradient.')
                add(f'{prefix}.signal.group_mean', forward_only=True, note='Sum three GDN signals and divide by three; do not deduplicate the additions between layers.')
        else:
            key_width = gdn_heads * key_dim
            value_width = value_heads * value_dim
            gemms(f'{prefix}.gdn.gemm', [
                (hidden, key_width), (hidden, key_width), (hidden, value_width),
                (hidden, value_heads), (hidden, value_heads),
                (hidden, value_width), (value_width, hidden),
            ])
            add(f'{prefix}.gdn.convolution', note='All three causal depthwise convolutions, fused SiLU and actual input/weight gradients; bind the FLA kernel source/version.')
            add(f'{prefix}.gdn.delta_rule', note='Actual chunk_gated_delta_rule forward/backward DAG, including QK L2 normalization and every chunk/iteration. Do not substitute a recurrent approximation or blanket factor 3.')
            add(f'{prefix}.gdn.elementwise', note='Beta sigmoid/negative-eigenvalue scaling, decay construction, gated RMSNorm, and their actual backward; separate kernel-internal work to avoid double counting.')
            if signals:
                iterations = micro * sequence * gdn_heads
                state_elements = key_dim * value_dim
                add(f'{prefix}.signal.scan_arithmetic', [
                    dict(kind='multiply', phase='forward', elements=state_elements, calls=iterations),
                    dict(kind='matmul', phase='forward', m=1, n=key_dim, k=value_dim, calls=iterations),
                    dict(kind='subtract', phase='forward', elements=value_dim, calls=iterations),
                    dict(kind='multiply', phase='forward', elements=value_dim, calls=iterations),
                    dict(kind='multiply', phase='forward', elements=state_elements, calls=iterations),
                    dict(kind='add', phase='forward', elements=state_elements, calls=iterations),
                ], forward_only=True, note='Separate reference scan: decay, prediction, error, beta scaling, outer product and state addition, once per head and token. This is algorithmic work, not activation checkpoint recomputation.')
                add(f'{prefix}.signal.scan_other', forward_only=True, note='Key normalization, exp(g), two L2 norms, ratio and head mean/clipping. Record scalar-op conventions and boundary behavior; no backward under detach=True.')
    return dict(
        schema=SCHEMA, status='incomplete', model_type=model['type'],
        model_config_sha256=file_hash(model_config),
        batch_shape=dict(micro_batch_size=micro, sequence_length=sequence, padded_vocab_size=vocab),
        model_shape=dict(
            num_layers=model['num_layers'], block_pattern=pattern,
            hidden_size=hidden, intermediate_size=intermediate,
            num_attention_heads=heads, num_key_value_heads=kv_heads,
            attention_head_dim=head_dim, gdn_num_heads=gdn_heads,
            gdn_num_value_heads=value_heads, gdn_head_dim=key_dim,
            gdn_value_dim=value_dim,
        ),
        source_files=sources, components=components,
        notes=[
            'Fill every component, attach local source SHA-256 evidence and set status=reviewed; only then set top-level status=complete.',
            'Candidate GEMMs assume input and weight gradients. Remove only gradients the actual computation graph does not compute, including frozen/no-grad cases.',
            'Parameter sharing deduplicates N and N_a only; FLOPs count every call and loop iteration.',
            'Add kernel source files to source_files and reference their keys in evidence. A repository wrapper alone is not evidence for an opaque FLA kernel.',
            'Omitted/zero-cost arithmetic requires a counting_basis explaining the exact reference exclusion; do not silently omit nonstandard operators.',
        ],
    )


def audit_ledger(ledger, model_config, settings, micro_batch_size, vocab_size, source_root):
    if not isinstance(ledger, dict):
        raise ValueError('ledger must be a JSON object')
    if ledger.get('schema') != SCHEMA or ledger.get('status') != 'complete':
        raise ValueError('a complete reviewed graph ledger is required; the template is not a FLOPs result')
    model = settings['model']
    if model.get('type') == 'olmo_hybrid_1b_with_signals' and settings.get('signal', {}).get('detach') is not True:
        raise ValueError('review component coverage and backward routing for detach=False first')
    if ledger.get('model_config_sha256') != file_hash(model_config) or ledger.get('model_type') != model['type']:
        raise ValueError('ledger model/configuration does not match this run')
    shape = dict(micro_batch_size=micro_batch_size, sequence_length=model['context_length'], padded_vocab_size=vocab_size)
    if ledger.get('batch_shape') != shape:
        raise ValueError('ledger batch/sequence/vocabulary dimensions differ from the actual run')
    value_dim = Decimal(str(model['gdn_value_expand'])) * model['gdn_head_dim']
    expected_model_shape = dict(
        num_layers=model['num_layers'], block_pattern=model['block_pattern'],
        hidden_size=model['hidden_size'], intermediate_size=model['intermediate_size'],
        num_attention_heads=model['num_attention_heads'],
        num_key_value_heads=model['num_key_value_heads'],
        attention_head_dim=model['attention_head_dim'],
        gdn_num_heads=model['gdn_num_heads'],
        gdn_num_value_heads=model['gdn_num_value_heads'],
        gdn_head_dim=model['gdn_head_dim'], gdn_value_dim=int(value_dim),
    )
    if ledger.get('model_shape') != expected_model_shape:
        raise ValueError('ledger model dimensions differ from the actual configuration')
    sources = ledger.get('source_files')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('source files and SHA-256 evidence are required')
    for name in ('training/src/model_config.py', 'training/src/train_ddp.py'):
        if name not in sources or sources[name].get('path') != name:
            raise ValueError(f'actual model/runner source binding is required: {name}')
    if model['type'] == 'olmo_hybrid_1b_with_signals':
        name = 'training/src/signal_hybrid.py'
        if name not in sources or sources[name].get('path') != name:
            raise ValueError('actual signal implementation source binding is required')
    for key, source in sources.items():
        path = Path(source['path'])
        if not path.is_absolute():
            path = source_root / path
        if source.get('sha256') != file_hash(path):
            raise ValueError(f'accounting evidence source changed: {key}')
    components = ledger.get('components')
    if not isinstance(components, list):
        raise ValueError('components must be a list')
    if any(not isinstance(component, dict) or not isinstance(component.get('id'), str) for component in components):
        raise ValueError('every component needs a string id')
    identities = [component['id'] for component in components]
    if len(set(identities)) != len(identities) or set(identities) != required_components(model):
        raise ValueError('duplicate, missing or unexpected graph components; review graph coverage first')
    result = {}
    for component in components:
        identity = component['id']
        if component.get('status') != 'reviewed' or not component.get('counting_basis'):
            raise ValueError(f'{identity}: reviewed source-backed counting_basis is required')
        evidence = component.get('evidence')
        if not isinstance(evidence, list) or not evidence or any(key not in sources for key in evidence):
            raise ValueError(f'{identity}: source evidence is required')
        if not isinstance(component.get('forward_only'), bool):
            raise ValueError(f'{identity}: explicit forward_only boolean required')
        if '.signal.scan_' in identity and settings.get('signal', {}).get('detach') is True and not component['forward_only']:
            raise ValueError(f'{identity}: detached scan must be forward-only')
        operations = component.get('operations')
        if not isinstance(operations, list):
            raise ValueError(f'{identity}: operations must be a list')
        if not operations and not component.get('zero_cost_justification'):
            raise ValueError(f'{identity}: empty operations require an explicit source-backed zero_cost_justification')
        if identity.endswith(('.gdn.convolution', '.gdn.delta_rule')):
            implementation = component.get('implementation', {})
            if not implementation.get('package_version_or_commit') or not implementation.get('entrypoint'):
                raise ValueError(f'{identity}: actual nonstandard kernel version/entrypoint are required')
            if not any(sources[key].get('role') == 'kernel' for key in evidence):
                raise ValueError(f'{identity}: bind the actual kernel source, not just its Python wrapper')
        counts = {'forward': 0, 'backward': 0, 'excluded': 0}
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError(f'{identity}: every operation must be a JSON object')
            phase = operation.get('phase')
            if phase not in {'forward', 'backward'}:
                raise ValueError(f'{identity}: phase must be forward or backward')
            if component['forward_only'] and phase == 'backward':
                raise ValueError(f'{identity}: forward-only component contains backward computation')
            cost = operation_flops(operation)
            exclusion = operation.get('exclude')
            if exclusion is not None:
                if exclusion not in EXCLUSIONS or not operation.get('exclusion_basis'):
                    raise ValueError(f'{identity}: unsupported/unexplained FLOPs exclusion')
                counts['excluded'] += cost
            else:
                counts[phase] += cost
        counts['total'] = counts['forward'] + counts['backward']
        result[identity] = counts
    totals = {name: sum(counts[name] for counts in result.values()) for name in ('forward', 'backward', 'excluded', 'total')}
    if totals['total'] <= 0:
        raise ValueError('complete graph must have positive FLOPs')
    return {'components': result, 'totals': totals, 'source_files': sources}


def decimal_fraction(value):
    with localcontext() as context:
        context.prec = 50
        return str(Decimal(value.numerator) / value.denominator)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from count_flops import audit_metrics, budget_report, positive_flops, PDF_BUDGETS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--write-ledger-template', type=Path)
    modes.add_argument('--operator-ledger', type=Path)
    modes.add_argument('--auto-hybrid', action='store_true', help='Generate the complete versioned Hybrid graph automatically')
    parser.add_argument('--scalar-op-cost', type=int, help='Explicit cost of exp/sqrt/division/log; competition rules do not specify it')
    parser.add_argument('--write-operator-ledger', type=Path, help='Save the complete automatically generated operator ledger')
    parser.add_argument('--runtime-metadata', type=Path, action='append', help='Runtime kernel/model evidence; repeat for the initial run and each resumed segment')
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--padded-vocab-size', type=int, required=True)
    parser.add_argument('--micro-batch-size', type=int, default=1)
    parser.add_argument('--data-parallel-size', type=int)
    parser.add_argument('--grad-accum-steps', type=int)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--target-flops', type=positive_flops)
    parser.add_argument('--run-plan', type=Path)
    parser.add_argument('--metrics', type=Path)
    parser.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        destinations = [args.write_ledger_template, args.output, args.write_operator_ledger]
        inputs = [args.model_config, args.operator_ledger, args.run_plan, args.metrics] + (args.runtime_metadata or [])
        for destination in destinations:
            if destination is not None and any(path is not None and path.resolve() == destination.resolve() for path in inputs):
                raise ValueError('accounting output must not overwrite a model configuration, ledger or run record')
        resolved = [path.resolve() for path in destinations if path is not None]
        if len(set(resolved)) != len(resolved):
            raise ValueError('result and ledger output paths must differ')
        settings = yaml.safe_load(args.model_config.read_text(encoding='utf-8'))
        if not isinstance(settings, dict) or not isinstance(settings.get('model'), dict):
            raise ValueError('configuration must contain a model mapping')
        micro = integer(args.micro_batch_size, 'micro_batch_size')
        vocab = integer(args.padded_vocab_size, 'padded_vocab_size')
        if args.write_ledger_template:
            write_json(args.write_ledger_template, make_template(args.model_config, settings, micro, vocab))
            print(f'Incomplete audit template written to {args.write_ledger_template}; no budget is certified.')
            return
        if args.output is None:
            raise ValueError('--output is required for graph accounting')
        dp = integer(args.data_parallel_size, 'data_parallel_size')
        accumulation = integer(args.grad_accum_steps, 'grad_accum_steps')
        if args.auto_hybrid:
            if args.scalar_op_cost is None:
                raise ValueError('--auto-hybrid requires an explicit --scalar-op-cost')
            from src.hybrid_flops import complete_hybrid_ledger
            ledger = complete_hybrid_ledger(args.model_config, settings, micro, vocab, args.scalar_op_cost)
        else:
            if args.scalar_op_cost is not None or args.write_operator_ledger is not None:
                raise ValueError('--scalar-op-cost and --write-operator-ledger require --auto-hybrid')
            ledger = json.loads(args.operator_ledger.read_text(encoding='utf-8'))
        audit = audit_ledger(ledger, args.model_config, settings, micro, vocab, args.source_root)
        tokens_per_micro = micro * settings['model']['context_length']
        multiplier = dp * accumulation
        cost = audit['totals']['total'] * multiplier
        tokens_per_step = tokens_per_micro * multiplier
        if args.plan_only:
            if args.target_flops is None:
                raise ValueError('--target-flops is required for planning')
            steps = args.target_flops // cost
            if steps < 1:
                raise ValueError('budget is smaller than one complete optimizer update')
            if args.target_flops in PDF_BUDGETS and steps * cost < args.target_flops * 99 // 100:
                raise ValueError('integer update plan falls below the previously supplied 1% tolerance; reduce the global batch')
            record_kind = 'planned_training_budget_not_completed_training'
        else:
            if args.run_plan is None or args.metrics is None:
                raise ValueError('--run-plan and --metrics are required for actual training accounting')
            steps = integer(args.checkpoint_step, 'checkpoint_step')
            run_plan = json.loads(args.run_plan.read_text(encoding='utf-8'))
            if run_plan.get('settings') != settings:
                raise ValueError('actual run configuration differs from the accounting configuration')
            # The current runner is pure DDP; TP/PP/CP need a different audited trace.
            if (run_plan.get('world_size'), run_plan.get('micro_batch_size'), run_plan.get('grad_accum_steps')) != (dp, micro, accumulation):
                raise ValueError('actual DDP/micro-batch/accumulation differs from the ledger run')
            data = run_plan.get('data', {})
            if data.get('context_length') != settings['model']['context_length'] or data.get('vocab_size') != vocab:
                raise ValueError('actual sequence/output vocabulary width differs from the ledger')
            if steps > integer(run_plan.get('max_steps'), 'run-plan.max_steps'):
                raise ValueError('checkpoint exceeds the declared training plan')
            if ledger.get('calculation_kind') == 'versioned_logical_reference_graph':
                if args.runtime_metadata is None:
                    raise ValueError('actual automatic accounting requires --runtime-metadata; --plan-only needs no runtime evidence')
                from src.hybrid_flops import verify_runtime_history
                audit['runtime_graph'] = verify_runtime_history(args.runtime_metadata, ledger, steps)
                ledger['runtime_kernel_verified'] = True
            audit['metrics'] = audit_metrics(args.metrics, steps, tokens_per_step)
            audit['run_plan_sha256'] = file_hash(args.run_plan)
            audit['metrics_sha256'] = file_hash(args.metrics)
            record_kind = 'completed_training_graph_accounting'
        total = cost * steps
        per_token = Fraction(audit['totals']['total'], tokens_per_micro)
        serialized_ledger = json.dumps(ledger, ensure_ascii=False, indent=2) + '\n'
        if args.write_operator_ledger is not None:
            write_json(args.write_operator_ledger, ledger)
        ledger_hash = (hashlib.sha256(serialized_ledger.encode('utf-8')).hexdigest()
                       if args.auto_hybrid else file_hash(args.operator_ledger))
        result = dict(
            record_kind=record_kind, protocol=SCHEMA, official_result=False,
            model_type=settings['model']['type'], operator_ledger_sha256=ledger_hash,
            operator_ledger_path=str(args.write_operator_ledger or args.operator_ledger) if (args.write_operator_ledger or args.operator_ledger) else None,
            operator_ledger=ledger if args.auto_hybrid and args.write_operator_ledger is None else None,
            calculation_kind=ledger.get('calculation_kind', 'custom_source_bound_graph'),
            runtime_kernel_verified=ledger.get('runtime_kernel_verified', False),
            conventions=ledger.get('conventions'), scalar_op_cost=ledger.get('scalar_op_cost'),
            implementation_sha256=file_hash(Path(__file__)),
            runtime=dict(data_parallel_size=dp, micro_batch_size=micro, grad_accum_steps=accumulation),
            optimizer_steps=steps, tokens_per_step=tokens_per_step,
            training_tokens=steps * tokens_per_step,
            flops_per_token=decimal_fraction(per_token),
            flops_per_token_exact=dict(numerator=per_token.numerator, denominator=per_token.denominator),
            per_micro_batch_flops=audit['totals'], per_step_flops=cost,
            cumulative_training_flops=total, audit=audit,
            budget_checks=budget_report(total, cost),
            rules=dict(add_or_multiply=1, fma=2, gemm='2mnk; count required gradients separately',
                       precision_independent=True, repeated_calls_counted=True,
                       excluded=sorted(EXCLUSIONS)),
        )
        if args.target_flops is not None:
            result.update(target_flops=args.target_flops, within_budget=total <= args.target_flops,
                          exact_target_at_complete_update_possible=args.target_flops % cost == 0)
        if args.plan_only:
            result['launcher_arguments'] = f'--micro-batch-size {micro} --grad-accum-steps {accumulation} --steps {steps}'
        write_json(args.output, result)
        print(json.dumps({key: value for key, value in result.items() if key not in {'audit', 'operator_ledger'}}, ensure_ascii=False, indent=2))
        if args.target_flops is not None and total > args.target_flops:
            raise SystemExit(1)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
