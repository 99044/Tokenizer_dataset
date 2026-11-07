"""Run GDN chunks in model precision, with selective IEEE FP32 recomputation."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def gradient_recovery_action(nonfinite, spike, fp32, retry_enabled):
    """Retry nonfinite low-precision steps; keep finite clipped gradients."""
    if nonfinite and retry_enabled and not fp32:
        return 'retry_fp32'
    if nonfinite:
        return 'abort_nonfinite'
    return 'accept_clipped' if spike else 'accept'


class GDNStabilityGuard:
    def __init__(self, original, output_dir, rank, force_fp32=False):
        self.original = original
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.force_fp32 = force_fp32
        self.context = {}
        self.fallback_count = 0
        self.call_index = 0

    def set_context(self, step, micro_step):
        self.context = {'step': step, 'micro_step': micro_step}
        self.call_index = 0

    def __call__(self, q, k, v, g, beta, scale=None, initial_state=None,
                 output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None):
        arguments = dict(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                         initial_state=initial_state, output_final_state=output_final_state,
                         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)
        ordinal = self.call_index
        self.call_index += 1
        if self.force_fp32:
            promoted = dict(arguments)
            for name in ['q', 'k', 'v', 'g', 'beta', 'initial_state']:
                if isinstance(promoted[name], torch.Tensor):
                    promoted[name] = promoted[name].float()
            output32, state32 = self.original(**promoted)
            if not torch.isfinite(output32).all().item():
                raise RuntimeError('GDN IEEE FP32 recurrence output is nonfinite')
            if state32 is not None and not torch.isfinite(state32).all().item():
                raise RuntimeError('GDN IEEE FP32 recurrence state is nonfinite')
            corrected = output32.to(q.dtype)
            if not torch.isfinite(corrected).all().item():
                raise RuntimeError('GDN IEEE FP32 recurrence exceeds the model dtype')
            return corrected, state32
        result = self.original(**arguments)
        output, state = result
        output_finite = bool(torch.isfinite(output).all().item())
        state_finite = state is None or bool(torch.isfinite(state).all().item())
        if output_finite and state_finite:
            return result
        original_dtype = output.dtype
        bad_elements = output.numel() - int(torch.isfinite(output).sum().item())
        # Release the failed autograd branch before building its replacement.
        # No backward/update has run for this result, and the original inputs
        # remain connected to their projections and trainable parameters.
        del output, state, result
        promoted = dict(arguments)
        for name in ['q', 'k', 'v', 'g', 'beta', 'initial_state']:
            if isinstance(promoted[name], torch.Tensor):
                promoted[name] = promoted[name].float()
        output32, state32 = self.original(**promoted)
        if not torch.isfinite(output32).all().item():
            raise RuntimeError('GDN IEEE FP32 recomputation is nonfinite')
        if state32 is not None and not torch.isfinite(state32).all().item():
            raise RuntimeError('GDN IEEE FP32 recomputed state is nonfinite')
        corrected = output32.to(original_dtype)
        if not torch.isfinite(corrected).all().item():
            raise RuntimeError('GDN recomputation exceeds the original output dtype range')
        self.fallback_count += 1
        event = {
            'event': 'gdn_ieee_fp32_recompute', 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'rank': self.rank, **self.context, 'gdn_ordinal': ordinal,
            'original_output_dtype': str(original_dtype), 'nonfinite_output_elements': bad_elements,
            'recompute_dtype': 'torch.float32', 'dot_precision': 'ieee',
            'recomputed_output_abs_max': float(output32.abs().max().item()),
            'fallback_count': self.fallback_count,
        }
        path = self.output_dir / f'numerical-fallbacks-rank-{self.rank}.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
        print(json.dumps(event), flush=True)
        return corrected, state32


_guard = None


def install_gdn_stability_guard(output_dir, rank, force_fp32=False):
    global _guard
    if os.environ.get('TRITON_F32_DEFAULT') != 'ieee':
        raise RuntimeError('GDN stability requires TRITON_F32_DEFAULT=ieee')
    if _guard is None:
        import olmo_core.nn.attention.recurrent as recurrent
        _guard = GDNStabilityGuard(recurrent.dispatch_chunk_gated_delta_rule, output_dir, rank, force_fp32)
        recurrent.dispatch_chunk_gated_delta_rule = _guard
    elif _guard.force_fp32 != force_fp32:
        raise RuntimeError('GDN recurrence precision cannot change within a process')
    return _guard
