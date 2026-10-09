#!/usr/bin/env python3
"""Account for training FLOPs using a supplied, audited FLOPs/token value.

Supply the official forward+backward FLOPs/token; no implicit 6P approximation
is made. Tokens include all pretraining/midtraining stages and padding compute.
"""

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flops-per-token', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--tokens', type=int)
    source.add_argument('--run-plan', type=Path, nargs='+', help='Full stage run-plan.json files; each stage exactly once')
    parser.add_argument('--budget', help='Maximum allowed FLOPs, e.g. 3e17')
    args = parser.parse_args()
    try:
        flops_per_token = Decimal(args.flops_per_token)
        budget = Decimal(args.budget) if args.budget else None
        if not flops_per_token.is_finite() or flops_per_token <= 0:
            raise ValueError('FLOPs/token must be finite and positive')
        if budget is not None and (not budget.is_finite() or budget <= 0):
            raise ValueError('budget must be finite and positive')
        tokens = args.tokens
        if args.run_plan:
            if len({p.resolve() for p in args.run_plan}) != len(args.run_plan):
                raise ValueError('a stage run-plan was supplied more than once')
            tokens = sum(int(json.loads(p.read_text())['planned_training_tokens']) for p in args.run_plan)
        if tokens < 0:
            raise ValueError('tokens must be nonnegative')
    except (ValueError, InvalidOperation, KeyError) as error:
        parser.error(str(error))
    total = flops_per_token * tokens
    result = dict(flops_per_token=str(flops_per_token), training_tokens=tokens,
                  total_training_flops=str(total), basis='supplied_audited_flops_per_token',
                  token_basis='planned_full_stages' if args.run_plan else 'supplied_actual_tokens')
    if budget is not None:
        result.update(budget=str(budget), within_budget=total <= budget)
    print(json.dumps(result, indent=2))
    if budget is not None and total > budget:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
