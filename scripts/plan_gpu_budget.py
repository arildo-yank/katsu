"""Price a complete pilot from measured CUDA timings; never provisions hardware."""
import argparse
import json
import math
from pathlib import Path


def plan(smoke, benchmark, hourly_rate, elapsed_seconds, fixed_cost=0.0):
    if not all(math.isfinite(x) and x >= 0 for x in (hourly_rate, elapsed_seconds, fixed_cost)) or hourly_rate == 0:
        raise ValueError('Actual hourly price and elapsed billable time are required')
    if benchmark['backend'] != 'cuda' or benchmark['steps'] != 100:
        raise ValueError('Requires a completed 100-step CUDA benchmark')
    if not smoke['finite_checks_passed'] or smoke['steps'] != 10:
        raise ValueError('Requires completed 10-step smoke')
    corpus_steps = math.ceil((benchmark['corpus_tokens'] - 1) / 256)
    sft_steps = 3 * smoke['train_examples']
    if corpus_steps <= 0 or sft_steps <= 0:
        raise ValueError('Empty training data')
    corpus_seconds = benchmark['mean_seconds_per_step']
    # Smoke elapsed includes evaluation/final writes: deliberately conservative.
    sft_seconds = smoke['elapsed_sft_seconds'] / smoke['steps']
    if not all(math.isfinite(x) and x > 0 for x in (corpus_seconds, sft_seconds)):
        raise ValueError('Invalid measured timings')
    training_seconds = 1.5 * (corpus_steps * corpus_seconds + sft_steps * sft_seconds)
    # Reserve setup/evaluation/export time plus $2 below the hard $10 cap.
    reserve_seconds = 1800 + benchmark['overhead_seconds']
    estimated_hours = math.ceil((elapsed_seconds + training_seconds + reserve_seconds) / 3600)
    estimated_total = fixed_cost + estimated_hours * hourly_rate
    affordable_hours = math.floor((8.0 - fixed_cost) / hourly_rate)
    return dict(authorized_cap_usd=10.0, planning_cap_usd=8.0,
                actual_hourly_rate_usd=hourly_rate, elapsed_billable_seconds=elapsed_seconds,
                pretrain_steps=corpus_steps, sft_steps=sft_steps,
                estimated_training_seconds=training_seconds,
                reserved_nontraining_seconds=reserve_seconds,
                estimated_total_usd=estimated_total,
                within_budget=estimated_total <= 8.0,
                stop_training_after_seconds=max(0, affordable_hours * 3600 - elapsed_seconds - reserve_seconds),
                note='Pilot: one corpus-token equivalent plus 3 SFT-example equivalents. '
                     'Random sampling does not guarantee every example is seen. '
                     'No model-quality guarantee. Provider deletion must be separately verified; '
                     'a local process timeout does not stop cloud billing.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-report', required=True, type=Path)
    p.add_argument('--benchmark-report', required=True, type=Path)
    p.add_argument('--hourly-rate', required=True, type=float)
    p.add_argument('--elapsed-seconds', required=True, type=float)
    p.add_argument('--fixed-cost', type=float, default=0.0)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    report = plan(json.loads(a.smoke_report.read_text()), json.loads(a.benchmark_report.read_text()),
                  a.hourly_rate, a.elapsed_seconds, a.fixed_cost)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report['within_budget']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
