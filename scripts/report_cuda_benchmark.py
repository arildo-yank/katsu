#!/usr/bin/env python3
"""Report measured speed; estimate a corpus-token equivalent, not model quality."""
import argparse,json,math,statistics
from pathlib import Path

def report(run):
    cfg=json.loads((run/'config.json').read_text());end=json.loads((run/'sft-report.json').read_text())
    rows=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    if len(rows)!=100 or any(x['phase'] not in ('pretrain','adapt') for x in rows):raise ValueError('requires exactly 100 corpus steps')
    if [r['step'] for r in rows]!=list(range(1,101)):raise ValueError('missing or duplicate step')
    times=[r['step_seconds'] for r in rows]
    if any(not math.isfinite(t) or t<=0 for t in times):raise ValueError('invalid timing')
    if any(not math.isfinite(r['loss']) for r in rows):raise ValueError('nonfinite loss')
    tokens=sum(r['input_tokens'] for r in rows);seconds=sum(times)
    n=end['corpus_tokens'];equivalent=math.ceil((n-1)/cfg['train_sequence_length'])
    overhead=max(0,end['wall_seconds']-seconds)
    return dict(backend=cfg['backend'],parameters=cfg['parameters'],steps=100,
        mean_seconds_per_step=seconds/100,median_seconds_per_step=statistics.median(times),
        p95_seconds_per_step=sorted(times)[94],input_tokens=tokens,tokens_per_second=tokens/seconds,
        measured_step_seconds=seconds,wall_seconds=end['wall_seconds'],overhead_seconds=overhead,
        corpus_tokens=n,one_corpus_token_equivalent_steps=equivalent,
        one_corpus_equivalent_training_seconds=equivalent*seconds/100,
        estimated_500_corpus_plus_500_sft_seconds=1000*seconds/100+overhead,
        note='SFT estimate uses corpus-step speed as a proxy; evaluation/setup/download extra. Random windows do not guarantee document coverage. No automatic launch authorized by this report.')
def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=report(a.run)
    with a.output.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
