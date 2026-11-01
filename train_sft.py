#!/usr/bin/env python3
"""Orquestra SFT nativo Rust; não usa PyTorch nem APIs externas."""
import argparse,json,os,subprocess,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,default=ROOT/'runs/mac-context-256-adapted')
p.add_argument('--output',type=Path,default=ROOT/'runs/mac-sft-special-v1')
p.add_argument('--train',type=Path,default=ROOT/'data/sft/arandu_sft.jsonl')
p.add_argument('--eval',type=Path,default=ROOT/'data/sft/arandu_eval.jsonl')
p.add_argument('--corpus',type=Path,default=ROOT/'corpus/machado-dom-casmurro.txt')
p.add_argument('--steps',type=int,default=3000)
p.add_argument('--adapt-steps',type=int,default=0)
p.add_argument('--learning-rate',type=float,default=5e-4)
p.add_argument('--dashboard-pointer',type=Path,default=ROOT/'active-run.json')
a=p.parse_args()
if not math.isfinite(a.learning_rate) or a.learning_rate<=0:p.error('learning-rate deve ser finita e positiva')
if a.steps<=0 or a.adapt_steps<0:p.error('steps positivo; adapt-steps não negativo')
train=a.train.resolve();holdout=a.eval.resolve()
tr=[json.loads(x) for x in train.read_text().splitlines()];ev=[json.loads(x) for x in holdout.read_text().splitlines()]
assert len(tr)>=200 and not ({r['text'] for r in tr}&{r['text'] for r in ev})
if a.output.exists():p.error('output deve ser novo; checkpoint base nunca é sobrescrito')
a.dashboard_pointer.parent.mkdir(parents=True,exist_ok=True)
pointer=a.dashboard_pointer.with_suffix('.tmp')
pointer.write_text(json.dumps({'metrics':str((a.output/'metrics.jsonl').resolve()),'name':a.output.name}))
pointer.replace(a.dashboard_pointer)
env=dict(os.environ);env.setdefault('RAYON_NUM_THREADS','4')
subprocess.run(['cargo','run','--release','--locked','-p','katu-server','--bin','train_sft','--',str(a.base.resolve()),str(train),str(holdout),str(a.corpus.resolve()),str(a.output.resolve()),str(a.steps),str(a.adapt_steps),str(a.learning_rate)],cwd=ROOT,env=env,check=True)
