"""Resume o treino observado; não exige loss monotônica em minibatches."""
import argparse, json, math, statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();root=Path(args.run)
rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
assert len(rows)>=2000 and [r['epoch'] for r in rows]==list(range(1,len(rows)+1))
assert all(math.isfinite(r[k]) for r in rows for k in ('loss','grad_l2','grad_max'))
first=statistics.mean(r['loss'] for r in rows[:100]);last=statistics.mean(r['loss'] for r in rows[-100:])
result={'steps':len(rows),'first_loss':rows[0]['loss'],'last_loss':rows[-1]['loss'],'first_100_mean':first,'last_100_mean':last,'mean_loss_decreased':last<first,'final_ema':rows[-1]['loss_ema'],'max_grad_l2':max(r['grad_l2'] for r in rows),'max_grad_element':max(r['grad_max'] for r in rows),'elapsed_seconds':rows[-1]['elapsed_seconds'],'all_recorded_values_finite':True,'note':'Checagem de pesos/gradientes finitos feita pelo trainer a cada passo. Não comprova qualidade de linguagem nem estabilidade por horas.'}
with (root/'stability-report.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
