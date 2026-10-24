"""Avalia respostas reais sem usar o fallback do cliente como acerto do modelo."""
import argparse,json
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8788');p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
opener=build_opener(ProxyHandler({}));results=[]
for line in Path(a.data).read_text().splitlines():
 row=json.loads(line)
 for temperature in [0.0,0.7]:
  req=Request(a.url+'/generate',json.dumps({'prompt':row['prompt'],'max_new_tokens':64,'temperature':temperature,'repetition_penalty':1.2,'seed':2026}).encode(),{'Content-Type':'application/json'})
  with opener.open(req,timeout=90) as r:answer=json.load(r)
  results.append({'id':row['id'],'temperature':temperature,'expected':row['response'],'response':answer,'exact_match':answer['text'].strip()==row['response'] and not answer['prompt_truncated']})
report={'cases':len(results),'exact_matches':sum(r['exact_match'] for r in results),'stopped_on_end':sum(r['response'].get('finish_reason')=='stop' for r in results),'exact_and_stopped':sum(r['exact_match'] and r['response'].get('finish_reason')=='stop' for r in results),'all_prompts_fit':all(not r['response']['prompt_truncated'] for r in results),'results':results}
with Path(a.output).open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!='results'}))
