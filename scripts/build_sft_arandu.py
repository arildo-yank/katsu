"""302 exemplos sintéticos, sem diagnósticos inferidos de ACWR; dez valores em holdout."""
import json
from pathlib import Path
SYSTEM='Você é o Arandu. Responda apenas com base no JSON. Não infira risco de lesão. Se faltar dado, diga que não há dados suficientes.'
def row(i,data,question,response):
    prompt='<|system|>'+SYSTEM+'\n<|user|>Dados: '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\nPergunta: '+question+'\n<|assistant|>'
    return dict(id=i,synthetic=True,player=data,question=question,prompt=prompt,response=response,text=prompt+response+'<|end|>')
heldout_values=[23,38,49,62,73,78,84,87,92,97]
rows=[]
for value in range(101):
 if value in heldout_values:continue
 for variant in range(2):
  data={'readiness':value}
  if variant:data['session']='treino-sintetico'
  rows.append(row(f'readiness-{value}-{variant}',data,'Qual é a prontidão do jogador?',f'Prontidão registrada: {value}/100.'))
for i in range(40):
 value=round(0.6+i*0.05,2)
 rows.append(row(f'acwr-{i}',{'acwr':value},'Qual o risco de lesão?',f'ACWR registrado: {value}. Não é possível determinar risco de lesão apenas com esse dado.'))
questions=['Qual é a prontidão do jogador?','Qual foi a distância percorrida?','Qual foi a velocidade máxima?','Qual é o ACWR?','Qual é a carga de treino?','O jogador está lesionado?','Qual é a frequência cardíaca?','Quantos sprints fez?','Qual é a duração do treino?','Qual é o nível de fadiga?']
for i,question in enumerate(questions):
 for variant in range(2):
  rows.append(row(f'missing-{i}-{variant}',{} if variant==0 else {'session':'treino-sintetico'},question,'Não tenho dados suficientes no contexto fornecido para afirmar isso.'))
for value in range(100,6100,100):
 rows.append(row(f'distance-{value}',{'distance_m':value},'Qual foi a distância percorrida?',f'Distância registrada: {value} m.'))
holdout=[row(f'heldout-{v}',{'readiness':v},'Qual é a prontidão do jogador?',f'Prontidão registrada: {v}/100.') for v in heldout_values]
root=Path(__file__).resolve().parents[1]/'data'/'sft';root.mkdir(parents=True,exist_ok=True)
assert len(rows)==302 and len({r['text'] for r in rows})==302
assert not ({r['text'] for r in rows}&{r['text'] for r in holdout})
for name,records in [('arandu_sft.jsonl',rows),('arandu_eval.jsonl',holdout)]:
 (root/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
