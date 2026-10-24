#!/usr/bin/env python3
"""Versioned synthetic EN context -> PT response; no medical advice or external model."""
import json,argparse
from pathlib import Path

def row(i,context,q,answer,group):
    p='<|system|>Você é o Arandu. Responda em português apenas com base no contexto. Não infira diagnósticos.\n<|user|>Context: '+context+'\nPergunta: '+q+'\n<|assistant|>'
    return dict(id=i,synthetic=True,context_language='en',response_language='pt-BR',group=group,prompt=p,response=answer,text=p+answer+'<|end|>')
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    train=[json.loads(x) for x in (a.source/'arandu_sft.jsonl').read_text().splitlines()];ev=[json.loads(x) for x in (a.source/'arandu_eval.jsonl').read_text().splitlines()]
    facts=[
      ('The session lasted {v} minutes.','Quanto tempo durou a sessão?','A sessão durou {v} minutos.'),
      ('The player covered {v} metres.','Qual foi a distância percorrida?','O jogador percorreu {v} metros.'),
      ('The athlete completed {v} sprints.','Quantos sprints foram realizados?','O atleta realizou {v} sprints.'),
      ('The recovery period was {v} hours.','Qual foi o período de recuperação?','O período de recuperação foi de {v} horas.'),
    ]
    for j,(context,q,answer) in enumerate(facts):
        for v in range(10,40):
            target=ev if v in (17,29,37) else train
            target.append(row(f'en-fact-{j}-{v}',context.format(v=v),q,answer.format(v=v),'numeric'))
    pairs=[
      ('The report measures training load. It does not estimate injury probability.','O relatório estima a chance de lesão?','Não. O relatório mede carga de treino e não estima a probabilidade de lesão.'),
      ('The study reports an association, not a causal effect.','O estudo demonstra causalidade?','Não. O estudo relata uma associação, não um efeito causal.'),
      ('Ligament refers to tissue connecting bones at a joint.','O que significa ligament neste contexto?','Neste contexto, ligament significa ligamento, tecido que conecta ossos em uma articulação.'),
      ('Biomechanics studies movement and the forces acting on the body.','O que o texto diz sobre biomecânica?','O texto diz que a biomecânica estuda o movimento e as forças que atuam sobre o corpo.'),
      ('The dataset contains no readiness score.','Qual é a prontidão do atleta?','O contexto não informa a prontidão do atleta.'),
      ('The report has no information about knee injuries.','O atleta tem lesão no joelho?','O contexto não informa se o atleta tem lesão no joelho.'),
      ('Recovery was monitored, but no return-to-play decision was recorded.','O atleta foi liberado para jogar?','O contexto não registra uma decisão de retorno ao jogo.'),
      ('The ACWR value is reported without an individual injury diagnosis.','Esse dado confirma uma lesão?','Não. O contexto não apresenta diagnóstico individual de lesão.'),
    ]
    for j,(c,q,r) in enumerate(pairs):train.append(row(f'en-summary-{j}',c,q,r,'summary'))
    for j,(c,q,r) in enumerate([
      ('The review describes a correlation and states that causation was not established.','A revisão prova uma relação de causa e efeito?','Não. A revisão descreve correlação e afirma que causalidade não foi estabelecida.'),
      ('No assessment of player availability is included in the supplied excerpt.','O jogador está disponível para a partida?','O trecho fornecido não informa a disponibilidade do jogador.'),
      ('The record uses seconds for duration and metres for distance.','Quais são as unidades usadas?','O registro usa segundos para duração e metros para distância.')]):ev.append(row(f'en-heldout-summary-{j}',c,q,r,'heldout-summary'))
    assert not ({r['text'] for r in train}&{r['text'] for r in ev})
    for name,rows in [('arandu_sft.jsonl',train),('arandu_eval.jsonl',ev)]:
        assert len({r['text'] for r in rows})==len(rows)
        (a.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps(dict(train=len(train),evaluation=len(ev),new_train=len(train)-302,new_evaluation=len(ev)-10)))
if __name__=='__main__':main()
