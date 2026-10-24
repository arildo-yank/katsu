#!/usr/bin/env python3
"""Dimensiona os perfis sem alocar pesos. Nunca inicia treino automaticamente."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from export_to_gguf import load_bpe


def host_memory():
    if platform.system() == 'Linux':
        fields={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
        return fields['MemTotal'],fields['MemAvailable']
    if platform.system() == 'Darwin':
        total=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True).strip())
        pressure=subprocess.check_output(['memory_pressure','-Q'],text=True)
        match=re.search(r'free percentage:\s*(\d+)%',pressure)
        if not match:raise ValueError('Não foi possível medir memória disponível')
        return total,total*int(match[1])/100
    raise ValueError('Preflight implementado para Linux e macOS')


def estimate(vocab,profile,batch,sequence):
    d,layers,heads={'139m':(768,13,12),'349m':(1024,22,16)}[profile]
    params=2*vocab*d+512*d+layers*(12*d*d+9*d)+2*d
    return {'profile':profile,'parameters':params,'batch':batch,'sequence':sequence,
            'estimated_host_bytes':params*32+batch*sequence*d*layers*64+batch*heads*sequence**2*layers*16+batch*sequence*vocab*16}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--sequence-length',type=int,default=16)
    args=p.parse_args()
    if args.batch_size<1 or not 1<=args.sequence_length<=512:raise ValueError('Batch/sequência inválidos')
    vocab=len(load_bpe(args.tokenizer)[0]);total,available=host_memory()
    budget=int(available*0.7)
    plans=[estimate(vocab,profile,args.batch_size,args.sequence_length) for profile in ('139m','349m')]
    for row in plans:
        row['host_memory_check']='fits_estimate' if row['estimated_host_bytes']<=budget else 'insufficient_headroom'
        row['training_executed']=False
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],text=True).strip()
    except (FileNotFoundError,subprocess.CalledProcessError):gpu=None
    report={'platform':platform.platform(),'total_host_bytes':total,'available_host_bytes':int(available),
            'planning_budget_bytes':budget,'gpu':gpu,'plans':plans,
            'note':'Estimativa conservadora, não garantia de pico. Nenhum peso foi alocado.'}
    with Path(args.output).open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
