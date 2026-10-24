"""Fail closed if the sealed corpus, tokenizer, or SFT files have changed."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
seal = json.loads((root / 'runs/bpe-arandu-v1/data-seal.json').read_text())
for name, expected in seal['files'].items():
    path = root / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Data gate failed: {name}')
if not seal['checks_passed']:
    raise SystemExit('Data checks did not pass')
if '--selected' in sys.argv:
    for key, expected in seal['selected'].items():
        if Path(os.environ[key]) != Path(expected):
            raise SystemExit(f'{key} differs from the audited data: regenerate the seal before use')
print('Data gate passed: bilingual corpus, Rust BPE 32000, cross-lingual SFT and SHA-256.')
