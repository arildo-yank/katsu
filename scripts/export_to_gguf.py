#!/usr/bin/env python3
"""KATSUCK1 + KATSUBP1 -> GGUF v3/F32, arquitetura customizada katu.
Não anuncia compatibilidade com llama.cpp: Arandu precisa de um loader Katu.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile

U32, U64, F32, BOOL, STRING, ARRAY = 4, 10, 6, 7, 8, 9
ALIGNMENT = 32


def exact(file, size):
    data = file.read(size)
    if len(data) != size:
        raise ValueError('Arquivo truncado')
    return data


def unpack(file, fmt):
    return struct.unpack('<' + fmt, exact(file, struct.calcsize('<' + fmt)))


def byte_alphabet():
    # Mapeamento reversível convencional byte->Unicode, sem U+FFFD.
    mapped = {b: chr(b) for b in range(33, 127)}
    mapped.update({b: chr(b) for b in range(161, 173)})
    mapped.update({b: chr(b) for b in range(174, 256)})
    next_code = 256
    for b in range(256):
        if b not in mapped:
            mapped[b] = chr(next_code)
            next_code += 1
    return mapped


def load_bpe(path):
    tokens = [bytes([i]) for i in range(256)]
    merges, seen = [], set()
    with Path(path).open('rb') as file:
        magic = exact(file, 8)
        if magic not in (b'KATSUBP1', b'KATSUBP2'):
            raise ValueError('Tokenizer precisa ser KATSUBP1 ou KATSUBP2')
        count, = unpack(file, 'I')
        if count > 65536 - 256:
            raise ValueError('Vocabulário excede 65536 tokens')
        total = 256
        for _ in range(count):
            a, b = unpack(file, 'II')
            if a >= len(tokens) or b >= len(tokens) or (a, b) in seen:
                raise ValueError('Merge inválido ou duplicado')
            size = len(tokens[a]) + len(tokens[b])
            total += size
            if size > 1_048_576 or total > 64 * 1024**2:
                raise ValueError('Vocabulário expandido excede limite')
            tokens.append(tokens[a] + tokens[b])
            merges.append((a, b))
            seen.add((a, b))
        if magic == b'KATSUBP2':
            if len(tokens)+4>65536: raise ValueError('Sem espaço para tokens especiais')
            tokens.extend(x.encode() for x in ('<|system|>','<|user|>','<|assistant|>','<|end|>'))
        if file.read(1):
            raise ValueError('Bytes extras após tokenizer')
    return tokens, merges


def layout(config):
    v, t, d, layers, heads = (config[k] for k in ('vocab_size', 'block_size', 'embed_dim', 'layers', 'heads'))
    if any(type(n) is not int or n <= 0 for n in (v, t, d, heads)) or type(layers) is not int or layers < 0 or d % heads:
        raise ValueError('Config inválida')
    yield 'token_embd.weight', [v, d]
    yield 'position_embd.weight', [t, d]
    for layer in range(layers):
        base = f'blk.{layer}'
        yield base + '.attn_norm.weight', [d]
        yield base + '.attn_norm.bias', [d]
        for h in range(heads):
            for name in ('q', 'k', 'v'):
                yield f'{base}.head.{h}.attn_{name}.weight', [d, d // heads]
            yield f'{base}.head.{h}.attn_output.weight', [d // heads, d]
        yield base + '.ffn_norm.weight', [d]
        yield base + '.ffn_norm.bias', [d]
        yield base + '.ffn_up.weight', [d, 4 * d]
        yield base + '.ffn_up.bias', [4 * d]
        yield base + '.ffn_down.weight', [4 * d, d]
        yield base + '.ffn_down.bias', [d]
    yield 'output_norm.weight', [d]
    yield 'output_norm.bias', [d]
    yield 'output.weight', [d, v]


def checkpoint_index(path, config):
    tensors = []
    size = Path(path).stat().st_size
    expected = list(layout(config))
    with Path(path).open('rb') as file:
        if exact(file, 8) != b'KATSUCK1':
            raise ValueError('Checkpoint precisa ser KATSUCK1')
        count, = unpack(file, 'Q')
        if count != len(expected):
            raise ValueError('Número de tensores difere da configuração')
        for name, shape in expected:
            rank, = unpack(file, 'Q')
            if rank != len(shape) or rank > 4:
                raise ValueError(f'Rank inválido em {name}')
            dims = list(unpack(file, 'Q' * rank))
            if dims != shape:
                raise ValueError(f'Shape inválido em {name}: {dims} != {shape}')
            nbytes = math.prod(dims) * 4
            offset = file.tell()
            if offset + nbytes > size:
                raise ValueError('Dados de tensor truncados')
            tensors.append((name, dims, offset, nbytes))
            file.seek(nbytes, 1)
        if file.tell() != size:
            raise ValueError('Bytes extras após checkpoint')
    return tensors


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def string(value):
    data = value.encode('utf-8')
    return struct.pack('<Q', len(data)) + data


def pack(kind, value):
    if kind == STRING:
        return string(value)
    if kind == ARRAY:
        subtype, values = value
        return struct.pack('<IQ', subtype, len(values)) + b''.join(pack(subtype, x) for x in values)
    return struct.pack({U32: '<I', U64: '<Q', F32: '<f', BOOL: '<?'}[kind], value)


def export(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise ValueError('Saída já existe; escolha um novo arquivo')
    config = json.loads((run / 'config.json').read_text())
    tokens, merges = load_bpe(run / 'tokenizer.bpe')
    if config['vocab_size'] != len(tokens):
        raise ValueError('Vocabulário e configuração não correspondem')
    tensors = checkpoint_index(run / 'model.ckpt', config)
    alphabet = byte_alphabet()
    texts = [''.join(alphabet[b] for b in token) for token in tokens]
    metadata = {
        'general.architecture': (STRING, 'katu'),
        'general.name': (STRING, 'Katu native'),
        'general.alignment': (U32, ALIGNMENT),
        'general.file_type': (U32, 0),
        'katu.format_version': (U32, 1),
        'katu.tensor_layout': (STRING, 'per-head-row-major-v1'),
        'katu.context_length': (U32, config['block_size']),
        'katu.embedding_length': (U32, config['embed_dim']),
        'katu.block_count': (U32, config['layers']),
        'katu.attention.head_count': (U32, config['heads']),
        'katu.attention.layer_norm_epsilon': (F32, 1e-5),
        'katu.source_checkpoint_sha256': (STRING, sha256(run / 'model.ckpt')),
        'katu.source_tokenizer_sha256': (STRING, sha256(run / 'tokenizer.bpe')),
        'tokenizer.ggml.model': (STRING, 'katu-bpe'),
        'tokenizer.ggml.tokens': (ARRAY, (STRING, texts)),
        'tokenizer.ggml.merges': (ARRAY, (STRING, [texts[a] + ' ' + texts[b] for a, b in merges])),
        'tokenizer.ggml.add_bos_token': (BOOL, False),
        'tokenizer.ggml.add_eos_token': (BOOL, False),
        'tokenizer.katu.pre': (STRING, 'unicode-white-space-inclusive-v1'),
        'tokenizer.katu.encoding': (STRING, 'gpt2-byte-to-unicode'),
        'tokenizer.katu.merge_ids': (ARRAY, (U32, [i for pair in merges for i in pair])),
    }
    base_vocab=256+len(merges)
    if len(tokens)==base_vocab+4:
        metadata['tokenizer.katu.special_ids']=(ARRAY,(U32,list(range(base_vocab,base_vocab+4))))
        metadata['tokenizer.katu.special_tokens']=(ARRAY,(STRING,['<|system|>','<|user|>','<|assistant|>','<|end|>']))
        metadata['tokenizer.ggml.eos_token_id']=(U32,base_vocab+3)
    header = bytearray(struct.pack('<4sIQQ', b'GGUF', 3, len(tensors), len(metadata)))
    for key, (kind, value) in metadata.items():
        header += string(key) + struct.pack('<I', kind) + pack(kind, value)
    offset = 0
    for name, dims, _, nbytes in tensors:
        # GGML dimension 0 is contiguous: reverse logical row-major shape.
        header += string(name) + struct.pack('<I', len(dims))
        header += struct.pack('<' + 'Q' * len(dims), *reversed(dims))
        header += struct.pack('<IQ', 0, offset)
        offset += (nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    header += b'\0' * (-len(header) % ALIGNMENT)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.katu-gguf-', delete=False) as out:
            temporary = Path(out.name)
            out.write(header)
            with (run / 'model.ckpt').open('rb') as source:
                for _, _, start, nbytes in tensors:
                    source.seek(start)
                    remaining = nbytes
                    while remaining:
                        chunk = exact(source, min(1024**2, remaining))
                        out.write(chunk)
                        remaining -= len(chunk)
                    out.write(b'\0' * (-nbytes % ALIGNMENT))
            out.flush()
            os.fsync(out.fileno())
        # Commit atômico sem sobrescrever um arquivo criado por outra execução.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'output': str(output), 'tensors': len(tensors), 'vocab_size': len(tokens), 'sha256': sha256(output),
            'architecture': 'katu', 'requires_katu_loader': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output), indent=2))


if __name__ == '__main__':
    main()
