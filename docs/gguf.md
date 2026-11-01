# Exportação GGUF nativa

```sh
python3 scripts/export_to_gguf.py --run runs/native-smoke-20261024 --output runs/native-smoke-20261024/Katu-2M-v0.1-F32.gguf
python3 -m unittest discover -s scripts -p 'test_export_to_gguf.py' -v
```

Lê `config.json`, `model.ckpt` (KATSUCK1) e `tokenizer.bpe` (KATSUBP1).
Escreve GGUF v3 F32, alinhamento 32, sem dependência Python externa. Valida
shapes e merges, rejeita truncamento e não sobrescreve arquivo existente.
Pesos são copiados byte a byte; dimensões são invertidas para a ordem GGML.

`general.architecture=katu` e `tokenizer.ggml.model=katu-bpe` são intencionais:
este modelo não é compatível automaticamente com llama.cpp/Ollama. O consumidor HTTP
precisa implementar um loader dessa arquitetura e sua segmentação original.
O arquivo guarda tokens com mapeamento reversível byte→Unicode, merges em ordem,
pares de IDs em `tokenizer.katu.merge_ids` e identificação do pretokenizador
`unicode-white-space-inclusive-v1`. Não substitua pela regex GPT-2.
Não há BOS/EOS inseridos automaticamente. Embeddings de entrada/saída são separados;
a atenção mantém projeções por cabeça, conforme o checkpoint Rust.

O checkpoint local de 2.061.152 parâmetros gerou 29 tensores, 32.000 tokens e
31.744 merges. Os testes usam um leitor independente do writer para conferir
bytes, dimensões, metadados e rejeições. Exportar preserva o modelo; não comprova
qualidade de linguagem ou execução em um runtime externo.

Referência de formato: [GGUF oficial](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
Backend opcional: [cudarc 0.19.9](https://docs.rs/cudarc/0.19.9/cudarc/).

## Tokenizer BP2

O leitor também aceita KATSUBP2: quatro tokens reservados, anexados após os merges, são preservados em `tokenizer.katu.special_ids`, `tokenizer.katu.special_tokens` e `tokenizer.ggml.eos_token_id`. A inserção automática de BOS/EOS continua desativada; o template fornece os marcadores.
