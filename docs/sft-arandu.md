# SFT independente do Arandu

O dataset contém exemplos sintéticos locais; nenhum backend de outro projeto
é acessado. Gere o dataset base com `python3 scripts/build_sft_arandu.py`.
Arquivos: `data/sft/arandu_sft.jsonl` e `data/sft/arandu_eval.jsonl`.
A recusa usa o contexto fornecido, sem citar uma aplicação externa.

O treinador `train_sft.py` aceita `--train` e `--eval`. Para a versão bilíngue,
use `data/sft-bilingual-v2/` e o tokenizer `runs/bpe-bilingual-v2/tokenizer.bpe`.
Tokenizers com merges diferentes não podem ser trocados em checkpoints antigos.

O serviço Rust oferece /health e /generate em loopback. A avaliação livre é feita
por `scripts/eval_sft_http.py`, sem qualquer cliente de outro repositório.
Resultados anteriores foram arquivados quando continham a integração removida;
a alteração de nomes e prompts não significa que modelos antigos foram retreinados.
