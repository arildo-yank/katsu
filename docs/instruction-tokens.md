# Tokens de instrução

BPE KATSUBP2 reserva `<|system|>`, `<|user|>`, `<|assistant|>` e `<|end|>`.
No vocabulário de 32000 tokens básicos, os IDs são 32000–32003. São atômicos.
O servidor interrompe a geração em `<|end|>` e omite o marcador da resposta.
O SFT calcula loss nos tokens de resposta e fim, excluindo o prompt.

A construção independente usa `scripts/build_sft_arandu.py`; exemplos bilíngues
usam `scripts/build_bilingual_sft.py`. Não há dependência de aplicação externa.
O marcador não garante correção factual: avaliar geração livre separadamente.
