# Pacote de soberania

[Índice](../README.md) · [Posicionamento](../posicionamento.md)

Quatro verificações que transformam "o Katsu é auditável e roda sem internet" em algo que um
auditor confere sozinho. Todas rodam no Mac ou em Linux, sem custo.

| Verificação | Script | Resultado em 31/10/2026 (Mac M3, Rust 1.99.1) |
|---|---|---|
| Lista de componentes (SBOM) | `scripts/soberania/sbom.py` | 69 crates de terceiros, todas com licença permissiva e hash |
| Código de terceiros guardado com hash | `scripts/soberania/vendor.sh` | 71 crates, 78 MB; hash do manifesto `ed22f127c91a9dc7…` |
| Compilar e testar com a rede cortada | `scripts/soberania/offline.sh` | **passou**: `curl` bloqueado; testes de core, rag e server, build com CUDA e exemplo de treino, em 45 s |
| Build reproduzível | `scripts/soberania/reproducible.sh` | **passou**: os 4 binários (`katsu-server`, `train_sft`, `tokenize_corpus`, `train_bilingual_bpe`) idênticos e sem caminho da máquina |

## 1. Lista de componentes: `sbom.json`

[`sbom.json`](sbom.json), formato CycloneDX 1.5: cada crate de terceiros com versão, licença, o
SHA-256 do `Cargo.lock` e quais crates do Katsu a usam. Gerado só do `Cargo.lock`, sem data: o
mesmo lock gera o mesmo arquivo. O CI roda `sbom.py --check` e falha se alguém mudar uma
dependência sem atualizar a lista.

Contagem (dependências normais, com todas as features):

| Crate do Katsu | Crates de terceiros |
|---|---:|
| `katsu-core` sem CUDA | 6 |
| `katsu-core` com CUDA | 10 (`cudarc`, `libloading`, `cfg-if`, `windows-link` a mais) |
| `katsu-rag` | 11 |
| `katsu-python` | 22 (PyO3) |
| `katsu-server` | 62 (`tokio`, `axum`, `hyper`, `serde`) |

Licenças: todas permissivas (MIT, Apache-2.0, BSD-3-Clause, Unicode-3.0, Unlicense, BSL-1.0).
Nenhuma GPL.

## 2. Código de terceiros: `vendor/`

`vendor.sh` baixa o fonte de cada dependência para `vendor/` (fora do git) e grava
`vendor/MANIFEST.sha256`, o hash de cada arquivo em ordem fixa. É o único passo que usa a rede. O
hash do manifesto identifica todo o código de terceiros: o órgão baixa uma vez, confere e guarda.

O `vendor/` tem 2 crates a mais que o SBOM (71 contra 69): são dependências só de compilação
(`pyo3-build-config`, `target-lexicon`), que não entram no binário.

## 3. Rede cortada: `offline.sh`

Compila e testa com um `CARGO_HOME` vazio e temporário, de modo que nada vem do cache da máquina, só
de `vendor/`. A rede fica cortada pelo sistema: no macOS, `sandbox-exec` nega toda conexão menos
`localhost` (os testes do servidor usam a porta local); no Linux, `unshare -rn`. Antes de compilar,
o script prova o corte: um `curl` para crates.io de dentro do isolamento precisa falhar.

Roda os testes de `katsu-core`, `katsu-rag` e `katsu-server`, compila o servidor com CUDA e treina o
exemplo `train`.

## 4. Build reproduzível: `reproducible.sh`

Compila o `katsu-server` (release, com CUDA) duas vezes, em duas cópias do código em caminhos
diferentes, sem rede, e compara o SHA-256 dos binários. Os caminhos da máquina saem do binário com
`--remap-path-prefix` (fonte, cargo home e toolchain, que traria o nome do usuário).

No macOS, a primeira tentativa deu binários diferentes em 48 bytes: código e dados iguais, só o
`LC_UUID` e a assinatura que o cobre. O linker calcula o UUID incluindo os caminhos dos `.o`
intermediários. `-C link-arg=-Wl,-S` (não gravar esses caminhos) resolveu. O script também falha
se encontrar o `$HOME` ou a pasta temporária dentro de um binário.

## Limites

- **O compilador** (rustc) vem do projeto Rust. É aberto, mas o pacote verifica o Katsu e as
  dependências, não o compilador. Reproduzir o build exige a mesma versão do rustc.
- **O driver CUDA** é da NVIDIA, fechado. O Katsu o carrega em tempo de execução
  (`dynamic-loading`); compilar com CUDA não exige o driver, rodar exige.
- **Reproduzível foi verificado na mesma máquina**, em caminhos diferentes. Duas máquinas
  diferentes, ou macOS contra Linux, ainda não foram comparadas.
