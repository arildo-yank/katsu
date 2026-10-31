#!/usr/bin/env bash
# Baixa o código-fonte de todas as dependências (Cargo.lock) para vendor/ e grava o hash de cada
# arquivo. É o único passo que usa a rede; depois dele, offline.sh compila e testa sem internet.
#
#   scripts/soberania/vendor.sh
#
# vendor/MANIFEST.sha256 lista o SHA-256 de cada arquivo, em ordem fixa. O hash desse manifesto
# identifica o código de terceiros inteiro: duas máquinas que baixarem o mesmo Cargo.lock obtêm o
# mesmo número, e um auditor pode conferir arquivo por arquivo.
set -euo pipefail
cd "$(dirname "$0")/../.."

rm -rf vendor
cargo vendor --quiet --locked --versioned-dirs vendor >/dev/null
(cd vendor && find . -type f ! -name MANIFEST.sha256 -print0 | LC_ALL=C sort -z \
  | xargs -0 shasum -a 256 > MANIFEST.sha256)
echo "vendor/: $(find vendor -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') crates, $(du -sh vendor | cut -f1)"
echo "manifesto: $(shasum -a 256 vendor/MANIFEST.sha256 | cut -d' ' -f1)"
