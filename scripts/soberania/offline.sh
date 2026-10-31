#!/usr/bin/env bash
# Compila e testa o Katsu com a rede cortada, a partir de vendor/ (rode vendor.sh antes).
#
#   scripts/soberania/offline.sh
#
# - Cargo com CARGO_HOME vazio e temporário: nada vem do cache da máquina, só de vendor/.
# - macOS: sandbox-exec nega toda rede, menos 127.0.0.1 (os testes do servidor sobem na porta local).
#   Linux: unshare -rn cria um namespace sem rede além do loopback.
# - Antes de tudo, prova que a rede está mesmo cortada: um curl de dentro do isolamento tem de falhar.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f vendor/MANIFEST.sha256 ] || { echo "falta vendor/: rode scripts/soberania/vendor.sh"; exit 1; }
(cd vendor && shasum -a 256 -c --quiet MANIFEST.sha256) || { echo "vendor/ não bate com o manifesto"; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export CARGO_HOME="$work/cargo-home" CARGO_TARGET_DIR="$work/target"
mkdir -p "$CARGO_HOME"
cat > "$CARGO_HOME/config.toml" <<TOML
[source.crates-io]
replace-with = "vendored"
[source.vendored]
directory = "$PWD/vendor"
[net]
offline = true
TOML
# Pelo rustup (não o binário direto): ele aponta as ferramentas do toolchain para o libLLVM.
toolchain=$(rustup show active-toolchain | cut -d' ' -f1)

case "$(uname)" in
  Darwin)
    profile='(version 1)(allow default)(deny network*)(allow network* (remote ip "localhost:*"))(allow network* (local ip "localhost:*"))'
    isolate() { sandbox-exec -p "$profile" "$@"; } ;;
  Linux) isolate() { unshare -rn sh -c 'ip link set lo up 2>/dev/null; exec "$@"' sh "$@"; } ;;
  *) echo "sistema sem isolamento de rede conhecido"; exit 1 ;;
esac

if isolate curl -sS -m 5 -o /dev/null https://crates.io 2>/dev/null; then
  echo "FALHA: a rede não está cortada dentro do isolamento"; exit 1
fi
echo "rede cortada: curl para crates.io falhou dentro do isolamento, como esperado"

run() { echo "+ $*"; isolate rustup run "$toolchain" cargo "$@" --locked --offline; }
run test -p katsu-core -q
run test -p katsu-rag -q
run test -p katsu-server -q
run build -p katsu-server --features cuda -q
run run -p katsu-core --example train -q > "$work/train.out"
tail -1 "$work/train.out"
echo "OK: compilado e testado sem rede, só com vendor/"
