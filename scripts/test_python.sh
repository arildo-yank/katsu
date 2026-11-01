#!/bin/sh
# Compila a ponte Python (katu-python) e roda python/tests, sem maturin e sem instalar nada.
#   PYTHON=python3.13 sh scripts/test_python.sh
# Para gerar o wheel instalável, use o maturin (docs/MANUAL_KATU_PYTHON.md).
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
case "$(uname)" in Darwin) export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-undefined -C link-arg=dynamic_lookup" ;; esac
TARGET="${CARGO_TARGET_DIR:-target}/python-ext"
PYO3_PYTHON="$(command -v "$PYTHON")" CARGO_TARGET_DIR="$TARGET" cargo build -p katu-python --release
for lib in "$TARGET"/release/lib_native.dylib "$TARGET"/release/lib_native.so; do
  [ -f "$lib" ] && cp "$lib" python/katu/_native.abi3.so
done
cd python && "$PYTHON" -m unittest discover -s tests -v
