#!/usr/bin/env bash
# Compila o katsu-server (release, com CUDA) duas vezes, em duas cópias do código em caminhos
# diferentes, sem rede, e compara o SHA-256 dos binários. Iguais = quem compilar o mesmo fonte com
# o mesmo compilador obtém o mesmo binário, e pode conferir o que recebeu.
#
#   scripts/soberania/reproducible.sh    # precisa de vendor/ (vendor.sh)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f vendor/MANIFEST.sha256 ] || { echo "falta vendor/: rode scripts/soberania/vendor.sh"; exit 1; }

work=$(mktemp -d)
[ -n "${KEEP:-}" ] && echo "mantido: $work" || trap 'rm -rf "$work"' EXIT
# Pelo rustup (não o binário direto): ele aponta as ferramentas do toolchain para o libLLVM.
toolchain=$(rustup show active-toolchain | cut -d' ' -f1)
cargo() { rustup run "$toolchain" cargo "$@"; }
echo "compilador: $(rustup run "$toolchain" rustc --version)"
sysroot=$(rustup run "$toolchain" rustc --print sysroot)

build() {
  local src=$1
  mkdir -p "$src"
  # Só o que o git rastreia, mais vendor/: nada de target/, corpus ou checkpoints.
  git ls-files -z | rsync -a --from0 --files-from=- ./ "$src/"
  rsync -a vendor "$src/"
  mkdir -p "$src/.home"
  cat > "$src/.home/config.toml" <<TOML
[source.crates-io]
replace-with = "vendored"
[source.vendored]
directory = "vendor"
[net]
offline = true
TOML
  # Caminhos da máquina saem do binário: o fonte vira /katsu, o cargo home /cargo e o toolchain
  # (que traz o nome do usuário) /rust. No macOS, -Wl,-S impede o linker de gravar os caminhos dos
  # .o intermediários, que entrariam no cálculo do LC_UUID e mudariam o binário de pasta para pasta.
  local flags="--remap-path-prefix=$src=/katsu --remap-path-prefix=$src/.home=/cargo --remap-path-prefix=$sysroot=/rust"
  [ "$(uname)" = Darwin ] && flags="$flags -C link-arg=-Wl,-S"
  (cd "$src" && CARGO_HOME="$src/.home" CARGO_TARGET_DIR="$src/target" RUSTFLAGS="$flags" \
    cargo build -q --release --locked --offline -p katsu-server --features cuda)
}

build "$work/a/katsu"
build "$work/b/outro/caminho/katsu"
bins=$(cd "$work/a/katsu/target/release" && find . -maxdepth 1 -type f -perm -u+x | LC_ALL=C sort)
fail=0
for bin in $bins; do
  a=$(shasum -a 256 "$work/a/katsu/target/release/$bin" | cut -d' ' -f1)
  b=$(shasum -a 256 "$work/b/outro/caminho/katsu/target/release/$bin" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "igual     ${bin#./}  $a"; else echo "DIFERENTE ${bin#./}  $a $b"; fail=1; fi
done
for bin in $bins; do
  if strings "$work/a/katsu/target/release/$bin" | grep -qE "$HOME|$work"; then
    echo "AVISO: ${bin#./} ainda contém um caminho desta máquina"; fail=1
  fi
done
[ "$fail" = 0 ] && echo "OK: build reproduzível" || { echo "FALHA: binários diferentes"; exit 1; }
