#!/bin/sh
# Baixa o corpus do Katsu LM: literatura brasileira em domínio público (Project Gutenberg).
# Uso: scripts/fetch-corpus.sh [pasta]   (padrão: corpus/)
set -eu
dest="${1:-corpus}"
mkdir -p "$dest"
while read -r id name; do
  [ -s "$dest/$name.txt" ] && continue
  curl -sSfL -o "$dest/$name.txt" "https://www.gutenberg.org/cache/epub/$id/pg$id.txt"
  echo "ok $name"
  sleep 1 # sem sobrecarregar o Gutenberg
done <<'LIST'
55752 machado-dom-casmurro
54829 machado-memorias-postumas
55682 machado-quincas-borba
56737 machado-esau-e-jaco
55797 machado-memorial-de-aires
67162 machado-helena
53101 machado-a-mao-e-a-luva
67780 machado-iaia-garcia
33056 machado-historias-sem-data
57001 machado-papeis-avulsos
67740 alencar-iracema
67724 alencar-o-guarani-1
67725 alencar-o-guarani-2
38496 alencar-ubirajara
44540 alencar-cinco-minutos
67831 alencar-a-pata-da-gazela
69187 azevedo-o-cortico
67535 barreto-policarpo-quaresma
74475 guimaraes-a-escrava-isaura
68541 pompeia-o-ateneu
68635 taunay-inocencia
62525 taunay-historias-brasileiras
LIST
