# Corpus do Katsu LM

[Índice](README.md)

O Katsu LM é treinado só com **literatura brasileira em domínio público**. Os textos
vêm do Project Gutenberg e não são versionados no git (`/corpus/` está no `.gitignore`).
Para reproduzir o corpus exatamente:

```sh
scripts/fetch-corpus.sh
```

## Obras (22, ~7,5 MB, ~6,8 milhões de caracteres após limpeza)

| Autor | Obras | Gutenberg |
| --- | --- | --- |
| Machado de Assis (1839–1908) | Dom Casmurro, Memórias Póstumas de Brás Cubas, Quincas Borba, Esaú e Jacó, Memorial de Aires, Helena, A Mão e a Luva, Iaiá Garcia, Histórias sem Data, Papéis Avulsos | 55752, 54829, 55682, 56737, 55797, 67162, 53101, 67780, 33056, 57001 |
| José de Alencar (1829–1877) | Iracema, O Guarani (2 volumes), Ubirajara, Cinco Minutos, A Pata da Gazela | 67740, 67724, 67725, 38496, 44540, 67831 |
| Aluísio Azevedo (1857–1913) | O Cortiço | 69187 |
| Lima Barreto (1881–1922) | Triste Fim de Policarpo Quaresma | 67535 |
| Bernardo Guimarães (1825–1884) | A Escrava Isaura | 74475 |
| Raul Pompeia (1863–1895) | O Ateneu | 68541 |
| Visconde de Taunay (1843–1899) | Inocência, Histórias Brasileiras | 68635, 62525 |

Todos os autores morreram há mais de 70 anos; as obras estão em domínio público no
Brasil (Lei 9.610/98, art. 41). O Project Gutenberg distribui os arquivos sob a sua
licença; o exemplo `katsu_lm` remove o cabeçalho e o rodapé dessa licença antes do treino.
Ficaram de fora traduções para outros idiomas (por exemplo, *Brazilian Tales*).

## Características

- As edições usam a **ortografia da época** ("elle", "cousa", "pharmacia"). O modelo
  aprende essa grafia; um corpus com ortografia atual exigiria outras fontes.
- O vocabulário de caracteres tem ~135 símbolos (letras acentuadas, pontuação, numerais
  romanos dos capítulos).
- Validação: os últimos 10% de cada obra, para todas estarem representadas.
