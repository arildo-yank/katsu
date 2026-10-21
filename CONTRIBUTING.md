# Contribuindo com Katsu

Obrigado por ajudar a construir o core. Issues e pull requests podem ser escritas
em português ou inglês. English contributions are welcome.

## Comece por aqui

Leia o [README](README.md), a [arquitetura](docs/architecture.md) e o
[roadmap](docs/roadmap.md). A API ainda está em desenvolvimento; proponha mudanças
pequenas, com exemplos que mostrem o problema e o resultado esperado.

## Ambiente e verificações

Use Rust stable, rustfmt e Clippy. Na raiz do workspace:

```sh
cargo fmt --all -- --check
cargo test -p katsu-core
cargo clippy -p katsu-core --all-targets -- -D warnings
```

Para rodar essas verificações a cada commit, ative o hook versionado:

```sh
git config core.hooksPath .githooks
```

Para alterações em Conv2D, execute também o exemplo em release. Se as dependências
não puderem ser resolvidas, use a [validação offline](docs/testing.md) e informe
exatamente o que foi e não foi validado. Não adicione dependências ao core sem
explicar a necessidade e o impacto.

## Operações e gradientes

- Descreva shapes aceitos, layouts, erros e comportamento para dimensões zero.
- Inclua um caso numérico independente e pequeno que possa ser calculado à mão.
- Para novas derivadas, compare com diferenças finitas e teste operandos compartilhados.
- Verifique acumulação em backward repetido e preserve a identidade dos nós.
- Mantenha dados usados pelo backward imutáveis ou defina explicitamente um novo contrato.
- O crate proíbe unsafe. Mudanças nessa política exigem uma proposta separada.

## Documentação e SVG

Atualize a referência quando uma assinatura ou semântica mudar. Exemplos devem
compilar com a API atual. Nos SVGs de [docs/assets](docs/assets), preserve viewBox,
title/desc e contraste. Não inclua scripts, rastreadores, fontes remotas ou imagens
externas. Não use badges de CI aprovado sem uma execução real.

## Pull requests

Explique o problema, a mudança observável, os testes executados e as limitações.
Use o [template](.github/pull_request_template.md). Não inclua credenciais, imagens
de configurações pessoais, dados de terceiros ou artefatos de build.
Mudanças em dependências e arquivos de lock devem ter contexto explícito.

## Conduta e licença

Seja respeitoso, discuta código e evidências, e mantenha dados pessoais fora das
conversas públicas. Não são aceitos assédio ou discriminação.
Ao contribuir, você concorda em disponibilizar sua contribuição sob a
[licença MIT](LICENSE) do projeto. Não há CLA adicional configurado.
