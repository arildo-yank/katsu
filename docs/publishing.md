# Preparação da publicação

[Índice](README.md) · [Rascunhos de lançamento](launch.md)

Este documento não executa publicação e não pressupõe a existência de uma release.
A captura de tela de configurações de uma conta não comprova a existência ou a
URL de um repositório. Não inclua essa captura entre os arquivos do projeto.

## Antes do primeiro push

- [ ] Confirmar a URL e a visibilidade do repositório de destino.
- [ ] Revisar `git status` e os diffs, incluindo o índice, sem descartar trabalho existente.
- [ ] Conferir licença, README, exemplos e links locais.
- [ ] Executar os comandos do [guia de testes](testing.md).
- [ ] Verificar que target, ambientes virtuais e credenciais não estão versionados.
- [ ] Registrar as alterações revisadas em commit e publicar no remoto correto.

## Antes do anúncio

- [ ] Abrir README e SVGs no GitHub e conferir renderização.
- [ ] Aguardar o workflow; só então adicionar badge ligado à URL real do CI.
- [ ] Trocar a URL pendente nos [textos de lançamento](launch.md) pela URL pública verificada.
- [ ] Manter o título como prévia de core Rust, sem promessas não testadas.
- [ ] Não anunciar `pip install katsu`, Grad-CAM, treino ou benchmarks ainda inexistentes.

## Antes de uma release de pacotes

- [ ] Verificar disponibilidade e titularidade dos nomes nos registries.
- [ ] Validar empacotamento e instalação limpa, incluindo Python.
- [ ] Definir versões suportadas e uma política de compatibilidade.
- [ ] Atualizar changelog, criar tag coerente e registrar artefatos publicados.

Não há canal privado de segurança nem financiamento configurado nesta versão.
Não publique segredos em issues. Se o GitHub oferecer relato privado no repositório,
use-o para uma vulnerabilidade sensível; caso contrário, solicite um contato privado
sem expor detalhes exploráveis. O mantenedor deve definir esse canal antes de adoção sensível.
