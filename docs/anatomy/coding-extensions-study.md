# Vulcano: extensões, registro de tools e seleção pela CLI

Estudo em 2026-10-02. O registro de tools por CodingExtensions e a seleção pela CLI foram
implementados nesta branch após aprovação do plano. A ponte de AgentExtensions
para o harness continua uma proposta posterior. A revisão corrente da TUI permanece
no worktree `/tmp/msgflux-coding-resume`, branch `feat/coding-resume`.

**Referência principal: Pi v1.** Seu contrato de extensão, montagem do harness e
UX de seleção pela CLI orientam a proposta. Tau é uma referência secundária de
implementação em Python; suas ausências de funcionalidades não limitam o desenho
do Vulcano. As adaptações ao msgFlux devem reutilizar o runtime existente e
explicitar diferenças de comportamento, como seleção pelo perfil e factories
de tools por sessão.

## Referências verificadas

Foram clonados os repositórios oficiais e três subagentes Luna estudaram Pi,
Tau e o encaixe no runtime do msgFlux. A análise foi de código e documentação;
não executamos extensões externas nem instalamos os harnesses.

| Projeto | Referência analisada | Checkout local |
| --- | --- | --- |
| Pi | `main` em `9fba660cf1caca0ade5bea72269352416e595a19`; release `v1.0.0` em `a13d35a742c6ef8462812a28fbe1d8c8b7431c32` | `/tmp/msgflux-pi-extension-study` |
| Tau | `v0.4.7` / `e4eab0dc5d6c7a92dc40e660087e55f0d5929eba` | `/tmp/msgflux-tau-extension-study` |
| msgFlux | `feat/coding-resume`, commit `5d9bd5a9b43101485d4738b2fcd53658f3805aad` com alterações da revisão ainda sem commit | `/tmp/msgflux-coding-resume` |

A [release Pi v1.0.0](https://github.com/earendil-works/pi/releases/tag/v1.0.0)
foi publicada em 2026-10-01. Comparamos a tag com a main nos arquivos de API de
extensão, loader, sessão, SDK, resource loader, documentação de extensões e
parser da CLI: só o tratamento de padrões vazios em `--models` mudou nesses
arquivos. Os comportamentos de tools descritos abaixo também estão na v1.
A [release Tau v0.4.7](https://github.com/huggingface/tau/releases/tag/v0.4.7)
foi publicada em 2026-10-01.

## O que Pi e Tau fazem

### Pi

A API pública se chama `ExtensionAPI`. Uma função recebe `pi` e usa métodos
como `registerTool`, `registerCommand` e `on`. O loader espera factories
síncronas ou assíncronas e descarta registros de uma tentativa que falhou.
Isso oferece uma entrada simples para compor o harness.

Fontes: [contrato e exemplos](https://github.com/earendil-works/pi/blob/9fba660cf1caca0ade5bea72269352416e595a19/packages/coding-agent/docs/extensions.md),
[loader](https://github.com/earendil-works/pi/blob/9fba660cf1caca0ade5bea72269352416e595a19/packages/coding-agent/src/core/extensions/loader.ts).

`--tools` é uma allowlist de nomes que cobre builtins, tools de extensões e
custom tools. `--exclude-tools` remove nomes depois dessa seleção. Também há
`--no-tools` e `--no-builtin-tools`. O carregamento de extensões é uma escolha
separada, por `--extension` e discovery. Uma extensão carregada pode executar
hooks mesmo que uma de suas tools não esteja ativa.

Fontes: [CLI](https://github.com/earendil-works/pi/blob/9fba660cf1caca0ade5bea72269352416e595a19/packages/coding-agent/docs/cli.md),
[parser](https://github.com/earendil-works/pi/blob/9fba660cf1caca0ade5bea72269352416e595a19/packages/coding-agent/src/cli/args.ts),
[SDK](https://github.com/earendil-works/pi/blob/9fba660cf1caca0ade5bea72269352416e595a19/packages/coding-agent/src/core/sdk.ts).

Pi distingue `direct`, `model-only`, `codemode`, `deferred` e `hidden`.
`direct` e `model-only` normalmente ativam no registro; `defaultActive: false`
pode impedir isso. Há seleção posterior por `setActiveTools`. Algumas tools
não declaradas ao modelo continuam invocáveis por outras tools. Assim, seu
conjunto ativo descreve exposição, não substitui autorização. Tools custom
podem substituir builtins com o mesmo nome.

Fontes: [exposição de tools](https://github.com/earendil-works/pi/blob/9fba660cf1caca0ade5bea72269352416e595a19/packages/coding-agent/docs/extensions.md#tool-exposure),
[montagem da sessão](https://github.com/earendil-works/pi/blob/9fba660cf1caca0ade5bea72269352416e595a19/packages/coding-agent/src/core/agent-session.ts).

### Tau

A entrada Python é `setup(tau: ExtensionAPI)`. Ela oferece `register_tool`,
`register_command` e `on`. `register_tool` recebe um `AgentTool` já construído,
com executor assíncrono; não é um protocolo de registro de classes para
instanciação posterior. Hooks recebem contexto na chamada.

Fontes: [API](https://github.com/huggingface/tau/blob/e4eab0dc5d6c7a92dc40e660087e55f0d5929eba/src/tau_coding/extensions/api.py),
[AgentTool](https://github.com/huggingface/tau/blob/e4eab0dc5d6c7a92dc40e660087e55f0d5929eba/src/tau_agent/tools.py).

Tau cria um runtime de extensões para a sessão e mantém identidade da origem dos
registros. Reload prepara outro runtime; referências antigas passam a ser
inválidas. Colisões entre extensões geram diagnóstico, enquanto uma tool de
extensão pode substituir uma builtin durante a composição.

Fontes: [runtime](https://github.com/huggingface/tau/blob/e4eab0dc5d6c7a92dc40e660087e55f0d5929eba/src/tau_coding/extensions/runtime.py),
[sessão](https://github.com/huggingface/tau/blob/e4eab0dc5d6c7a92dc40e660087e55f0d5929eba/src/tau_coding/session.py).

A CLI carrega extensões por `-e/--extension`, com controles de discovery. Não
há `--tools` equivalente no commit estudado. A documentação também distingue
capacidades presentes de funcionalidades ainda não suportadas, como registro
de novas flags por extensões.

Fontes: [CLI](https://github.com/huggingface/tau/blob/e4eab0dc5d6c7a92dc40e660087e55f0d5929eba/src/tau_coding/cli.py),
[guia de extensões](https://github.com/huggingface/tau/blob/e4eab0dc5d6c7a92dc40e660087e55f0d5929eba/website/content/guides/extensions.md).

## Escopo aprovado

Implementar registro de tools por CodingExtensions e seleção por perfil/CLI.
Os únicos modos serão `active` e `deferred`. As modalidades adicionais do Pi
acima são referência de pesquisa, não APIs a implementar. Mensagens queued por
AgentInbox, codemode e novos backends de workspace ficam para etapas posteriores.
A CLI já chama `open_coding_workspace`; esse helper hoje escolhe LocalWorkspace.

## Decisões recomendadas para o Vulcano

1. Ampliar `CodingExtensions`, que já registra comandos e painéis. O parâmetro
   da função pode ser `c`; seu nome é escolha do autor.
2. Registrar fábricas de tools, incluindo classes, e instanciar somente as
   selecionadas, uma vez por sessão. Uma thread nova continua em memória até
   a primeira mensagem.
3. Usar os mesmos nomes em perfil e CLI, para builtins e tools de extensões.
   Registro torna uma tool selecionável; perfil ou CLI a ativa ou a difere.
4. Reutilizar `ToolLibrary`, compilador, `tool_search`, injeção declarada,
   eventos, approvals e permissões existentes. O registro de fábricas do
   coding não é um segundo catálogo runtime nem um gerador de schemas.
5. Recusar colisões e nomes desconhecidos com origem e nomes disponíveis na
   mensagem de erro. Não substituir builtins silenciosamente.
6. Carregar as declarações antes de resolver a seleção, em TUI e `--print`.
   Manter inicialmente o entry point existente `--extension MODULE:FUNCTION`.
7. Documentar `AgentExtension.tools()` para quem usa o Agent diretamente.
   Hooks e estado de AgentExtension continuam no lifecycle atual do Agent.

Não é necessário adicionar agora os modos `codemode`/`model-only`, discovery de
arquivos do projeto, instalador de plugins, flags arbitrárias de extensões,
reload ou troca da seleção durante uma execução. Esses itens têm contratos
próprios e não são requisitos para registrar e selecionar tools.

## API de CodingExtensions

```python
# meu_pacote/vulcano.py
from msgflux.coding.extensions import CodingExtensions


class UppercaseTool:
    name = "uppercase"

    def __call__(self, text: str) -> str:
        """Return the supplied text in uppercase."""
        return text.upper()


def register(c: CodingExtensions):
    c.register_tool(UppercaseTool)
    c.register_command("hello", lambda args: f"Hello {args}")
```

Contrato implementado (simplificado em 2026-10-04):

```python
register_tool(tool) -> RegistrationHandle
```

- Aceitar função sync/async, classe ou instância callable. Não repetir nome ou
  descrição em parâmetros do registro: usar os metadados nativos da tool.
- Registro não executa funções nem constrói classes. Classes selecionadas são
  construídas por sessão; funções e objetos fornecidos pertencem ao chamador.
- Compilar pelo caminho existente da ToolLibrary, preservando schema, descrição
  e injeção declarada, como `Hidden[AgentWorkspace]` com `runtime_inputs`.
- Guardar `ToolSpec` em `msgspec.Struct` em `extensions/records.py`; registro em
  `registry.py`; `__init__.py` apenas reexporta.
- O handle remove a declaração para composições futuras, preservando ownership
  e rollback do callback de registro.
- O perfil/CLI determina active/deferred sem mutar a função, objeto ou classe
  registrada. Manter loading, declaração e configuração do executor coerentes.
- Somente instâncias construídas pelo host entram no fechamento da sessão,
  inclusive em falha parcial. Objetos fornecidos continuam caller-owned.

### Perfil

```toml
default_profile = "lite"

[profiles.lite.tools]
active = ["workspace", "uppercase"]
deferred = ["web_fetch"]
```

O módulo torna `uppercase` conhecido; a configuração determina como usá-lo.
`web_fetch` continua builtin. Tools declaradas e não selecionadas não devem ser
instanciadas nem entrar no catálogo do Agent.

### CLI

```bash
vulcano --extension meu_pacote.vulcano:register \
  --tools workspace,uppercase \
  --deferred-tools web_fetch
```

**Precedência implementada:** sem flags de tools, usar as duas listas do perfil.
Se qualquer flag `--tools` ou `--deferred-tools` estiver presente, o par de listas
CLI substitui a seleção de tools do perfil inteira; a lista não fornecida fica
vazia. Assim `--tools read` seleciona somente `read`, sem herdar tools deferred
do perfil. `--tools ''` seleciona um conjunto vazio de tools opcionais.

As flags têm default `None`, distinguindo ausência de lista vazia. CSV ignora
espaços externos; nomes vazios no meio, duplicatas e interseção active/deferred
geram erro. Aplicar o mesmo resolver após expandir grupos de capabilities;
interseções de modos geradas por expansão também devem gerar erro. Listas CLI
não modificam o TOML. `--profile`, `--config` e `-c` continuam funcionando.

Tools auxiliares derivadas pelo runtime permanecem automáticas: `task` quando
há execução em background, `tool_search` para deferred e comunicação de progresso
conforme modo/provider. Isso deve aparecer nos exemplos como superfície efetiva
do Agent, sem prometer que uma seleção vazia remove todo suporte do harness.
As restrições atuais de ferramentas OpenAI nativas também continuam valendo.

## AgentExtension: uso existente e ponte futura

O exemplo abaixo usa a API existente do Agent, independentemente do Vulcano:

```python
from msgflux.nn import Agent, AgentExtension


class TextActions(AgentExtension):
    def __init__(self):
        super().__init__("text_actions")

    def tools(self):
        tool = UppercaseTool()
        tool.tool_config = {"defer_loading": True}
        return (tool,)


agent = Agent(name="main", model=model, extensions=[TextActions()])
```

`Agent.register_extension` já instala tools, hooks, estado e handles com
ownership. `CodingExtensions` não deve cadastrar hooks manualmente em um sistema
paralelo. A mesma classe tool pode ser usada nos dois exemplos.

Uma futura ponte `register_agent_extension(name, factory)` precisa ter uma
seleção própria de extensões de comportamento. Selecionar uma tool não implica
ativar todos os hooks de um pacote. Além disso, `AgentExtension.tools()` instala
suas contribuições ao registrar a extensão; o CLI atual não consegue selecionar
um subconjunto ou sobrescrever seu modo deferred mantendo ownership.

Essa ponte deve vir numa etapa separada, com contrato explícito para tools
contribuídas e hooks. Não envolver a extensão num proxy que simule binding,
estado ou remoção; usar um ponto público de seleção no lifecycle, caso esse
requisito seja confirmado. O primeiro PR entrega registro de tools no coding e
seleção por CLI, e documenta o uso existente de AgentExtension sem anunciar essa
ponte como pronta.

## Ordem de implementação e arquivos afetados

### PR A: registro, seleção e documentação

1. `coding/extensions/records.py`, `registry.py`, `__init__.py`: spec da factory,
   `register_tool`, validação, ownership dos registros e testes de atomicidade.
2. `coding/tools.py`: catálogo de factories builtin/custom e resolução das
   selecionadas; impedir instanciamento antecipado das tools não escolhidas.
3. `coding/config.py`: função pura para precedência perfil/CLI e validação.
4. `coding/cli.py`: `--tools`, `--deferred-tools`; carregar declarações antes do
   host em TUI e print; closure da factory de sessão usa o registro, com cleanup.
5. `tests/coding/test_extensions.py`, `test_tools.py`, `test_cli.py`,
   `test_draft.py`: integração com a composição, scopes e SQLite reais.
6. `docs/learn/coding.md` e `docs/learn/nn/agent/extensions.md`: exemplos de
   classes, registro por extensão, perfis, CLI, deferred e ciclo de vida.

Essa implementação está sobre os commits das correções de coding e a main
atualizada, mantendo a base de workspace e o provider já mergeados.

### Etapa B: AgentExtensions no harness

Planejar a ponte de factories selecionáveis e a política das tools contribuídas
quando houver uso concreto de hooks/estado no coding. Depende do PR A; não é
necessária para que extensões `register(c)` adicionem tools selecionáveis.

## Testes e riscos concretos

- Importar/carregar um registro não cria tools, modelos ou diretórios de thread.
- O mesmo entry point pode registrar tools, comandos e painéis; print usa suas
  tools sem importar Textual. O registro só acontece uma vez por host.
- Nomes inexistentes e colisões entre módulos, builtins e groups falham cedo,
  antes de abrir recursos de sessão. Nome da definição compilada é validado.
- Duas threads criam instâncias distintas de classes registradas. Selecionar a mesma classe como active
  e deferred em hosts diferentes não modifica atributos compartilhados.
- Perfil, CLI parcial, CLI com ambas as listas, listas vazias e `-c` seguem a
  precedência documentada. Mudança de modo não pode duplicar tool.
- O primeiro deferred instala `tool_search`; registro não selecionado não instala
  o bucket. Verificar schemas e dispatch pela ToolLibrary real.
- Chamada real com workspace injetado respeita suas permissões; read-only não
  ganha ferramentas builtin de escrita/Bash. Uma tool Python arbitrária que
  executa acesso ao host por fora do workspace não recebe isolamento só por ter
  sido registrada; seleção de nome não substitui autorização.
- Background, TaskTool e commentary mantêm o comportamento atual.
- Falha ao construir uma classe fecha recursos anteriores; trocar de sessão e fechar a
  aplicação fecha recursos selecionados. Cleanup de AgentExtension continua
  usando o lifecycle do Agent.
- Rollback da função `register(c)` deve desfazer seu lote caso ela falhe, para não
  deixar declarações parciais. Política inicial: falhar o carregamento solicitado
  explicitamente com origem e mensagem claras.

Validação: testes focados de coding e ToolLibrary, Ruff, MkDocs strict e
regressão offline após congelar o código. Testes de integração usam factories
observáveis, SQLite e workspace local; não precisam de chamadas pagas a modelos.

## Ajuste aprovado: registrar a própria tool

Após a recuperação da worktree em 2026-10-03, substituir a API de factories
por `register_tool(tool)`: função, classe ou instância callable, sem repetir
name/description no registro. Nome, descrição e schema seguem as declarações
nativas da biblioteca. Apenas active/deferred continuam disponíveis.

Ordem e arquivos:

1. `coding/extensions/{records,registry,__init__}.py`: ToolSpec com origem,
   inferência do nome sem construir classes; preservar handles e rollback.
2. `coding/tools.py`: construir classes selecionadas por sessão, compilar
   via ToolLibrary, aplicar loading na definição sem mutar a origem.
   `nn/modules/tool/implementations.py`: usar o nome do tipo como fallback
   para instâncias sem `name`, preservando o nome nativo de classes já
   materializadas pelo host. Validar função, classe e objeto por integração.
3. `coding/cli.py`: rastrear somente instâncias construídas pelo host para
   cleanup. Funções e instâncias fornecidas pertencem ao chamador.
4. Atualizar testes de registro/resolver/integração e exemplos em
   `docs/learn/coding.md` e `docs/learn/nn/agent/extensions.md`.

Riscos: nome dinâmico criado só no construtor não serve à seleção antecipada;
instâncias compartilhadas mantêm estado e lifecycle do chamador; schema e
loading devem continuar coerentes nas definições compiladas. Verificar
funções sync/async, classes lazy, objetos configurados, workspace injetado,
deferred discovery/call, isolamento de configurações e cleanup em erro.

Erro DNS relatado: reproduzir separadamente com gpt-6-luna e web_fetch
deferred, identificando se a origem é o provider, parser ou página alvo.
Não atribuir falha de rede ao loading sem evidência.
