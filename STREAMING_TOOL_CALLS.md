# Streaming e Parsing de Tool Calls

## Dois Paradigmas: SDK Estruturado vs XML em Texto

Existem duas formas principais de fazer tool calling:

### 1. SDK Estruturado (OpenAI, Anthropic SDK)

O modelo gera blocos separados para texto e tools. **Não mistura**.

- Usado por: OpenAI SDK, Anthropic SDK (tool_use blocks)
- Vantagem: Parsing simples, estrutura clara
- Desvantagem: Modelo para de gerar texto para chamar tool

Exemplo de eventos SSE (Anthropic):

    event: content_block_start
    data: {"type": "tool_use", "id": "toolu_xxx", "name": "Read"}

    event: content_block_delta
    data: {"partial_json": "{\"file_path\": \"/path\"}"}

    event: content_block_stop


### 2. XML Tags em Texto (Claude Code, Agents)

O modelo gera **tudo como texto**, usando XML tags para marcar tool calls.
Texto e tools podem vir **no mesmo stream contínuo**.

- Usado por: Claude Code, Claude Agent SDK
- Vantagem: Texto e tools intercalados naturalmente
- Desvantagem: Requer parser XML incremental

**Estrutura das tags (exemplo simplificado):**

    [OPEN_TAG]antml:function_calls[CLOSE_TAG]
    [OPEN_TAG]antml:invoke name="Read"[CLOSE_TAG]
    [OPEN_TAG]antml:parameter name="file_path"[CLOSE_TAG]/path/to/file[END_TAG]antml:parameter[CLOSE_TAG]
    [END_TAG]antml:invoke[CLOSE_TAG]
    [END_TAG]antml:function_calls[CLOSE_TAG]

Onde OPEN_TAG = "<", CLOSE_TAG = ">", END_TAG = "</"

**Fluxo típico:**

    Token stream chegando:
    "Vou" " ler" " o" " arquivo" "." "\n" "\n"
    "<" "antml" ":" "function" "_calls" ">" "\n"
    "<" "antml" ":" "invoke" " name" "=\"" "Read" "\">" "\n"
    ...

O parser precisa:
1. Fazer streaming do texto normalmente até detectar abertura de tag
2. Ao detectar, parar o streaming de texto para o usuário
3. Acumular buffer até fechar a tag de function_calls
4. Parsear XML, executar tool
5. Continuar streaming do texto após a tag


## Arquitetura do Parser XML Incremental

    ┌─────────────────────────────────────────────────────────────────┐
    │                        Token Stream                             │
    │   "Vou ler..." → "<function_calls>" → "..." → "</...>"   │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Streaming Parser                            │
    │                                                                 │
    │   Estado: TEXT | MAYBE_TAG | IN_TAG | IN_TOOL_BLOCK            │
    │                                                                 │
    │   TEXT:        Stream direto para usuário                       │
    │   MAYBE_TAG:   Viu "<", aguardando próximos chars               │
    │   IN_TAG:      Dentro de tag, não fazer stream                  │
    │   IN_TOOL_BLOCK: Acumulando conteúdo da tool call               │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐   ┌──────────┐   ┌──────────┐
              │  Texto   │   │   Tool   │   │  Texto   │
              │  (user)  │   │  (exec)  │   │  (user)  │
              └──────────┘   └──────────┘   └──────────┘


## Implementação do Parser

### Estados do Parser

    from enum import Enum, auto

    class ParserState(Enum):
        TEXT = auto()           # Streaming texto normal
        MAYBE_TAG = auto()      # Viu "<", esperando confirmar
        IN_TAG_NAME = auto()    # Lendo nome da tag
        IN_TOOL_BLOCK = auto()  # Dentro do bloco de tools


### Classe StreamingXMLParser

    class StreamingXMLParser:
        """Parser incremental para detectar tool calls em XML."""

        # Constantes - nome da tag que inicia bloco de tools
        TOOL_BLOCK_START = "antml:function_calls"
        TOOL_BLOCK_END = "/antml:function_calls"

        def __init__(self):
            self.state = ParserState.TEXT
            self.buffer = ""
            self.tag_buffer = ""
            self.tool_block_buffer = ""
            self.depth = 0  # Para tags aninhadas

        def feed(self, token: str) -> tuple[str, list[dict]]:
            """Processa um token e retorna (texto_para_stream, tools_detectadas)."""
            text_output = ""
            detected_tools = []

            for char in token:
                result = self._process_char(char)
                if result:
                    if result["type"] == "text":
                        text_output += result["content"]
                    elif result["type"] == "tool":
                        detected_tools.append(result["content"])

            return text_output, detected_tools

        def _process_char(self, char: str) -> dict | None:
            if self.state == ParserState.TEXT:
                return self._handle_text(char)
            elif self.state == ParserState.MAYBE_TAG:
                return self._handle_maybe_tag(char)
            elif self.state == ParserState.IN_TAG_NAME:
                return self._handle_tag_name(char)
            elif self.state == ParserState.IN_TOOL_BLOCK:
                return self._handle_tool_block(char)

        def _handle_text(self, char: str) -> dict | None:
            if char == "<":
                self.state = ParserState.MAYBE_TAG
                self.tag_buffer = "<"
                return None
            return {"type": "text", "content": char}

        def _handle_maybe_tag(self, char: str) -> dict | None:
            self.tag_buffer += char

            # Verificar se é início de tag válida
            if char.isalpha() or char == "/" or char == "a":
                self.state = ParserState.IN_TAG_NAME
                return None

            # Não é tag, flush buffer como texto
            self.state = ParserState.TEXT
            result = {"type": "text", "content": self.tag_buffer}
            self.tag_buffer = ""
            return result

        def _handle_tag_name(self, char: str) -> dict | None:
            self.tag_buffer += char

            if char == ">":
                tag_name = self._extract_tag_name(self.tag_buffer)

                if tag_name == self.TOOL_BLOCK_START:
                    # Início de bloco de tools
                    self.state = ParserState.IN_TOOL_BLOCK
                    self.tool_block_buffer = self.tag_buffer
                    self.depth = 1
                    self.tag_buffer = ""
                    return None

                # Tag normal, não é tool - flush como texto
                self.state = ParserState.TEXT
                result = {"type": "text", "content": self.tag_buffer}
                self.tag_buffer = ""
                return result

            return None

        def _handle_tool_block(self, char: str) -> dict | None:
            self.tool_block_buffer += char

            # Detectar fechamento do bloco
            if self.tool_block_buffer.endswith(">" + self.TOOL_BLOCK_END + ">"):
                # Bloco completo - parsear
                tool_data = self._parse_tool_block(self.tool_block_buffer)
                self.state = ParserState.TEXT
                self.tool_block_buffer = ""
                return {"type": "tool", "content": tool_data}

            return None

        def _extract_tag_name(self, tag: str) -> str:
            """Extrai nome da tag de '<nome>' ou '<nome attr="val">'"""
            # Remove < e >
            inner = tag[1:-1]
            # Pega até o primeiro espaço ou fim
            return inner.split()[0] if " " in inner else inner

        def _parse_tool_block(self, xml_block: str) -> list[dict]:
            """Parseia bloco XML e extrai tool calls."""
            # Aqui você usaria um parser XML real
            # Exemplo simplificado:
            import re

            tools = []
            # Regex para extrair invokes
            invoke_pattern = r'invoke name="([^"]+)"'
            param_pattern = r'parameter name="([^"]+)">([^<]*)<'

            for invoke_match in re.finditer(invoke_pattern, xml_block):
                tool_name = invoke_match.group(1)
                # Encontrar parâmetros desta invoke
                # ... parsing completo aqui
                tools.append({"name": tool_name, "params": {}})

            return tools


## Uso do Parser no AgentStreamer

    class AgentStreamer:
        def __init__(self, tool_executor, on_text=None):
            self.parser = StreamingXMLParser()
            self.tool_executor = tool_executor
            self.on_text = on_text or print

        async def process_stream(self, token_stream):
            """Processa stream de tokens do modelo."""

            async for token in token_stream:
                # Alimentar parser
                text, tools = self.parser.feed(token)

                # Stream texto imediatamente
                if text:
                    self.on_text(text)

                # Executar tools detectadas
                for tool in tools:
                    result = await self.tool_executor(
                        tool["name"],
                        tool["params"]
                    )
                    # Injetar resultado de volta no contexto
                    await self._inject_result(tool, result)


## Comparação Final

    ┌────────────────────┬─────────────────────┬─────────────────────┐
    │     Aspecto        │   SDK Estruturado   │    XML em Texto     │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ Texto + Tool       │ Blocos separados    │ Mesmo stream        │
    │ juntos?            │                     │                     │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ Parser             │ Simples (JSON)      │ Complexo (XML inc.) │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ Múltiplas tools    │ Índice separado     │ Tags aninhadas      │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ UX streaming       │ Pausa para tool     │ Fluido, contínuo    │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ Usado por          │ OpenAI, Anthropic   │ Claude Code,        │
    │                    │ (SDK nativo)        │ Agent SDK           │
    └────────────────────┴─────────────────────┴─────────────────────┘


## Considerações para msgflux

Se você quer o comportamento do Claude Code (texto + tool no mesmo stream):

1. **Não use o tool_use nativo do SDK** - Use geração de texto puro
2. **Defina as tools no system prompt** como XML schema
3. **Implemente parser XML incremental** para detectar tags
4. **Stream texto** até detectar tag de abertura
5. **Buffer** o conteúdo da tool até fechar
6. **Execute** e injete resultado
7. **Continue** streaming

Esta abordagem dá mais controle e UX mais fluida, mas requer mais trabalho de implementação.
