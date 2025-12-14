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


## Plano de Implementação para msgflux

### Etapa 1: Refatorar Streaming no Cliente OpenAI

**Problema dos métodos atuais (`_stream_generate`, `_astream_generate`):**

Os métodos existentes têm limitações fundamentais:
1. Acumulam tool calls em `ToolCallAggregator` até o FIM do stream
2. Só processam tool calls DEPOIS que o stream termina
3. Não permitem execução de tools "inline" durante o stream
4. Usam background task que não permite controle fino do fluxo

**Comportamento atual (problemático para AgentStreamer):**

    async for chunk in model_output:
        if delta.tool_calls:
            aggregator.process(...)  # Só acumula, não executa
    # Só aqui, após loop, temos os tool calls completos
    stream_response.data = aggregator

**Comportamento necessário:**

    async for chunk in model_output:
        if delta.content:
            yield StreamChunk(type="text", content=delta.content)
        if delta.tool_calls:
            # Acumula até tool call completo
            # Quando completo: yield e permite execução
            if tool_complete:
                yield StreamChunk(type="tool_call", tool_call=complete_tool)

**Solução: Criar novos métodos `stream()` e `astream()`**

Não modificar os existentes (manter compatibilidade), criar novos que retornem
iteradores reais com chunks tipados.

**Arquivos a modificar:**
- `src/msgflux/models/providers/openai.py`
- `src/msgflux/models/response.py` (novos tipos)
- `src/msgflux/models/types.py` (protocol para streaming)

**Novos métodos no OpenAIChatCompletion:**

    def stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict]] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """Retorna iterator síncrono de chunks tipados.

        Diferente de __call__(stream=True), este método:
        - Retorna Iterator real (não ModelStreamResponse)
        - Emite chunks tipados (text, tool_call, reasoning, usage)
        - Permite processamento inline de tool calls
        - Não usa background task
        """
        # Preparar parâmetros
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]
        if system_prompt:
            messages.insert(0, ChatBlock.system(system_prompt))

        params = {
            "messages": messages,
            "model": self.model_id,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.sampling_run_params,
        }
        if tool_schemas:
            params["tools"] = tool_schemas
            params["parallel_tool_calls"] = self.parallel_tool_calls

        # Executar stream
        response = self.client.chat.completions.create(**params)

        # Estado para acumular tool calls
        tool_buffers: Dict[int, ToolCallBuffer] = {}

        for chunk in response:
            if not chunk.choices:
                # Usage chunk (final)
                if chunk.usage:
                    yield StreamChunk(
                        type="usage",
                        usage=chunk.usage.to_dict()
                    )
                continue

            delta = chunk.choices[0].delta

            # Texto
            if delta.content:
                yield StreamChunk(type="text", content=delta.content)

            # Reasoning (se disponível)
            reasoning = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
            )
            if reasoning:
                yield StreamChunk(type="reasoning", content=reasoning)

            # Tool calls (acumular até completo)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_buffers:
                        tool_buffers[idx] = ToolCallBuffer(
                            index=idx,
                            id=tc.id,
                            name=tc.function.name if tc.function else None,
                            arguments=""
                        )
                    buf = tool_buffers[idx]
                    if tc.id:
                        buf.id = tc.id
                    if tc.function:
                        if tc.function.name:
                            buf.name = tc.function.name
                        if tc.function.arguments:
                            buf.arguments += tc.function.arguments

            # Verificar finish_reason para emitir tool calls completos
            if chunk.choices[0].finish_reason == "tool_calls":
                for idx in sorted(tool_buffers.keys()):
                    buf = tool_buffers[idx]
                    yield StreamChunk(
                        type="tool_call",
                        tool_call=ToolCallComplete(
                            id=buf.id,
                            name=buf.name,
                            arguments=buf.arguments
                        )
                    )

    async def astream(
        self,
        messages: List[Dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Versão async de stream()."""
        # Mesma lógica, mas com async for
        # ... (implementação similar)


**Novos tipos (em models/response.py ou streaming/types.py):**

    @dataclass
    class StreamChunk:
        """Chunk individual do stream.

        Tipos:
        - text: Conteúdo textual para exibir ao usuário
        - reasoning: Conteúdo de raciocínio/thinking (se modelo suportar)
        - tool_call: Tool call completo, pronto para execução
        - usage: Estatísticas de uso (tokens)
        """
        type: Literal["text", "reasoning", "tool_call", "usage"]
        content: Optional[str] = None
        tool_call: Optional["ToolCallComplete"] = None
        usage: Optional[Dict[str, Any]] = None

    @dataclass
    class ToolCallBuffer:
        """Buffer para acumular tool call durante stream."""
        index: int
        id: Optional[str] = None
        name: Optional[str] = None
        arguments: str = ""

    @dataclass
    class ToolCallComplete:
        """Tool call completo, pronto para execução."""
        id: str
        name: str
        arguments: str  # JSON string

        def get_params(self) -> Dict[str, Any]:
            """Parse arguments JSON para dict."""
            import json
            return json.loads(self.arguments) if self.arguments else {}


### Etapa 2: Implementar AgentStreamer a partir do nn.Agent

Criar uma classe que herda/estende Agent para suportar streaming com execução
de tools em tempo real.

**Arquivos a criar:**
- `src/msgflux/nn/modules/agent_streamer.py`

**Arquivos a modificar:**
- `src/msgflux/nn/modules/__init__.py`
- `src/msgflux/nn/__init__.py`

**Classe AgentStreamer:**

    class AgentStreamer(Agent):
        """Agent com suporte a streaming real e execução de tools em tempo real.

        Usa o paradigma XML-em-texto para permitir texto e tool calls
        intercalados no mesmo stream.
        """

        def __init__(
            self,
            name: str,
            model: ChatCompletionModel,
            *,
            on_text: Optional[Callable[[str], None]] = None,
            on_tool_start: Optional[Callable[[str, str, dict], None]] = None,
            on_tool_end: Optional[Callable[[str, Any], None]] = None,
            **kwargs
        ):
            super().__init__(name, model, **kwargs)
            self.on_text = on_text or self._default_text_handler
            self.on_tool_start = on_tool_start
            self.on_tool_end = on_tool_end
            self.parser = StreamingXMLParser()

        def stream(
            self,
            message: Optional[Union[str, Mapping[str, Any], Message]] = None,
            **kwargs
        ) -> Iterator[StreamEvent]:
            """Stream de eventos (texto, tool_start, tool_result, etc)."""
            inputs = self._prepare_task(message, **kwargs)
            model_state = inputs["model_state"]
            vars = inputs["vars"]

            while True:
                # Stream do modelo
                for chunk in self._stream_model(model_state, vars):
                    text, tools = self.parser.feed(chunk.content or "")

                    # Yield texto
                    if text:
                        yield StreamEvent(type="text", content=text)
                        if self.on_text:
                            self.on_text(text)

                    # Processar tools detectadas
                    for tool in tools:
                        yield StreamEvent(type="tool_start", tool_name=tool["name"])
                        if self.on_tool_start:
                            self.on_tool_start(tool["id"], tool["name"], tool["params"])

                        result = self._execute_tool(tool, model_state, vars)

                        yield StreamEvent(type="tool_result", tool_result=result)
                        if self.on_tool_end:
                            self.on_tool_end(tool["id"], result)

                # Verificar se precisa continuar loop (mais tools)
                if not self._should_continue(model_state):
                    break

        async def astream(
            self,
            message: Optional[Union[str, Mapping[str, Any], Message]] = None,
            **kwargs
        ) -> AsyncIterator[StreamEvent]:
            """Async stream de eventos."""
            # Implementação similar mas async
            ...


**Classe StreamEvent:**

    @dataclass
    class StreamEvent:
        """Evento do stream do AgentStreamer."""
        type: Literal["text", "tool_start", "tool_result", "done", "error"]
        content: Optional[str] = None
        tool_name: Optional[str] = None
        tool_params: Optional[dict] = None
        tool_result: Optional[Any] = None
        error: Optional[str] = None


### Estrutura Final de Arquivos

    src/msgflux/
    ├── models/
    │   ├── providers/
    │   │   └── openai.py          # Adicionar stream(), astream()
    │   └── response.py            # Adicionar StreamChunk, ToolCallDelta
    ├── nn/
    │   └── modules/
    │       ├── agent.py           # Agent existente
    │       └── agent_streamer.py  # NOVO: AgentStreamer
    └── streaming/                 # NOVO: Módulo de streaming
        ├── __init__.py
        ├── parser.py              # StreamingXMLParser
        ├── events.py              # StreamEvent, StreamChunk
        └── handlers.py            # Handlers de callbacks


### Passos de Implementação

**Fase 1: Streaming no Cliente**
1. Criar `StreamChunk` e `ToolCallDelta` em models/response.py
2. Adicionar `stream()` ao OpenAIChatCompletion
3. Adicionar `astream()` ao OpenAIChatCompletion
4. Testar streaming básico

**Fase 2: Parser XML Incremental**
1. Criar módulo `streaming/`
2. Implementar `StreamingXMLParser`
3. Implementar `StreamEvent`
4. Testar parser isoladamente

**Fase 3: AgentStreamer**
1. Criar `AgentStreamer` herdando de `Agent`
2. Implementar `stream()` com loop de tools
3. Implementar `astream()`
4. Integrar callbacks (on_text, on_tool_start, on_tool_end)
5. Testar fluxo completo

**Fase 4: Integração**
1. Exportar novas classes
2. Documentar uso
3. Criar exemplos
