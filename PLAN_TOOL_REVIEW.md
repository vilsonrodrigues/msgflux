# Plano: Mecanismo de Review para Tools

## Resumo

Implementar um sistema de review que permite interceptar e revisar inputs (parâmetros escritos pelo modelo) e outputs (resultados gerados pela tool) antes de prosseguir com a execução.

## Inspiração: Claude Code

O Claude Code usa um sistema de permissões que:
1. Verifica regras de deny/allow
2. Exibe informações detalhadas ao usuário (tool name, parameters, reasoning)
3. Permite aprovar/negar chamadas
4. Suporta callbacks programáticos (`canUseTool`)

## Design Proposto

### Nova Propriedade em `tool_config`

```python
@tool_config(
    review=ReviewConfig(
        input=review_input_fn,   # Callable para revisar inputs
        output=review_output_fn, # Callable para revisar outputs
    )
)
def my_tool(query: str):
    ...
```

### Classe `ReviewConfig`

```python
# src/msgflux/tools/review.py
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

# Type aliases para as funções de review
ReviewInputFn = Callable[[str, str, dict], Union[bool, dict]]
# (tool_id, tool_name, params) -> bool | modified_params

ReviewOutputFn = Callable[[str, str, dict, Any], Union[bool, Any]]
# (tool_id, tool_name, params, result) -> bool | modified_result

@dataclass
class ReviewConfig:
    """Configuration for tool review.

    Args:
        input: Function to review/modify inputs before execution.
               Returns True to proceed, False to block, or dict to modify params.
        output: Function to review/modify outputs after execution.
                Returns True to proceed, False to block, or value to replace result.
    """
    input: Optional[ReviewInputFn] = None
    output: Optional[ReviewOutputFn] = None
```

### Resultado do Review

```python
@dataclass
class ReviewResult:
    """Result of a review operation."""
    approved: bool
    modified_value: Optional[Any] = None  # None = use original
    reason: Optional[str] = None  # Reason for denial (if denied)
```

### Assinaturas das Funções de Review

#### Input Review

```python
def review_input(tool_id: str, tool_name: str, params: dict) -> ReviewResult:
    """Review tool inputs before execution.

    Args:
        tool_id: Unique ID of the tool call
        tool_name: Name of the tool being called
        params: Parameters the model passed to the tool

    Returns:
        ReviewResult with:
        - approved=True, modified_value=None: proceed with original params
        - approved=True, modified_value=dict: proceed with modified params
        - approved=False: block execution, reason explains why
    """
```

#### Output Review

```python
def review_output(tool_id: str, tool_name: str, params: dict, result: Any) -> ReviewResult:
    """Review tool output after execution.

    Args:
        tool_id: Unique ID of the tool call
        tool_name: Name of the tool
        params: Parameters used (possibly modified by input review)
        result: Result returned by the tool

    Returns:
        ReviewResult with:
        - approved=True, modified_value=None: proceed with original result
        - approved=True, modified_value=Any: proceed with modified result
        - approved=False: block result, return error instead
    """
```

### Fluxo de Execução na ToolLibrary

```
1. Model gera tool call
2. [INPUT REVIEW] Se review.input configurado:
   - Chama review.input(tool_id, tool_name, params)
   - Se denied → registra erro, não executa tool
   - Se approved com modificação → usa params modificados
   - Se approved → usa params originais
3. Executa tool com params (originais ou modificados)
4. [OUTPUT REVIEW] Se review.output configurado:
   - Chama review.output(tool_id, tool_name, params, result)
   - Se denied → substitui result por mensagem de erro
   - Se approved com modificação → usa result modificado
   - Se approved → usa result original
5. Retorna ToolResponses
```

### Async Support

```python
# Para suportar async, as funções podem ser async ou sync
AsyncReviewInputFn = Callable[[str, str, dict], Awaitable[ReviewResult]]
AsyncReviewOutputFn = Callable[[str, str, dict, Any], Awaitable[ReviewResult]]
```

---

## Estrutura de Arquivos

### Novos Arquivos

```
src/msgflux/tools/
├── review.py      # ReviewConfig, ReviewResult, type hints
```

### Arquivos Modificados

```
src/msgflux/tools/config.py    # Adicionar parâmetro `review`
src/msgflux/tools/__init__.py  # Exportar ReviewConfig, ReviewResult
src/msgflux/__init__.py        # Exportar no nível raiz
src/msgflux/nn/modules/tool.py # Implementar lógica de review em forward/aforward
```

---

## Passos de Implementação

### Passo 1: Criar módulo de review
1. Criar `src/msgflux/tools/review.py`
2. Implementar `ReviewResult` dataclass
3. Implementar `ReviewConfig` dataclass
4. Definir type aliases para as funções

### Passo 2: Atualizar tool_config
1. Adicionar parâmetro `review: Optional[ReviewConfig]` ao decorator
2. Incluir `review` no dict `tool_config`
3. Atualizar docstring

### Passo 3: Atualizar exports
1. Atualizar `src/msgflux/tools/__init__.py`
2. Atualizar `src/msgflux/__init__.py`

### Passo 4: Implementar review na ToolLibrary
1. Em `forward()`:
   - Antes de executar: chamar input review se configurado
   - Após executar: chamar output review se configurado
   - Tratar casos de denial e modificação
2. Em `aforward()`:
   - Mesmo para versão async
   - Suportar funções de review sync e async

### Passo 5: Testes
1. Criar `tests/test_tool_review.py`
2. Testar input review (approve, deny, modify)
3. Testar output review (approve, deny, modify)
4. Testar combinação input + output
5. Testar async

---

## Exemplos de Uso

### Exemplo 1: Review Simples de Input

```python
from msgflux import tool_config
from msgflux.tools import ReviewConfig, ReviewResult

def require_confirmation(tool_id: str, tool_name: str, params: dict) -> ReviewResult:
    """Pede confirmação para qualquer tool call."""
    print(f"Tool: {tool_name}")
    print(f"Params: {params}")
    response = input("Approve? (y/n): ")
    return ReviewResult(approved=response.lower() == 'y')

@tool_config(review=ReviewConfig(input=require_confirmation))
def delete_file(path: str):
    """Delete a file."""
    os.remove(path)
    return f"Deleted {path}"
```

### Exemplo 2: Sanitização de Input

```python
def sanitize_sql_params(tool_id: str, tool_name: str, params: dict) -> ReviewResult:
    """Remove caracteres perigosos de queries SQL."""
    if "query" in params:
        sanitized = params["query"].replace(";", "").replace("--", "")
        return ReviewResult(approved=True, modified_value={"query": sanitized})
    return ReviewResult(approved=True)

@tool_config(review=ReviewConfig(input=sanitize_sql_params))
def run_query(query: str):
    """Run SQL query."""
    ...
```

### Exemplo 3: Filtro de Output

```python
def filter_sensitive_data(tool_id: str, tool_name: str, params: dict, result: Any) -> ReviewResult:
    """Remove dados sensíveis do resultado."""
    if isinstance(result, dict):
        filtered = {k: v for k, v in result.items() if k not in ["password", "token"]}
        return ReviewResult(approved=True, modified_value=filtered)
    return ReviewResult(approved=True)

@tool_config(review=ReviewConfig(output=filter_sensitive_data))
def get_user_data(user_id: str):
    """Get user data."""
    ...
```

### Exemplo 4: Async Review

```python
async def async_approval(tool_id: str, tool_name: str, params: dict) -> ReviewResult:
    """Aprovação assíncrona (ex: webhook, API externa)."""
    response = await external_approval_api.request_approval(tool_name, params)
    return ReviewResult(
        approved=response.approved,
        reason=response.reason if not response.approved else None
    )

@tool_config(review=ReviewConfig(input=async_approval))
async def sensitive_operation():
    ...
```

---

## Considerações

### Performance
- Review adiciona latência a cada tool call
- Considerar cache para decisões repetidas
- Funções de review devem ser leves

### Segurança
- Review NÃO deve ser bypassável pelo modelo
- Erros na função de review devem bloquear execução (fail-safe)
- Logs de todas as decisões de review

### Telemetria
- Capturar no trace: review decision, modifications, denial reasons
- Permitir auditar todas as decisões

### Compatibilidade
- Tools sem `review` configurado funcionam normalmente
- Backward compatible com código existente

---

## Arquivos Críticos

| Arquivo | Ação |
|---------|------|
| `src/msgflux/tools/review.py` | CRIAR |
| `src/msgflux/tools/config.py` | MODIFICAR |
| `src/msgflux/tools/__init__.py` | MODIFICAR |
| `src/msgflux/__init__.py` | MODIFICAR |
| `src/msgflux/nn/modules/tool.py` | MODIFICAR |
| `tests/test_tool_review.py` | CRIAR |
