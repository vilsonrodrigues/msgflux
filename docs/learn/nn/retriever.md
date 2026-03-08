# nn.Retriever

The `nn.Retriever` module provides a unified interface for information retrieval using web search, lexical search, semantic search, or vector databases.

All code examples use the recommended import pattern:

```python
import msgflux as mf
import msgflux.nn as nn
```

## Quick Start

### AutoParams Initialization (Recommended)

This is the preferred and recommended way to define retrievers in msgFlux. It promotes reusability and clear intent.

```python
import msgflux as mf
import msgflux.nn as nn

class DocumentRetriever(nn.Retriever):
    """Semantic retriever for technical documentation."""
    # AutoParams automatically uses class name as 'name'
    response_mode = "plain_response"

# Setup backend
vector_db = mf.data.VectorDB.qdrant(
    collection_name="documents",
    url="http://localhost:6333"
)
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Create instance with defaults
doc_retriever = DocumentRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5, "threshold": 0.7}
)

# Use the retriever
results = doc_retriever("What is machine learning?")
print(results)
```

### Traditional Initialization

For quick scripts or one-off usage:

```python
retriever = nn.Retriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5}
)
```

---

## Retriever Types

### 1. Vector Database Retrieval (Semantic)

Most common type - uses embeddings for semantic similarity search.

```python
class KnowledgeBaseRetriever(nn.Retriever):
    """Semantic retriever for company knowledge base with high precision."""
    response_mode = "plain_response"

# Create with strict thresholds for high precision
kb_retriever = KnowledgeBaseRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5, "threshold": 0.75, "return_score": True}
)

results = kb_retriever("How do I use async functions in Python?")
```

### 2. Web Retrieval

Search the web for real-time information.

```python
class WebResearcher(nn.Retriever):
    """Web search retriever for real-time information."""
    response_mode = "plain_response"

# Create web retriever client
web_search = mf.DataRetriever.tavily(api_key="your-api-key")

researcher = WebResearcher(
    retriever=web_search,
    config={"top_k": 10}
)

# Search current events
results = researcher("Latest news on artificial intelligence")
```

### 3. Lexical Retrieval

Traditional keyword-based search (BM25, TF-IDF).

```python
class DocumentSearcher(nn.Retriever):
    """Fast lexical search for exact keyword matching."""
    response_mode = "plain_response"

# Create lexical retriever
lexical_search = mf.DataRetriever.bm25(
    documents=["doc1.txt", "doc2.txt", "doc3.txt"]
)

searcher = DocumentSearcher(
    retriever=lexical_search,
    config={"top_k": 3}
)

results = searcher("Python async await")
```

---

## Advanced Configuration

### Message Field Mapping

Use `Message` objects for structured input/output. This decouples your retriever from the specific data structure.

```python
class ContextualRetriever(nn.Retriever):
    """Retriever that processes structured message inputs."""
    response_mode = "message"  # Return Message object

retriever = ContextualRetriever(
    retriever=vector_db,
    model=embedder,
    message_fields={
        "task_inputs": "query.user"  # Extract query from message.query.user
    },
    config={"top_k": 5}
)

# Use with Message
msg = mf.Message()
msg.set("query.user", "What is dependency injection?")

result_msg = retriever(msg)
# Results are written to the default location or configured output
print(result_msg.get("retriever.results"))
```

### Response Templates

Format retrieval results using Jinja templates before returning them.

```python
class FormattedRetriever(nn.Retriever):
    """Retriever with custom result formatting."""
    response_mode = "plain_response"

retriever = FormattedRetriever(
    retriever=vector_db,
    model=embedder,
    templates={
        "response": """
        Found {{ results|length }} relevant articles:
        {% for result in results %}
        {{ loop.index }}. {{ result.data }} (Score: {{ result.score }})
        {% endfor %}
        """
    },
    config={"top_k": 5, "return_score": True}
)

formatted_results = retriever("machine learning best practices")
```

### Runtime Configuration Override

Override configuration at call time for flexibility.

```python
retriever = FlexibleRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 3}  # Default
)

# Normal retrieval
results = retriever("python tutorial")

# Override parameters at runtime
results = retriever(
    "machine learning",
    task_inputs="deep learning neural networks",  # Override query
    config={"top_k": 10}                          # Override config
)
```

---

## Creating Retriever Hierarchies

Build specialized retrievers through inheritance to share configuration.

```python
# Base retriever for all documentation
class BaseDocRetriever(nn.Retriever):
    """Base retriever for documentation with common config."""
    response_mode = "plain_response"

# High precision retriever for critical docs
class CriticalDocRetriever(BaseDocRetriever):
    """High precision retriever for critical documentation (compliance, security)."""
    # Inherits behavior from BaseDocRetriever

# Broad retriever for general exploration
class ExploratoryRetriever(BaseDocRetriever):
    """Broad retriever for exploratory searches."""

# Create instances with different thresholds
critical = CriticalDocRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 3, "threshold": 0.85}  # Strict
)

exploratory = ExploratoryRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 10, "threshold": 0.5}  # Permissive
)
```

---

## Integration with Agents

Retrievers are commonly used as tools for agents to implement RAG patterns.

```python
# Define retriever
kb_retriever = CompanyKnowledgeRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5}
)

# Define retriever as a tool function
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for information."""
    results = kb_retriever(query)
    return str(results)

# Create agent with retriever tool
class SupportAgent(nn.Agent):
    """Customer support agent with access to knowledge base."""
    model = mf.Model.chat_completion("openai/gpt-4")
    tools = [search_knowledge_base]

support_agent = SupportAgent()

# Agent can now use retriever to answer questions
response = support_agent("How do I reset my password?")
```

---

## Async Support

Retrievers provide first-class `async` support via `aforward` or `acall`.

```python
import asyncio

async def search_multiple():
    """Perform multiple searches concurrently."""
    results = await asyncio.gather(
        retriever.aforward("machine learning"),
        retriever.aforward("deep learning"),
        retriever.aforward("neural networks")
    )
    return results

# Run async searches
results = asyncio.run(search_multiple())
```

---

## Best Practices

1.  **Use Domain-Specific Retrievers**: Create specialized classes (`TechnicalDocRetriever`, `CustomerQueryRetriever`) rather than generic instances.
2.  **Tune Thresholds**: Use strict thresholds (0.8+) for precision-critical tasks and lower thresholds (0.5-0.6) for exploration.
3.  **Return Scores**: Enable `return_score=True` to debug retrieval quality and filter low-confidence results in your logic.
