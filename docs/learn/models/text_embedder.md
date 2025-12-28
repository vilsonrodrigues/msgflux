# Text Embedder

The `text_embedder` model transforms text into dense vector representations (embeddings) that capture semantic meaning. These vectors enable similarity search, semantic retrieval, clustering, and classification tasks.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Overview

Text embeddings convert sentences, paragraphs, or documents into numerical vectors that encode their semantic meaning. Unlike simple word counts or TF-IDF, embeddings capture:

- **Semantic similarity**: Similar meanings have similar vectors
- **Contextual understanding**: Same words in different contexts get different embeddings
- **Dimensionality reduction**: High-dimensional text → fixed-size vectors

### Common Use Cases

- **Semantic Search**: Find documents similar to a query
- **RAG (Retrieval-Augmented Generation)**: Retrieve relevant context for LLMs
- **Clustering**: Group similar documents
- **Classification**: Train classifiers on embeddings
- **Recommendation**: Find similar items

## Quick Start

### Basic Usage

```python
import msgflux as mf

# Create embedder
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Generate embedding
response = embedder("Hello, world!")

# Get vector
embedding = response.consume()
print(len(embedding))  # 1536
print(embedding[:5])   # [0.123, -0.456, 0.789, -0.234, 0.567]
```

### With Custom Dimensions

```python
import msgflux as mf

# OpenAI models support custom dimensions
embedder = mf.Model.text_embedder(
    "openai/text-embedding-3-small",
    dimensions=256  # Reduce from 1536 to 256
)

response = embedder("Compact embedding")
embedding = response.consume()
print(len(embedding))  # 256
```

## Supported Providers

### OpenAI

```python
import msgflux as mf

# text-embedding-3-small (1536 dims, $0.02/1M tokens)
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# text-embedding-3-large (3072 dims, $0.13/1M tokens)
embedder = mf.Model.text_embedder("openai/text-embedding-3-large")

# Legacy ada-002 (1536 dims, $0.10/1M tokens)
embedder = mf.Model.text_embedder("openai/text-embedding-ada-002")
```

### Jina AI

```python
import msgflux as mf

# Specialized embedding models
embedder = mf.Model.text_embedder("jinaai/jina-embeddings-v3")
```

### Together AI

```python
import msgflux as mf

# Open-source embeddings
embedder = mf.Model.text_embedder("together/togethercomputer/m2-bert-80M-8k-retrieval")
```

### Local (Ollama)

```python
import msgflux as mf

# Local embeddings with Ollama
embedder = mf.Model.text_embedder("ollama/nomic-embed-text")
```

### Local (vLLM)

```python
import msgflux as mf

# Self-hosted with vLLM
embedder = mf.Model.text_embedder(
    "vllm/BAAI/bge-small-en-v1.5",
    base_url="http://localhost:8000"
)
```

## Batch Processing

Process multiple texts efficiently:

### Sequential Processing

```python
import msgflux as mf

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

texts = [
    "The quick brown fox",
    "jumps over the lazy dog",
    "Machine learning is amazing"
]

embeddings = []
for text in texts:
    response = embedder(text)
    embeddings.append(response.consume())

print(f"Generated {len(embeddings)} embeddings")
```

### Parallel Processing

```python
import msgflux as mf
import msgflux.nn.functional as F

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

texts = [
    "Text 1",
    "Text 2",
    "Text 3",
    "Text 4"
]

# Process in parallel
results = F.map_gather(
    embedder,
    args_list=[(text,) for text in texts]
)

# Extract embeddings
embeddings = [r.consume() for r in results]
print(f"Generated {len(embeddings)} embeddings in parallel")
```

### Async Batch Processing

```python
import msgflux as mf
import asyncio

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

async def embed_batch(texts):
    tasks = [embedder.acall(text) for text in texts]
    responses = await asyncio.gather(*tasks)
    return [r.consume() for r in responses]

texts = ["Text 1", "Text 2", "Text 3"]
embeddings = asyncio.run(embed_batch(texts))
```

## Response Caching

Cache embeddings to avoid redundant API calls:

### Enabling Cache

```python
import msgflux as mf

# Enable cache (highly recommended for embeddings)
embedder = mf.Model.text_embedder(
    "openai/text-embedding-3-small",
    enable_cache=True,
    cache_size=1000  # Cache up to 1000 embeddings
)

# First call - hits API
response1 = embedder("machine learning")
print(response1.consume()[:5])

# Second call - returns cached result
response2 = embedder("machine learning")
print(response2.consume()[:5])  # Same result, no API call

# Check cache stats
if embedder._response_cache:
    stats = embedder._response_cache.cache_info()
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
```

## Working with Embeddings

### Cosine Similarity

```python
import msgflux as mf
import numpy as np

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Generate embeddings
text1 = "I love machine learning"
text2 = "Machine learning is great"
text3 = "The weather is nice today"

emb1 = embedder(text1).consume()
emb2 = embedder(text2).consume()
emb3 = embedder(text3).consume()

# Calculate similarities
sim_1_2 = cosine_similarity(emb1, emb2)
sim_1_3 = cosine_similarity(emb1, emb3)

print(f"Similarity (text1, text2): {sim_1_2:.4f}")  # ~0.85
print(f"Similarity (text1, text3): {sim_1_3:.4f}")  # ~0.30
```

### Semantic Search

```python
import msgflux as mf
import numpy as np

def semantic_search(query, documents, embedder, top_k=3):
    """Find most similar documents to query."""

    # Embed query
    query_emb = np.array(embedder(query).consume())

    # Embed all documents
    doc_embs = [np.array(embedder(doc).consume()) for doc in documents]

    # Calculate similarities
    similarities = [
        np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        for doc_emb in doc_embs
    ]

    # Get top-k
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(documents[i], similarities[i]) for i in top_indices]

# Example usage
embedder = mf.Model.text_embedder("openai/text-embedding-3-small", enable_cache=True)

documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "The weather is sunny today",
    "Neural networks are powerful",
    "I like to eat pizza"
]

query = "Tell me about AI and ML"
results = semantic_search(query, documents, embedder)

for doc, score in results:
    print(f"{score:.4f}: {doc}")
# 0.7234: Machine learning uses algorithms
# 0.6891: Neural networks are powerful
# 0.4123: Python is a programming language
```

## RAG Integration

Embeddings are essential for Retrieval-Augmented Generation:

### Building a Simple RAG System

```python
import msgflux as mf
import numpy as np

class SimpleRAG:
    def __init__(self, embedder_model, chat_model):
        self.embedder = embedder_model
        self.chat = chat_model
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs):
        """Add documents to knowledge base."""
        self.documents.extend(docs)

        # Generate embeddings
        for doc in docs:
            emb = self.embedder(doc).consume()
            self.embeddings.append(np.array(emb))

    def retrieve(self, query, top_k=3):
        """Retrieve most relevant documents."""
        query_emb = np.array(self.embedder(query).consume())

        similarities = [
            np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            for doc_emb in self.embeddings
        ]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def query(self, question):
        """Ask question with RAG."""
        # Retrieve relevant docs
        context_docs = self.retrieve(question)
        context = "\n\n".join(context_docs)

        # Generate answer with context
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

        response = self.chat(messages=[{"role": "user", "content": prompt}])
        return response.consume()

# Usage
embedder = mf.Model.text_embedder("openai/text-embedding-3-small", enable_cache=True)
chat = mf.Model.chat_completion("openai/gpt-4o")

rag = SimpleRAG(embedder, chat)

# Add knowledge
rag.add_documents([
    "msgflux is a Python library for building AI systems.",
    "The Model class provides unified access to different AI providers.",
    "AutoParams allows dataclass-style module definitions.",
    "msgflux supports OpenAI, Anthropic, and Google models."
])

# Ask questions
answer = rag.query("What is msgflux?")
print(answer)
```

## Dimensions and Performance

### Choosing Dimensions

```python
import msgflux as mf

# Higher dimensions = better accuracy, more storage/compute
embedder_large = mf.Model.text_embedder(
    "openai/text-embedding-3-large",
    dimensions=3072  # Full size
)

# Lower dimensions = faster, less storage, slightly lower accuracy
embedder_small = mf.Model.text_embedder(
    "openai/text-embedding-3-small",
    dimensions=256   # Reduced from 1536
)

# Trade-off example:
# - 3072 dims: 99% accuracy, 12x storage
# - 1536 dims: 98% accuracy, 6x storage
# - 512 dims:  95% accuracy, 2x storage
# - 256 dims:  92% accuracy, 1x storage
```

### Performance Comparison

```python
import time
import msgflux as mf

text = "Sample text for embedding"

# Full dimensions
embedder_full = mf.Model.text_embedder(
    "openai/text-embedding-3-small",
    dimensions=1536
)

start = time.time()
emb_full = embedder_full(text).consume()
time_full = time.time() - start

# Reduced dimensions
embedder_reduced = mf.Model.text_embedder(
    "openai/text-embedding-3-small",
    dimensions=256
)

start = time.time()
emb_reduced = embedder_reduced(text).consume()
time_reduced = time.time() - start

print(f"Full (1536d): {time_full:.4f}s, {len(emb_full)} dims")
print(f"Reduced (256d): {time_reduced:.4f}s, {len(emb_reduced)} dims")
print(f"Size reduction: {len(emb_reduced)/len(emb_full)*100:.1f}%")
```

## Response Metadata

Access usage and cost information:

```python
import msgflux as mf
from msgflux.models.profiles import get_model_profile

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

response = embedder("This is a test sentence")

# Check metadata
print(response.metadata)
# {'usage': {'prompt_tokens': 5, 'total_tokens': 5}}

# Calculate cost
profile = get_model_profile("text-embedding-3-small", provider_id="openai")
if profile:
    tokens = response.metadata.usage.total_tokens
    cost = tokens * profile.cost.input_per_token
    print(f"Cost: ${cost:.6f}")
```

## Best Practices

### 1. Enable Caching

```python
# Good - Cache embeddings (they're deterministic)
embedder = mf.Model.text_embedder(
    "openai/text-embedding-3-small",
    enable_cache=True,
    cache_size=5000  # Adjust based on corpus size
)
```

### 2. Batch Similar Texts

```python
# Good - Process similar workloads together
import msgflux.nn.functional as F

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Batch all product descriptions
product_embeddings = F.map_gather(
    embedder,
    args_list=[(desc,) for desc in product_descriptions]
)

# Batch all user queries separately
query_embeddings = F.map_gather(
    embedder,
    args_list=[(query,) for query in user_queries]
)
```

### 3. Choose Appropriate Dimensions

```python
# Good - Balance accuracy vs performance/cost
# For production search with millions of vectors:
embedder = mf.Model.text_embedder(
    "openai/text-embedding-3-small",
    dimensions=512  # Good balance
)

# For high-accuracy semantic tasks:
embedder = mf.Model.text_embedder(
    "openai/text-embedding-3-large",
    dimensions=3072  # Maximum accuracy
)
```

### 4. Normalize for Cosine Similarity

```python
import numpy as np

def normalize_embedding(embedding):
    """Normalize embedding for cosine similarity."""
    embedding = np.array(embedding)
    return embedding / np.linalg.norm(embedding)

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Normalize embeddings once
emb = normalize_embedding(embedder("text").consume())

# Now dot product = cosine similarity
similarity = np.dot(emb1, emb2)  # Faster than full cosine formula
```

### 5. Handle Long Texts

```python
import msgflux as mf

def chunk_text(text, max_tokens=8000):
    """Split text into chunks under token limit."""
    # Simple word-based chunking (use tiktoken for accurate count)
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens // 4:  # Rough estimate
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def embed_long_text(text, embedder):
    """Embed long text by chunking and averaging."""
    chunks = chunk_text(text)

    embeddings = []
    for chunk in chunks:
        emb = embedder(chunk).consume()
        embeddings.append(np.array(emb))

    # Average embeddings
    return np.mean(embeddings, axis=0).tolist()

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")
long_text = "..." * 10000  # Very long text

embedding = embed_long_text(long_text, embedder)
```

## Common Patterns

### Document Deduplication

```python
import msgflux as mf
import numpy as np

def find_duplicates(documents, embedder, threshold=0.95):
    """Find duplicate documents based on embedding similarity."""
    embeddings = [
        np.array(embedder(doc).consume())
        for doc in documents
    ]

    duplicates = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )

            if similarity >= threshold:
                duplicates.append((i, j, similarity))

    return duplicates

embedder = mf.Model.text_embedder("openai/text-embedding-3-small", enable_cache=True)

docs = [
    "Python is great",
    "Python is awesome",  # Very similar
    "Java is also good",
    "The weather is nice"
]

dupes = find_duplicates(docs, embedder)
for i, j, sim in dupes:
    print(f"Documents {i} and {j} are {sim:.2%} similar")
```

### Clustering

```python
import msgflux as mf
import numpy as np
from sklearn.cluster import KMeans

def cluster_documents(documents, embedder, n_clusters=3):
    """Cluster documents using K-means on embeddings."""
    # Generate embeddings
    embeddings = [
        embedder(doc).consume()
        for doc in documents
    ]

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Group by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for doc, label in zip(documents, labels):
        clusters[label].append(doc)

    return clusters

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

docs = [
    "Python programming",
    "Machine learning with Python",
    "Cooking recipes",
    "Italian cuisine",
    "Deep learning tutorial",
    "Baking bread"
]

clusters = cluster_documents(docs, embedder, n_clusters=2)
for cluster_id, docs in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")
```

## Error Handling

```python
import msgflux as mf

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

try:
    response = embedder("Some text")
    embedding = response.consume()
except ImportError:
    print("Provider not installed")
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## See Also

- [Model](model.md) - Model factory and registry
- [Chat Completion](chat_completion.md) - Chat models
- [Data Retrievers](../data/retrievers.md) - Vector databases and retrievers
- [RAG Patterns](../patterns/rag.md) - RAG implementation patterns
