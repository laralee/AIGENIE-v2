# Embed Items Using Jina AI Embedding API

Generates embeddings using Jina AI's embedding models via their REST API
at `https://api.jina.ai/v1/embeddings`. Uses Bearer token auth and
supports Jina-specific features: task adapters, Matryoshka dimension
truncation, and late chunking.

The Jina API follows an OpenAI-compatible request/response schema, with
additional parameters for task type and output dimensions.

## Usage

``` r
embed_items_jina(
  embedding.model = "jina-embeddings-v3",
  jina_api_key,
  items,
  task = "text-matching",
  dimensions = NULL,
  silently = FALSE
)
```

## Arguments

- embedding.model:

  Jina embedding model name (e.g., "jina-embeddings-v3")

- jina_api_key:

  Jina AI API key

- items:

  Data frame with 'statement' and 'ID' columns

- task:

  Character. Task adapter for optimized embeddings. One of:

  "text-matching"

  :   Sentence similarity (default for AIGENIE)

  "retrieval.query"

  :   Encode queries for retrieval

  "retrieval.passage"

  :   Encode passages for indexing

  "classification"

  :   Text classification

  "separation"

  :   Clustering or reranking

- dimensions:

  Optional integer. Output embedding dimensions for Matryoshka-capable
  models (v3: 256-1024, v4: 128-2048). NULL uses the model default (v3:
  1024, v4: 2048).

- silently:

  Logical. Suppress progress messages?

## Value

A list with 'embeddings' matrix and 'success' flag
