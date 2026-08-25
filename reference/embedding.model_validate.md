# Validate and Detect Embedding Model Provider

Determines which provider to use for embeddings based on the model name.

Validates that the embedding model is one of the supported OpenAI, Jina
AI, or HuggingFace models.

## Usage

``` r
embedding.model_validate(embedding.model, provider = "auto", hf.token = NULL)

embedding.model_validate(embedding.model, provider = "auto", hf.token = NULL)
```

## Arguments

- embedding.model:

  A string.

- provider:

  One of "auto", "openai", "jina", "huggingface", or "local".

## Value

Character string: "openai", "jina", "huggingface", or "local"

## Details

Allowed OpenAI models:

- "text-embedding-3-small"

- "text-embedding-3-large"

- "text-embedding-ada-002"

Allowed Jina AI models:

- jina-embeddings-v4, jina-embeddings-v3, jina-clip-v2

- jina-code-embeddings-1.5b, jina-code-embeddings-0.5b

- jina-embeddings-v2-base-en/zh/de/es/code, jina-embeddings-v2-small-en

Allowed HuggingFace models:

- BAAI/bge series (bge-small-en-v1.5, bge-base-en-v1.5,
  bge-large-en-v1.5)

- thenlper/gte series (gte-small, gte-base, gte-large)
