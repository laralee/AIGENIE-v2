# Generate Embeddings Using Any Supported Provider

Unified interface for generating embeddings that automatically routes to
the appropriate provider based on the model name.

## Usage

``` r
generate_embeddings(
  embedding.model,
  items,
  openai.API = NULL,
  hf.token = NULL,
  jina.API = NULL,
  silently = FALSE,
  ...
)
```

## Arguments

- embedding.model:

  Character string specifying the embedding model

- items:

  Data frame with 'statement' and 'ID' columns

- openai.API:

  Optional OpenAI API key

- hf.token:

  Optional HuggingFace token

- jina.API:

  Optional Jina AI API key

- silently:

  Logical. Suppress progress messages?

- ...:

  Additional arguments passed to provider-specific functions

## Value

A list with 'embeddings' matrix and 'success' flag
