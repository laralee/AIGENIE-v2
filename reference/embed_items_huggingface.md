# Embed Items Using HuggingFace Models

Generates embeddings using HuggingFace models. Tries the Inference API
first, then falls back to the sentence-transformers library for
unsupported models.

## Usage

``` r
embed_items_huggingface(
  embedding.model = "BAAI/bge-base-en-v1.5",
  hf.token = NULL,
  items,
  silently = FALSE
)
```

## Arguments

- embedding.model:

  HuggingFace model name

- hf.token:

  Optional HuggingFace API token

- items:

  Data frame with 'statement' and 'ID' columns

- silently:

  Logical. Suppress progress messages?

## Value

A list with 'embeddings' matrix and 'success' flag
