# Embed Items Using OpenAI's Embedding API

Generates embeddings using OpenAI's embedding models.

## Usage

``` r
embed_items(embedding.model, openai.API, items, silently)
```

## Arguments

- embedding.model:

  OpenAI embedding model name

- openai.API:

  OpenAI API key

- items:

  Data frame with 'statement' and 'ID' columns

- silently:

  Logical. Suppress progress messages?

## Value

A list with 'embeddings' matrix and 'success' flag
