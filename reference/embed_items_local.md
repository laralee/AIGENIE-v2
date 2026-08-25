# Embed Items Using Local Transformer Models

Generates raw embeddings using locally loaded BERT-family models. These
are raw encoder outputs, not similarity-optimized embeddings.

## Usage

``` r
embed_items_local(
  embedding.model,
  items,
  pooling.strategy = "mean",
  device = "auto",
  batch.size = 32,
  max.length = 512,
  silently = FALSE
)
```

## Arguments

- embedding.model:

  Character string specifying the model

- items:

  Data frame with 'statement' and 'ID' columns

- pooling.strategy:

  Character. One of "mean", "cls", "max"

- device:

  Character. One of "auto", "cpu", "cuda", "mps"

- batch.size:

  Integer. Batch size for processing

- max.length:

  Integer. Maximum sequence length

- silently:

  Logical. Suppress progress messages?

## Value

A list with 'embeddings' matrix and 'success' flag
