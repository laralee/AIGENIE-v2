# Validate Local Embedding Model

Validates that the embedding model is appropriate for local raw
embeddings. Checks for BERT-family models that provide raw feature
extraction.

## Usage

``` r
validate_local_embedding_model(embedding.model, silently = FALSE)
```

## Arguments

- embedding.model:

  Character string specifying model identifier or path

- silently:

  Logical. Suppress informational messages

## Value

The validated model identifier or path
