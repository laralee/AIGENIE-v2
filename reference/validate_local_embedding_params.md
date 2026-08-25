# Validate Local Embedding Parameters

Validates parameters specific to local embedding generation

## Usage

``` r
validate_local_embedding_params(
  device,
  batch.size,
  pooling.strategy,
  max.length
)
```

## Arguments

- device:

  Device for computation ("auto", "cpu", "cuda", "mps")

- batch.size:

  Number of items to process simultaneously

- pooling.strategy:

  Strategy for pooling token embeddings

- max.length:

  Maximum sequence length for tokenization

## Value

A list of validated parameters
