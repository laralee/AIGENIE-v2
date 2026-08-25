# Validate All User Inputs for Local GENIE

Validate All User Inputs for Local GENIE

## Usage

``` r
validate_user_input_local_GENIE(
  items,
  embedding.matrix,
  embedding.model,
  device,
  batch.size,
  pooling.strategy,
  max.length,
  EGA.model,
  EGA.algorithm,
  EGA.uni.method,
  embeddings.only,
  run.overall,
  all.together,
  plot,
  silently
)
```

## Arguments

- items:

  Data frame with columns: statement, attribute, type, ID

- embedding.matrix:

  Optional numeric matrix of pre-computed item embeddings, with
  embedding dimensions in rows and items in columns.

- embedding.model:

  Local embedding model identifier or path

- device:

  Device for embeddings ("auto", "cpu", "cuda", "mps")

- batch.size:

  Batch size for embedding generation

- pooling.strategy:

  Pooling strategy ("mean", "cls", "max")

- max.length:

  Maximum sequence length for embeddings

- EGA.model:

  EGA network model ("glasso", "TMFG", or NULL)

- EGA.algorithm:

  EGA algorithm ("walktrap", "leiden", "louvain")

- EGA.uni.method:

  EGA unidimensionality method ("louvain", "expand", "LE")

- embeddings.only:

  Whether to stop after embeddings

- run.overall:

  Logical. Whether to fit an additional pooled EGA to items retained
  after item-type-level reduction.

- all.together:

  Logical. Whether to run the reduction pipeline on all item types
  together rather than separately.

- plot:

  Whether to show plots

- silently:

  Whether to suppress messages

## Value

A list of all validated parameters ready for local GENIE execution
