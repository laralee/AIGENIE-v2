# Validate All User Inputs for GENIE

Validate All User Inputs for GENIE

## Usage

``` r
validate_user_input_GENIE(
  items,
  embedding.matrix,
  openai.API,
  hf.token,
  jina.API,
  embedding.model,
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

  A data frame with columns: statement, attribute, type, ID

- embedding.matrix:

  Optional numeric matrix/data frame with items as columns

- openai.API:

  OpenAI API key (string or NULL)

- hf.token:

  HuggingFace token (string or NULL)

- jina.API:

  Jina API key (string or NULL)

- embedding.model:

  Embedding model identifier (string)

- EGA.model:

  EGA network model (string or NULL)

- EGA.algorithm:

  EGA algorithm (string)

- EGA.uni.method:

  EGA unidimensionality method (string)

- embeddings.only:

  Whether to stop after embeddings (boolean)

- run.overall:

  Logical. Whether to fit an additional pooled EGA to items retained
  after item-type-level reduction.

- all.together:

  Logical. Whether to run the reduction pipeline on all item types
  together rather than separately.

- plot:

  Whether to show plots (boolean)

- silently:

  Whether to suppress messages (boolean)

## Value

A named list containing all validated and normalized parameters
