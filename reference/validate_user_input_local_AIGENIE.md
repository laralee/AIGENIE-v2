# Validate All User Inputs for Local AI-GENIE

Comprehensive validation of all inputs for local model execution. Reuses
existing validators where applicable and adds local-specific
validations.

## Usage

``` r
validate_user_input_local_AIGENIE(
  item.attributes,
  model.path,
  embedding.model,
  main.prompts,
  temperature,
  top.p,
  target.N,
  domain,
  scale.title,
  item.examples,
  audience,
  item.type.definitions,
  response.options,
  prompt.notes,
  system.role,
  EGA.model,
  EGA.algorithm,
  EGA.uni.method,
  n.ctx,
  n.gpu.layers,
  max.tokens,
  device,
  batch.size,
  pooling.strategy,
  max.length,
  keep.org,
  items.only,
  embeddings.only,
  adaptive,
  run.overall,
  all.together,
  plot,
  silently
)
```

## Arguments

- item.attributes:

  Named list of attributes (same as API version)

- model.path:

  Path to local GGUF model

- embedding.model:

  Local embedding model identifier

- main.prompts:

  Optional custom prompts

- temperature:

  LLM temperature

- top.p:

  LLM top-p sampling

- target.N:

  Target number of items

- domain:

  Assessment domain

- scale.title:

  Scale name

- item.examples:

  Example items

- audience:

  Target audience

- item.type.definitions:

  Type definitions

- response.options:

  Response scale options

- prompt.notes:

  Additional prompt instructions

- system.role:

  System prompt

- EGA.model:

  EGA model type

- EGA.algorithm:

  EGA algorithm

- EGA.uni.method:

  EGA unidimensionality method

- n.ctx:

  Context window size

- n.gpu.layers:

  GPU layers

- max.tokens:

  Maximum generation tokens

- device:

  Embedding computation device

- batch.size:

  Embedding batch size

- pooling.strategy:

  Embedding pooling strategy

- max.length:

  Embedding max sequence length

- keep.org:

  Keep original data

- items.only:

  Generate items only

- embeddings.only:

  Generate embeddings only

- adaptive:

  Use adaptive generation

- run.overall:

  Logical. Whether to fit an additional pooled EGA to items retained
  after item-type-level reduction.

- all.together:

  Logical. Whether to run the reduction pipeline on all item types
  together rather than separately.

- plot:

  Show plots

- silently:

  Suppress messages

## Value

A list of all validated parameters
