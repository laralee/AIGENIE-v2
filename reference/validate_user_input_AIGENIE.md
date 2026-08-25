# Validate All User Inputs for AI-GENIE

This function performs comprehensive validation and normalization of all
user-supplied inputs to the AI-GENIE package. It checks logical flags,
strings, model names, item attribute structures, and ensures consistency
across all interdependent components.

## Usage

``` r
validate_user_input_AIGENIE(
  item.attributes,
  openai.API,
  hf.token,
  main.prompts,
  groq.API,
  anthropic.API,
  jina.API,
  model,
  temperature,
  top.p,
  embedding.model,
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

  A named list of attributes and item types. Must be validated via
  `item.attributes_validate()`.

- openai.API:

  A string. OpenAI API key.

- hf.token:

  A string. HuggingFace API key.

- main.prompts:

  A named list of custom prompts that the user specifies (if desired)

- groq.API:

  A string or NULL. Groq API key.

- anthropic.API:

  Character. Anthropic API key. Can be NULL when Anthropic models are
  not used.

- jina.API:

  Character. Jina AI API key. Can be NULL when Jina embeddings are not
  used.

- model:

  A string. The user-specified language model. Will be resolved to a
  canonical model name using
  [`normalize_model_name()`](https://laralee.github.io/AIGENIE/reference/normalize_model_name.md).

- temperature:

  A numeric value between 0 and 2.

- top.p:

  A numeric value between 0 and 1.

- embedding.model:

  A string or NULL. Must be one of the accepted OpenAI embedding models.

- target.N:

  Either a scalar integer, NULL, or a named list/vector of integers
  corresponding to each attribute. Used for synthetic item generation.

- domain:

  A string describing the domain of the assessment.

- scale.title:

  A string naming the scale.

- item.examples:

  A data frame containing `type`, `attribute`, and `statement` columns.
  All values must be strings. Optional.

- audience:

  A string or NULL. The intended audience of the assessment.

- item.type.definitions:

  A named list mapping item types to their descriptions. Optional.

- response.options:

  An atomic vector of strings listing the response options users will
  have. Optional.

- prompt.notes:

  A named list or string that gives the LLM additional instructions to
  be appended to the prompt. Optional.

- system.role:

  A string or NULL. Used to customize the system prompt.

- EGA.model:

  A string or NULL. One of `"BGGM"`, `"glasso"`, or `"TMFG"`.

- EGA.algorithm:

  A string. One of `"leiden"`, `"louvain"`, or `"walktrap"`.

- EGA.uni.method:

  A string. One of `"expand"`, `"LE"`, or `"louvain"`.

- keep.org:

  A boolean. If TRUE, preserve original inputs in the output.

- items.only:

  A boolean. Whether to generate only items.

- embeddings.only:

  A boolean. Whether to run in embedding-only mode.

- adaptive:

  A boolean. Whether adaptive design logic should be applied.

- run.overall:

  Logical. Whether to fit an additional pooled EGA to items retained
  after item-type-level reduction.

- all.together:

  Logical. Whether to run the reduction pipeline on all item types
  together rather than separately.

- plot:

  A boolean. Whether to display plots for visual diagnostics.

- silently:

  A boolean. If TRUE, suppresses warning messages.

## Value

A named list containing:

- target.N:

  A named list of integers, aligned with `item.attributes`

- EGA.model:

  Canonical model string or NULL

- EGA.uni.method:

  Canonical unidimensionality method

- EGA.algorithm:

  Canonical community detection algorithm

- model:

  Resolved model string for text generation

- item.type.definitions:

  Cleaned item type definitions (if provided)

- item.examples:

  Cleaned item examples (if provided)

- item.attributes:

  Cleaned and normalized item attributes

- prompt.notes:

  Cleaned and normalized prompt notes (if provided)

- main.prompts:

  Cleaned and normalized main prompts (if provided)

- custom:

  A flag signaling whether we are in custom mode or not

## Details

If any input is invalid or misaligned with the package’s expected
structure, informative errors or warnings are raised. Cleaned and
normalized objects are returned for use downstream.
