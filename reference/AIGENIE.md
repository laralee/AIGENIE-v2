# Generate, Validate, and Check Items using AI-GENIE

Generate, validate, and check your items for quality and redundancy
using AI-GENIE (Generative Psychometrics via AI-GENIE: Automatic Item
Generation and Validation via Network-Integrated Evaluation). AI-GENIE
is a methodology that combines the latest open-source LLMs and
generative artificial intelligence with advances in network
psychometrics to facilitate scale generation, selection, and validation.
The pipeline eliminates the need to generate hundreds of items by
content experts, recruit diverse and experienced researchers, administer
items to thousands of participants, and employ modern psychometric
methods in the collected data.

## Usage

``` r
AIGENIE(
  item.attributes,
  openai.API = NULL,
  hf.token = NULL,
  main.prompts = NULL,
  groq.API = NULL,
  anthropic.API = NULL,
  jina.API = NULL,
  model = "gpt4o",
  temperature = 1,
  top.p = 1,
  embedding.model = "text-embedding-3-small",
  target.N = NULL,
  domain = NULL,
  scale.title = NULL,
  item.examples = NULL,
  audience = NULL,
  item.type.definitions = NULL,
  response.options = NULL,
  prompt.notes = NULL,
  system.role = NULL,
  EGA.model = NULL,
  EGA.algorithm = NULL,
  EGA.uni.method = NULL,
  uva.cut.off = 0.2,
  boot.iter = 500,
  ncores = NULL,
  keep.org = FALSE,
  items.only = FALSE,
  embeddings.only = FALSE,
  adaptive = TRUE,
  run.overall = FALSE,
  all.together = FALSE,
  plot = TRUE,
  silently = FALSE
)
```

## Arguments

- item.attributes:

  A named list of atomic character vectors (required). Describes the
  attributes or characteristics that each item type should encompass.
  These are not necessarily lower-order dimensions, but can be if the
  item types represent appropriate hierarchical constructs. Each nested
  list must have at least two unique attributes. Repeated attributes
  within the same nested list are not allowed, but attributes can be
  repeated across nested lists. Each name of the list must be unique,
  and all elements within sublists must be strings. For example,
  attributes of the personality trait "neuroticism" might include
  "anxious", "depressed", "insecure", or "emotional" since these
  characteristics encompass aspects of neuroticism that the item pool
  should address.

- openai.API:

  A character string or NULL (optional, default: NULL). The OpenAI API
  key for authentication with OpenAI's services. Required when using
  OpenAI's platform for either item generation or embedding. If NULL,
  users must provide either `groq.API` for item generation via Groq or
  `hf.token` for embeddings via Hugging Face.

- hf.token:

  A character string or NULL (optional, default: NULL). The Hugging Face
  API token for authentication with Hugging Face services. Required when
  using Hugging Face models for embeddings. If NULL, an `openai.API` key
  must be provided since the user will need to embed via OpenAI.

- main.prompts:

  A named list of character strings or NULL (optional, default: NULL).
  Custom prompts for item generation. If provided, this must be a named
  list where `names(main.prompts)` equals `names(item.attributes)`. Each
  prompt must explicitly mention all attributes found in the associated
  element of `item.attributes`. Users should not include instructions
  regarding layout/formatting of LLM response, as this is handled
  automatically for proper parsing. If NULL, AIGENIE builds appropriate
  prompts automatically based on other prompt-building parameters.

- groq.API:

  A character string or NULL (optional, default: NULL). The Groq API key
  for authentication with Groq's LLM services. Required when users want
  to generate items via Groq's API platform using open-source models.
  Commonly used in combination with `openai.API` since Groq does not
  provide embedding services.

- anthropic.API:

  A character string or NULL (optional, default: NULL). The Anthropic
  API key for authentication with Anthropic's Claude models. Required
  when using Claude models (e.g., "sonnet", "opus", "haiku") for item
  generation. Get a key at
  <https://platform.claude.com/docs/en/manage-claude/authentication>.

- jina.API:

  A character string or NULL (optional, default: NULL). The Jina AI API
  key for authentication with Jina's embedding services. Required when
  using Jina embedding models (e.g., "jina-embeddings-v3",
  "jina-embeddings-v4"). Free tier available at <https://jina.ai/>.

- model:

  A character string (optional, default: "gpt4o"). Specifies which large
  language model to use for item generation. Supports models from
  multiple providers:

  - **OpenAI**: `"gpt-4o"`, `"gpt-4"`, `"gpt-3.5-turbo"`, `"o1"`,
    `"o1-mini"`

  - **Anthropic**: `"sonnet"`, `"opus"`, `"haiku"`, or full names like
    `"claude-sonnet-4-5-20250929"`

  - **Groq**: `"llama-3.3-70b-versatile"`, `"mixtral-8x7b-32768"`,
    `"gemma2-9b-it"`, `"deepseek-r1-distill-llama-70b"`,
    `"qwen-2.5-72b"`

  Aliases like `"llama"`, `"mixtral"`, `"gemma"`, `"deepseek"`,
  `"claude"` are also accepted. The function automatically determines
  which API service to use based on the model name and available API
  keys.

- temperature:

  A numeric value (optional, default: 1). Controls the randomness and
  creativity of the LLM's item generation. Must be between 0-2, where
  lower values produce more deterministic outputs and higher values
  increase creativity and variability.

- top.p:

  A numeric value (optional, default: 1). Controls nucleus sampling for
  the LLM's text generation. Must be between 0-1, where lower values
  make the model more focused and higher values allow more diverse
  outputs. Can be used in conjunction with `temperature`.

- embedding.model:

  A character string (optional, default: "text-embedding-3-small").
  Specifies which model to use for generating embeddings of items.
  Supports multiple providers:

  - **OpenAI**: `"text-embedding-3-small"`, `"text-embedding-3-large"`,
    `"text-embedding-ada-002"`

  - **Jina AI**: `"jina-embeddings-v3"`, `"jina-embeddings-v4"`,
    `"jina-embeddings-v2-base-en"` (requires `jina.API`)

  - **HuggingFace**: `"BAAI/bge-small-en-v1.5"`,
    `"BAAI/bge-base-en-v1.5"`, `"thenlper/gte-base"`,
    `"sentence-transformers/all-MiniLM-L6-v2"`

  The provider is automatically detected based on the model name. Jina
  models support task adapters and Matryoshka dimension truncation for
  optimized embeddings.

- target.N:

  An integer, named list of integers, or NULL (optional, default: NULL).
  Specifies the number of items to generate for each item type. Can be a
  single integer (applies to all item types) or a named list of integers
  where `names(target.N)` equals `names(item.attributes)` for different
  numbers per item type. If NULL, 60 items per item type will be
  generated. A rule of thumb is about 60 items or more per item type for
  meaningful reduction analysis.

- domain:

  A character string or NULL (optional, default: NULL). Specifies the
  psychological or research domain for context in item generation.
  Should be specific (e.g., "personality", "child development") rather
  than general. If supplied, it will be used to construct appropriate
  prompts and system roles unless `system.role` is provided.

- scale.title:

  A character string or NULL (optional, default: NULL). Specifies the
  name or title of the scale being developed. Can be formal or
  descriptive, but more specific titles generally produce better
  results. If supplied, it will be used to construct appropriate prompts
  and system roles unless `system.role` is provided.

- item.examples:

  A data frame or NULL (optional, default: NULL). Provides example items
  to guide the LLM's generation style and format. Must be a data frame
  with columns: `statement` (the actual item), `attribute` (the item's
  attribute), and `type` (the item's type). All values must be non-empty
  strings, and the `attribute` and `type` must align with the
  `item.attributes` object. Items should be extremely high quality and
  validated if possible, as they serve as style templates.

- audience:

  A character string or NULL (optional, default: NULL). Specifies the
  target population for the scale being developed. Should be as specific
  as possible (e.g., "educated adults in rural America", "children with
  ASD in second grade") rather than general demographic categories. If
  supplied, it will be used to construct appropriate prompts and system
  roles unless `system.role` is provided.

- item.type.definitions:

  A named list of character strings or NULL (optional, default: NULL).
  Provides definitions or descriptions of each item type for the LLM.
  Must be a named list where `names(item.type.definitions)` equals
  `names(item.attributes)`. Useful when constructs or item types are
  obscure or potentially ambiguous, helping the LLM understand the item
  type or construct in your specific context. If supplied, it will be
  used to construct appropriate prompts and system roles unless
  `system.role` is provided.

- response.options:

  A character vector or NULL (optional, default: NULL). Specifies the
  response scale labels for the generated items (e.g., c("agree",
  "neither agree nor disagree", "disagree")). These labels provide
  context for item writing but do not appear in the actual items
  themselves. If supplied, it will be used to construct appropriate
  system roles unless `system.role` is provided.

- prompt.notes:

  A named list of character strings, character string, or NULL
  (optional, default: NULL). Allows users to add custom instructions or
  context to the prompts. Can be a named list where
  `names(prompt.notes)` equals `names(item.attributes)` for different
  notes per item type, or a single string applied to all item types.
  These notes are appended at the end of constructed prompts, allowing
  users to add brief additional requirements (e.g., "All items MUST
  begin with the stem 'I am someone who...'") without creating entirely
  custom prompts. Should be brief; otherwise, users should consider
  using `main.prompts`.

- system.role:

  A character string or NULL (optional, default: NULL). Defines the
  system role/persona for the LLM during item generation. If not
  provided, one is built automatically based on prompt-building
  parameters. Should be as specific as possible (e.g., "You are an
  expert scale developer and psychometrician with extensive expertise in
  drafting Likert-type items for children with ASD. Today, you will
  focus on developing robust, single-statement items that assess
  linguistic ability."). Applies to all LLM interactions.

- EGA.model:

  A character string or NULL (optional, default: NULL). Specifies which
  model to use for Exploratory Graph Analysis network construction.
  Valid options are "tmfg" or "glasso". If NULL, AIGENIE will test both
  "tmfg" and "glasso" models and automatically return the model that
  maximizes NMI (normalized mutual information). TMFG is a greedy but
  speedy network-building algorithm that works well for many
  applications, especially text. EBICglasso is slower but non-greedy and
  may capture more nuanced relationships.

- EGA.algorithm:

  A character string (optional, default is "walktrap" when there is a
  single trait and "louvain" when there is more than one trait).
  Specifies which community detection algorithm to use within the EGA
  framework. Valid options are "louvain", "walktrap", or "leiden". The
  algorithm operates separately from the network building specified by
  `EGA.model`.

- EGA.uni.method:

  A character string (optional, default: "louvain"). Specifies the
  method for handling unidimensional structures in EGA. Valid options
  are: "expand" (expands correlation matrix with four variables
  correlated 0.50; if dimensions \\\le\\ 2, data are unidimensional),
  "LE" (applies Leading Eigenvector algorithm; if dimensions = 1, uses
  LE solution), or "louvain" (applies Louvain algorithm; if dimensions =
  1, uses Louvain solution). This parameter is rarely modified by users.

- uva.cut.off:

  A numeric value in `[0, 1)` (optional, default: `0.20`). The weighted
  topological overlap threshold passed to
  [`EGAnet::UVA`](https://rdrr.io/pkg/EGAnet/man/UVA.html) during the
  redundancy-reduction step. Items with pairwise wTO at or above this
  value are flagged as redundant. Lower values are more aggressive
  (remove more items); higher values are more conservative.

- boot.iter:

  A positive integer (optional, default: 500). Number of bootstrap
  iterations used by
  [`EGAnet::bootEGA`](https://rdrr.io/pkg/EGAnet/man/bootEGA.html)
  during item-stability analyses and iterative stability filtering.

- ncores:

  A positive integer or `NULL` (optional, default: `NULL`). Number of
  processing cores passed to
  [`EGAnet::bootEGA`](https://rdrr.io/pkg/EGAnet/man/bootEGA.html). When
  `NULL`, AIGENIE does not pass an `ncores` argument, preserving the
  current default behavior of
  [`EGAnet::bootEGA`](https://rdrr.io/pkg/EGAnet/man/bootEGA.html).

- keep.org:

  A logical value (optional, default: FALSE). Controls whether the
  pre-reduced items generated by the model are returned to the user. If
  TRUE, returns the full item pool before psychometric reduction. Does
  not affect the reduction process.

- items.only:

  A logical value (optional, default: FALSE). Controls whether the
  function only generates items without running the full psychometric
  pipeline. If TRUE, skips embedding, EGA, and reduction steps,
  returning only a data frame with columns `ID`, `statement`, `type`,
  and `attribute`. Useful when users want to generate items with
  AIGENIE, embed them elsewhere, and use the `GENIE` function for
  reduction.

- embeddings.only:

  A logical value (optional, default: FALSE). Controls whether the
  function generates items and embeddings but skips psychometric
  reduction. If TRUE, returns a named list with `embeddings` (the
  embedding matrix) and `items` (the items data frame). If both
  `items.only` and `embeddings.only` are TRUE, defaults to
  `embeddings.only` behavior.

- adaptive:

  A logical value (optional, default: TRUE). Controls whether previously
  generated items are incorporated into subsequent prompts to reduce
  redundancy. Items are generated in batches to avoid context window
  limitations, potentially requiring multiple API calls. When TRUE,
  appends previously generated items so the model knows what has been
  generated to avoid repetition. Should always be enabled unless context
  limitations are a concern.

- run.overall:

  A logical value (optional, default: FALSE). Controls whether a *fit*
  analysis on the complete item pool is run *post-reduction.* By
  default, only type-level reduction analyses are run (i.e., items of
  like-type go through the pipeline independent of the other items in
  the pool). When this flag is `TRUE`, an additional analysis is run on
  the overall sample, but no further reductions at the overall level are
  made. If only one item type is present, this argument will be ignored.

- all.together:

  A logical value (optional, default: FALSE). Controls whether the
  *reduction* analysis on the complete item pool is run. By default,
  only type-level reduction analyses are run (i.e., items of like-type
  go through the pipeline independent of the other items in the pool).
  When this flag is `TRUE`, reductions are made at the overall level
  (i.e., all items go through the reduction pipeline together, agnostic
  of item type). If only one item type is present, this argument will be
  ignored.

- plot:

  A logical value (optional, default: TRUE). Controls whether
  visualizations are generated and displayed. When TRUE, generates EGA
  network comparison plots (before vs after reduction) for each item
  type and the sample overall. Plots are always saved and returned in
  the output object but can be suppressed from display for cleaner
  output.

- silently:

  A logical value (optional, default: FALSE). Controls console output
  and messaging during function execution. When TRUE, suppresses
  progress statements about item generation, embedding, and pipeline
  reduction. Does not affect warnings or errors, only informational
  messages. Operates independently of the `plot` parameter.

## Value

The structure of the return value depends on the function flags.

**Defaults:** `items.only = FALSE`, `embeddings.only = FALSE`,
`run.overall = FALSE`, `keep.org = FALSE`, `all.together = FALSE`.

**When `items.only = TRUE`:** Returns a `data.frame` of generated items
with columns: `ID`, `statement`, `type`, and `attribute`.

**When `embeddings.only = TRUE`:** Returns a named `list` with two
elements:

- `embeddings` — an embedding matrix/list (columns or rownames
  correspond to item IDs).

- `items` — the items `data.frame` described above.

**Default behaviour** (`items.only = FALSE`, `embeddings.only = FALSE`,
`run.overall = FALSE`, `keep.org = FALSE`, `all.together = FALSE`):
Returns a named `list` with two top-level elements:

- `item_type_level`:

  A named list where each name is an item type and each element is a
  per-type named list containing:

  `final_NMI`

  :   Numeric: final normalized mutual information after reduction.

  `initial_NMI`

  :   Numeric: initial NMI of the pre-reduced item pool.

  `embeddings`

  :   List or matrix of embeddings for this item type (see 'Notes on
      `embeddings`' below).

  `UVA`

  :   List from Unique Variable Analysis (contains at least `n_removed`,
      `n_sweeps`, `redundant_pairs` data.frame).

  `bootEGA`

  :   List with bootEGA results (e.g. `initial_boot`, `final_boot`,
      `n_removed`, `items_removed`, `initial_boot_with_redundancies`).

  `EGA.model_selected`

  :   Character: chosen EGA model (e.g. `"TMFG"` or `"Glasso"`).

  `final_items`

  :   `data.frame`: final items after reduction (columns include `ID`,
      `statement`, `attribute`, `type`, `EGA_com`).

  `final_EGA`

  :   EGA object (from EGAnet) after reduction.

  `initial_EGA`

  :   Initial EGA object computed on the pre-reduced item set.

  `start_N`

  :   Integer: initial number of items in this type.

  `final_N`

  :   Integer: final number of items in this type.

  `network_plot`

  :   `ggplot` / `patchwork` object comparing networks before vs after
      reduction.

  `stability_plot`

  :   `ggplot` / `patchwork` object showing item stability before vs
      after reduction.

- `overall`:

  Named list with aggregated results across all item types. Under the
  default this contains:

  `final_items`

  :   `data.frame` of final items across all types (columns as above).

  `embeddings`

  :   Embeddings for the full reduced item set (see 'Notes on
      `embeddings`' below). Note: `overall$embeddings` does **not**
      include `selected`.

**When `keep.org = TRUE`** (in addition to defaults above): The
top-level shape remains (`item_type_level` and `overall`) but includes
original (pre-reduction) information:

- `item_type_level`:

  Each per-type sublist contains: `final_NMI`, `initial_NMI`,
  `embeddings`, `UVA`, `bootEGA`, `EGA.model_selected`, `final_items`,
  `initial_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`,
  `network_plot`, `stability_plot`.

- `overall`:

  Contains `final_items`, `initial_items`, and `embeddings` for the full
  item pool.

For `keep.org = TRUE`, per-type `embeddings` contains at least:
`full_org`, `sparse_org`, `selected`, `full`, and `sparse`.
(`overall$embeddings` contains the same subcomponents **except**
`selected` is omitted.)

**When `run.overall = TRUE`** (`items.only = FALSE`,
`embeddings.only = FALSE`):

- `item_type_level`:

  Same per-type structure as the default (see above).

- `overall`:

  A named list with aggregated results (not limited to `final_items` and
  `embeddings`) containing: `final_NMI`, `initial_NMI`, `embeddings`,
  `EGA.model_selected`, `final_items`, `final_EGA`, `initial_EGA`,
  `start_N`, `final_N`, and `network_plot`.

**When `all.together = TRUE`** (regardless of `run.overall`): Results
are **not** split into `item_type_level` and `overall`. Instead the
function returns a single named list (applies to the full — possibly
`keep.org` modified — result set) containing: `final_NMI`,
`initial_NMI`, `embeddings`, `UVA`, `bootEGA`, `EGA.model_selected`,
`final_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`,
`network_plot`, and `stability_plot`.

## References

Golino, H. F., & Epskamp, S. (2017). Exploratory graph analysis: A new
approach for estimating the number of dimensions in psychological
research. *PLOS ONE, 12*(6), e0174035.
[doi:10.1371/journal.pone.0174035](https://doi.org/10.1371/journal.pone.0174035)

Christensen, A. P., Garrido, L. E., & Golino, H. (2023). Unique variable
analysis: A network psychometrics method to detect local dependence.
*Multivariate Behavioral Research, 58*(6), 1165–1182.
[doi:10.1080/00273171.2023.2194606](https://doi.org/10.1080/00273171.2023.2194606)

Christensen, A. P., & Golino, H. (2021). Estimating the stability of
psychological dimensions via bootstrap exploratory graph analysis: A
Monte Carlo simulation and tutorial. *Psych, 3*(3), 479–500.
[doi:10.3390/psych3030032](https://doi.org/10.3390/psych3030032)

Danon, L., Díaz-Guilera, A., Duch, J., & Arenas, A. (2005). Comparing
community structure identification. *Journal of Statistical Mechanics:
Theory and Experiment, 2005*(9), P09008.
[doi:10.1088/1742-5468/2005/09/P09008](https://doi.org/10.1088/1742-5468/2005/09/P09008)

Russell-Lasalandra, L. L., Christensen, A. P., & Golino, H. F. (2026).
Generative psychometrics via AI-GENIE: Automatic item generation and
validation with network-integrated evaluation. *Behavior Research
Methods*, *58*(8), 217.
[doi:10.3758/s13428-026-03082-1](https://doi.org/10.3758/s13428-026-03082-1)

## Examples

``` r
if (FALSE) { # \dontrun{
########################################################
#### Example 1: Using AI-GENIE with Default Prompts ####
########################################################

# Add an OpenAI API key
key <- "INSERT YOUR KEY HERE"

# Item type definitions
trait.definitions <- list(
  neuroticism = paste0(
    "Neuroticism is a personality trait that describes one's ",
    "tendency to experience negative emotions like anxiety, ",
    "depression, irritability, anger, and self-consciousness."
  ),
  openness = paste0(
    "Openness is a personality trait that describes how ",
    "open-minded, creative, and imaginative a person is."
  ),
  extraversion = paste0(
    "Extraversion is a personality trait that describes people ",
    "who are more focused on the external world than their ",
    "internal experience."
  )
)

# Item attributes
aspects.of.personality.traits <- list(
  neuroticism = c("anxious", "depressed", "insecure", "emotional"),
  openness = c("creative", "perceptual", "curious", "philosophical"),
  extraversion = c("friendly", "positive", "assertive", "energetic")
)

# Name the field or specialty
domain <- "Personality Measurement"

# Name the Inventory being created
scale.title <- "Three of 'Big Five:' A Streamlined Personality Inventory"

# Run AI-GENIE to generate, validate, and redundancy-check an item pool for your new scale.
personality.inventory.results <- AIGENIE(
  item.attributes = aspects.of.personality.traits,
  openai.API = key,
  domain = domain,
  scale.title = scale.title,
  item.type.definitions = trait.definitions
)

# View the final item pool
View(personality.inventory.results)


#######################################################
#### Example 2: Using AI-GENIE with Custom Prompts ####
#######################################################


# Define a custom system role
system.role <- paste0(
  "You are an expert methodologist who specializes in scale ",
  "development for personality measurement. You are especially ",
  "equipped to create novel personality items that mimic the ",
  "style of popular 'Big Five' assessments."
)

# Define custom prompts for each personality trait
custom.personality.prompts <- list(

  # Prompt for generating neuroticism traits
  neuroticism = paste0(
    "Generate unique, psychometrically robust single-statement items designed to assess ",
    "the Big Five personality trait neuroticism.",
    paste0(
      "Neuroticism has the following characteristics: anxious, ",
      "depressed, insecure, and emotional. "
    )
  ),

  # Prompt for generating openness traits
  openness = paste0(
    "Generate unique, psychometrically robust single-statement items designed to assess ",
    "the Big Five personality trait openness.",
    paste0(
      "Openness has the following characteristics: creative, ",
      "perceptual, curious, and philosophical"
    )
  ),

  # Prompt for generating extraversion traits
  extraversion = paste0(
    "Generate unique, psychometrically robust single-statement items designed to assess ",
    "the Big Five personality trait extraversion.",
    paste0(
      "Extraversion has the following characteristics: friendly, ",
      "positive, assertive, and energetic."
    )
  )

)

# Run AI-GENIE to generate, validate, and redundancy-check an item pool for your new scale.
personality.inventory.results.custom <- AIGENIE(
  item.attributes = aspects.of.personality.traits, # created in example 1
  main.prompts = custom.personality.prompts,
  system.role = system.role,
  openai.API = key, # created in example 1
  scale.title = scale.title # created in example 1
)

# View the final item pool
View(personality.inventory.results.custom)

################################################################
###### Or, Run AIGENIE with an Open Source Model via Groq ######
################################################################

# Add your API Key from Groq
groq.key <- "INSERT YOUR KEY HERE"

# Chose an open-source model like 'DeepSeek' or 'GPT oss'
open.source.model <- "GPT oss 120b"

# Use AIGENIE with an open source model via Groq
personality.inventory.results.gptoss <- AIGENIE(
  item.attributes = aspects.of.personality.traits, # created in example 1
  openai.API = key, # Created in example 1
  domain = domain, # Created in example 1
  scale.title = scale.title, # Created in example 1
  model = open.source.model, # Select a model available on Groq's API
  groq.API = groq.key
)

# View the final item pool
View(personality.inventory.results.gptoss)

################################################################
###### Or, Run AIGENIE with a Hugging Face Embedding Model #####
################################################################

# Chose a BAAI/bge series OR thenlper/gte series model
hf.embedding.model <- "BAAI/bge-large-en-v1.5"

# Create a HF Token to access the best models. Moderate useage will still be FREE
hf.token <- "INSERT YOUR KEY HERE"


# Use AIGENIE with an open source model via Groq
personality.inventory.results.hf <- AIGENIE(
  item.attributes = aspects.of.personality.traits, # created in example 1
  # OpenAI API key is not needed for this example #
  domain = domain, # Created in example 1
  scale.title = scale.title, # Created in example 1
  model = open.source.model, # Select a model available on Groq's API
  groq.API = groq.key,
  embedding.model = hf.embedding.model,
  hf.token = hf.token
)

# View the final item pool
View(personality.inventory.results.hf)

################################################################
#### Example 4: Using Anthropic Claude for Item Generation ####
################################################################

# Add your Anthropic API key
anthropic.key <- "INSERT YOUR KEY HERE"

# Use Claude Sonnet (or "opus", "haiku", or full model names)
personality.inventory.claude <- AIGENIE(
  item.attributes = aspects.of.personality.traits,
  anthropic.API = anthropic.key,
  openai.API = key,  # Still needed for embeddings
  model = "sonnet",  # Alias for claude-sonnet-4-5-20250929
  domain = domain,
  scale.title = scale.title,
  item.type.definitions = trait.definitions
)

# View the final item pool
View(personality.inventory.claude)

################################################################
#### Example 5: Using Jina AI Embeddings ####
################################################################

# Add your Jina API key (free tier available)
jina.key <- "INSERT YOUR KEY HERE"

# Use Jina embeddings with Groq for generation
personality.inventory.jina <- AIGENIE(
  item.attributes = aspects.of.personality.traits,
  groq.API = groq.key,
  jina.API = jina.key,
  model = "llama-3.3-70b-versatile",
  embedding.model = "jina-embeddings-v3",
  domain = domain,
  scale.title = scale.title,
  item.type.definitions = trait.definitions
)

# View the final item pool
View(personality.inventory.jina)

################################################################
#### Example 6: Anthropic + Jina (No OpenAI Required) ####
################################################################

# Full pipeline without OpenAI
personality.inventory.no.openai <- AIGENIE(
  item.attributes = aspects.of.personality.traits,
  anthropic.API = anthropic.key,
  jina.API = jina.key,
  model = "sonnet",
  embedding.model = "jina-embeddings-v3",
  domain = domain,
  scale.title = scale.title,
  item.type.definitions = trait.definitions
)

# View the final item pool
View(personality.inventory.no.openai)

} # }
```
