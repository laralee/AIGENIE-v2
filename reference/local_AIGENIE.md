# Generate and Validate Psychometric Scale Items Using Local Models

Local version of AI-GENIE that uses locally installed language models
and embeddings for complete privacy and offline operation. Generates
items, creates embeddings, and performs network psychometric reduction
entirely on the user's machine.

## Usage

``` r
local_AIGENIE(
  item.attributes,
  model.path,
  embedding.model = "bert-base-uncased",
  main.prompts = NULL,
  temperature = 1,
  top.p = 1,
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
  n.ctx = 4096,
  n.gpu.layers = -1,
  max.tokens = 1024,
  device = "auto",
  batch.size = 32,
  pooling.strategy = "mean",
  max.length = 512L,
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

  Named list of item types and their attributes (required)

- model.path:

  Path to local GGUF model file (required)

- embedding.model:

  Name or path to local embedding model (default: "bert-base-uncased")

- main.prompts:

  Custom prompts for item generation (optional)

- temperature:

  LLM temperature for randomness (0-2, default: 1)

- top.p:

  Top-p nucleus sampling parameter (0-1, default: 1)

- target.N:

  Number of items to generate per type (default: 60)

- domain:

  Content domain (e.g., "psychological")

- scale.title:

  Name of the scale

- item.examples:

  Data frame of example items

- audience:

  Target population

- item.type.definitions:

  Definitions for item types

- response.options:

  Response scale labels

- prompt.notes:

  Additional instructions for generation

- system.role:

  Custom system prompt

- EGA.model:

  Network model ("glasso", "TMFG", or NULL for auto)

- EGA.algorithm:

  Community detection algorithm (default: "walktrap" when there is one
  trait and "louvain" when there are multiple)

- EGA.uni.method:

  Unidimensionality method (default: "louvain")

- uva.cut.off:

  Numeric in `[0, 1)`. wTO threshold passed to
  [`EGAnet::UVA`](https://rdrr.io/pkg/EGAnet/man/UVA.html) for the
  redundancy-reduction step (default: 0.20). Lower values remove more
  items.

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

- n.ctx:

  Context window size (default: 4096)

- n.gpu.layers:

  GPU layers to use (-1 for all, default: -1)

- max.tokens:

  Maximum tokens per generation (default: 1024)

- device:

  Device for embeddings ("auto", "cpu", "cuda", "mps")

- batch.size:

  Batch size for embeddings (default: 32)

- pooling.strategy:

  Pooling for embeddings ("mean", "cls", "max")

- max.length:

  Max sequence length for embeddings (default: 512)

- keep.org:

  Keep original items and embeddings (default: FALSE)

- items.only:

  Generate items only, skip reduction (default: FALSE)

- embeddings.only:

  Generate embeddings only (default: FALSE)

- adaptive:

  Use adaptive generation (default: TRUE)

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

  Display network plots (default: TRUE)

- silently:

  Suppress progress messages (default: FALSE)

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
#### Running AIGENIE with a downloaded LLM model ######
########################################################

# Item type definitions
trait.definitions <- list(
 neuroticism = paste0(
   "Neuroticism is a personality trait that describes one's ",
   "tendency to experience negative emotions like anxiety, ",
   "depression, irritability, anger, and self-consciousness."
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
 extraversion = c("friendly", "positive", "assertive", "energetic")
)

# Name the field or specialty
domain <- "Personality Measurement"

# Name the Inventory being created
scale.title <- "Two of 'Big Five:' A Streamlined Personality Inventory"

# Add a file path name to a local text generation model downloaded on your computer
model.path <- "ADD FILE PATH TO DOWNLOADED MODEL HERE"


# Generate and validate items using a model installed on your machine
local_example <- local_AIGENIE(
 item.attributes = aspects.of.personality.traits,
 item.type.definitions = trait.definitions,
 domain = domain,
 model.path = model.path
)

} # }
```
