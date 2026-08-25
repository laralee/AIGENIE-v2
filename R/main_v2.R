#' Generate, Validate, and Check Items using AI-GENIE
#'
#' @description
#' Generate, validate, and check your items for quality and redundancy using AI-GENIE
#' (Generative Psychometrics via AI-GENIE: Automatic Item Generation and Validation via
#' Network-Integrated Evaluation). AI-GENIE is a methodology that combines the latest
#' open-source LLMs and generative artificial intelligence with advances in network
#' psychometrics to facilitate scale generation, selection, and validation. The pipeline
#' eliminates the need to generate hundreds of items by content experts, recruit diverse
#' and experienced researchers, administer items to thousands of participants, and employ
#' modern psychometric methods in the collected data.
#'
#' @param item.attributes A named list of atomic character vectors (required). Describes
#'   the attributes or characteristics that each item type should encompass. These are not
#'   necessarily lower-order dimensions, but can be if the item types represent appropriate
#'   hierarchical constructs. Each nested list must have at least two unique attributes.
#'   Repeated attributes within the same nested list are not allowed, but attributes can
#'   be repeated across nested lists. Each name of the list must be unique, and all
#'   elements within sublists must be strings. For example, attributes of the personality
#'   trait "neuroticism" might include "anxious", "depressed", "insecure", or "emotional"
#'   since these characteristics encompass aspects of neuroticism that the item pool should address.
#'
#' @param openai.API A character string or NULL (optional, default: NULL). The OpenAI API
#'   key for authentication with OpenAI's services. Required when using OpenAI's platform
#'   for either item generation or embedding. If NULL, users must provide either `groq.API`
#'   for item generation via Groq or `hf.token` for embeddings via Hugging Face.
#'
#' @param hf.token A character string or NULL (optional, default: NULL). The Hugging Face
#'   API token for authentication with Hugging Face services. Required when using Hugging
#'   Face models for embeddings. If NULL, an `openai.API` key must be provided since the
#'   user will need to embed via OpenAI.
#'
#' @param main.prompts A named list of character strings or NULL (optional, default: NULL).
#'   Custom prompts for item generation. If provided, this must be a named list where
#'   `names(main.prompts)` equals `names(item.attributes)`. Each prompt must explicitly
#'   mention all attributes found in the associated element of `item.attributes`. Users
#'   should not include instructions regarding layout/formatting of LLM response, as this
#'   is handled automatically for proper parsing. If NULL, AIGENIE builds appropriate
#'   prompts automatically based on other prompt-building parameters.
#'
#' @param groq.API A character string or NULL (optional, default: NULL). The Groq API
#'   key for authentication with Groq's LLM services. Required when users want to generate
#'   items via Groq's API platform using open-source models. Commonly used in combination
#'   with `openai.API` since Groq does not provide embedding services.
#'
#' @param anthropic.API A character string or NULL (optional, default: NULL). The Anthropic
#'   API key for authentication with Anthropic's Claude models. Required when using Claude
#'   models (e.g., "sonnet", "opus", "haiku") for item generation. Get a key at
#'   \url{https://console.anthropic.com/}.
#'
#' @param jina.API A character string or NULL (optional, default: NULL). The Jina AI API
#'   key for authentication with Jina's embedding services. Required when using Jina
#'   embedding models (e.g., "jina-embeddings-v3", "jina-embeddings-v4"). Free tier
#'   available at \url{https://jina.ai/}.
#'
#' @param model A character string (optional, default: "gpt4o"). Specifies which large
#'   language model to use for item generation. Supports models from multiple providers:
#'   \itemize{
#'     \item \strong{OpenAI}: \code{"gpt-4o"}, \code{"gpt-4"}, \code{"gpt-3.5-turbo"}, \code{"o1"}, \code{"o1-mini"}
#'     \item \strong{Anthropic}: \code{"sonnet"}, \code{"opus"}, \code{"haiku"}, or full names like \code{"claude-sonnet-4-5-20250929"}
#'     \item \strong{Groq}: \code{"llama-3.3-70b-versatile"}, \code{"mixtral-8x7b-32768"}, \code{"gemma2-9b-it"}, \code{"deepseek-r1-distill-llama-70b"}, \code{"qwen-2.5-72b"}
#'   }
#'   Aliases like \code{"llama"}, \code{"mixtral"}, \code{"gemma"}, \code{"deepseek"}, \code{"claude"} are also accepted.
#'   The function automatically determines which API service to use based on the model name
#'   and available API keys.
#'
#' @param temperature A numeric value (optional, default: 1). Controls the randomness and
#'   creativity of the LLM's item generation. Must be between 0-2, where lower values
#'   produce more deterministic outputs and higher values increase creativity and variability.
#'
#' @param top.p A numeric value (optional, default: 1). Controls nucleus sampling for the
#'   LLM's text generation. Must be between 0-1, where lower values make the model more
#'   focused and higher values allow more diverse outputs. Can be used in conjunction
#'   with `temperature`.
#'
#' @param embedding.model A character string (optional, default: "text-embedding-3-small").
#'   Specifies which model to use for generating embeddings of items. Supports multiple providers:
#'   \itemize{
#'     \item \strong{OpenAI}: \code{"text-embedding-3-small"}, \code{"text-embedding-3-large"}, \code{"text-embedding-ada-002"}
#'     \item \strong{Jina AI}: \code{"jina-embeddings-v3"}, \code{"jina-embeddings-v4"}, \code{"jina-embeddings-v2-base-en"} (requires \code{jina.API})
#'     \item \strong{HuggingFace}: \code{"BAAI/bge-small-en-v1.5"}, \code{"BAAI/bge-base-en-v1.5"}, \code{"thenlper/gte-base"}, \code{"sentence-transformers/all-MiniLM-L6-v2"}
#'   }
#'   The provider is automatically detected based on the model name. Jina models support
#'   task adapters and Matryoshka dimension truncation for optimized embeddings.
#'
#' @param target.N An integer, named list of integers, or NULL (optional, default: NULL).
#'   Specifies the number of items to generate for each item type. Can be a single integer
#'   (applies to all item types) or a named list of integers where `names(target.N)` equals
#'   `names(item.attributes)` for different numbers per item type. If NULL, 60 items per
#'   item type will be generated. A rule of thumb is about 60 items or more per item type
#'   for meaningful reduction analysis.
#'
#' @param domain A character string or NULL (optional, default: NULL). Specifies the
#'   psychological or research domain for context in item generation. Should be specific
#'   (e.g., "personality", "child development") rather than general. If supplied, it will
#'   be used to construct appropriate prompts and system roles unless `system.role` is provided.
#'
#' @param scale.title A character string or NULL (optional, default: NULL). Specifies
#'   the name or title of the scale being developed. Can be formal or descriptive, but
#'   more specific titles generally produce better results. If supplied, it will be used
#'   to construct appropriate prompts and system roles unless `system.role` is provided.
#'
#' @param item.examples A data frame or NULL (optional, default: NULL). Provides example
#'   items to guide the LLM's generation style and format. Must be a data frame with
#'   columns: `statement` (the actual item), `attribute` (the item's attribute), and
#'   `type` (the item's type). All values must be non-empty strings, and the `attribute`
#'   and `type` must align with the `item.attributes` object. Items should be extremely
#'   high quality and validated if possible, as they serve as style templates.
#'
#' @param audience A character string or NULL (optional, default: NULL). Specifies the
#'   target population for the scale being developed. Should be as specific as possible
#'   (e.g., "educated adults in rural America", "children with ASD in second grade")
#'   rather than general demographic categories. If supplied, it will be used to construct
#'   appropriate prompts and system roles unless `system.role` is provided.
#'
#' @param item.type.definitions A named list of character strings or NULL (optional,
#'   default: NULL). Provides definitions or descriptions of each item type for the LLM.
#'   Must be a named list where `names(item.type.definitions)` equals `names(item.attributes)`.
#'   Useful when constructs or item types are obscure or potentially ambiguous, helping
#'   the LLM understand the item type or construct in your specific context. If supplied,
#'   it will be used to construct appropriate prompts and system roles unless `system.role`
#'   is provided.
#'
#' @param response.options A character vector or NULL (optional, default: NULL). Specifies
#'   the response scale labels for the generated items (e.g., c("agree", "neither agree
#'   nor disagree", "disagree")). These labels provide context for item writing but do
#'   not appear in the actual items themselves. If supplied, it will be used to construct
#'   appropriate system roles unless `system.role` is provided.
#'
#' @param prompt.notes A named list of character strings, character string, or NULL
#'   (optional, default: NULL). Allows users to add custom instructions or context to
#'   the prompts. Can be a named list where `names(prompt.notes)` equals `names(item.attributes)`
#'   for different notes per item type, or a single string applied to all item types.
#'   These notes are appended at the end of constructed prompts, allowing users to add
#'   brief additional requirements (e.g., "All items MUST begin with the stem 'I am
#'   someone who...'") without creating entirely custom prompts. Should be brief; otherwise,
#'   users should consider using `main.prompts`.
#'
#' @param system.role A character string or NULL (optional, default: NULL). Defines the
#'   system role/persona for the LLM during item generation. If not provided, one is
#'   built automatically based on prompt-building parameters. Should be as specific as
#'   possible (e.g., "You are an expert scale developer and psychometrician with extensive
#'   expertise in drafting Likert-type items for children with ASD. Today, you will focus
#'   on developing robust, single-statement items that assess linguistic ability.").
#'   Applies to all LLM interactions.
#'
#' @param EGA.model A character string or NULL (optional, default: NULL). Specifies which
#'   model to use for Exploratory Graph Analysis network construction. Valid options are
#'   "tmfg" or "glasso". If NULL, AIGENIE will test both "tmfg" and "glasso" models and
#'   automatically return the model that maximizes NMI (normalized mutual information).
#'   TMFG is a greedy but speedy network-building algorithm that works well for many
#'   applications, especially text. EBICglasso is slower but non-greedy and may capture
#'   more nuanced relationships.
#'
#' @param EGA.algorithm A character string (optional, default is "walktrap" when there is a
#'   single trait and "louvain" when there is more than one trait). Specifies
#'   which community detection algorithm to use within the EGA framework. Valid options
#'   are "louvain", "walktrap", or "leiden". The algorithm operates separately from the
#'   network building specified by `EGA.model`.
#'
#' @param EGA.uni.method A character string (optional, default: "louvain"). Specifies
#'   the method for handling unidimensional structures in EGA. Valid options are: "expand"
#'   (expands correlation matrix with four variables correlated 0.50; if dimensions ≤ 2,
#'   data are unidimensional), "LE" (applies Leading Eigenvector algorithm; if dimensions = 1,
#'   uses LE solution), or "louvain" (applies Louvain algorithm; if dimensions = 1, uses
#'   Louvain solution). This parameter is rarely modified by users.
#'
#' @param uva.cut.off A numeric value in `[0, 1)` (optional, default: `0.20`). The weighted
#'   topological overlap threshold passed to `EGAnet::UVA` during the redundancy-reduction
#'   step. Items with pairwise wTO at or above this value are flagged as redundant. Lower
#'   values are more aggressive (remove more items); higher values are more conservative.
#'
#' @param keep.org A logical value (optional, default: FALSE). Controls whether the
#'   pre-reduced items generated by the model are returned to the user. If TRUE, returns
#'   the full item pool before psychometric reduction. Does not affect the reduction process.
#'
#' @param items.only A logical value (optional, default: FALSE). Controls whether the
#'   function only generates items without running the full psychometric pipeline. If TRUE,
#'   skips embedding, EGA, and reduction steps, returning only a data frame with columns
#'   `ID`, `statement`, `type`, and `attribute`. Useful when users want to generate items
#'   with AIGENIE, embed them elsewhere, and use the `GENIE` function for reduction.
#'
#' @param embeddings.only A logical value (optional, default: FALSE). Controls whether
#'   the function generates items and embeddings but skips psychometric reduction. If TRUE,
#'   returns a named list with `embeddings` (the embedding matrix) and `items` (the items
#'   data frame). If both `items.only` and `embeddings.only` are TRUE, defaults to
#'   `embeddings.only` behavior.
#'
#' @param adaptive A logical value (optional, default: TRUE). Controls whether previously
#'   generated items are incorporated into subsequent prompts to reduce redundancy. Items
#'   are generated in batches to avoid context window limitations, potentially requiring
#'   multiple API calls. When TRUE, appends previously generated items so the model knows
#'   what has been generated to avoid repetition. Should always be enabled unless context
#'   limitations are a concern.
#'
#' @param run.overall A logical value (optional, default: FALSE). Controls whether a *fit* analysis
#'    on the complete item pool is run *post-reduction.*
#'    By default, only type-level reduction analyses are run (i.e., items of like-type go through
#'    the pipeline independent of the other items in the pool). When this flag is `TRUE`, an additional
#'    analysis is run on the overall sample, but no further reductions at the overall level are made.
#'    If only one item type is present, this argument will be ignored.
#'
#' @param all.together A logical value (optional, default: FALSE). Controls whether the *reduction* analysis
#'    on the complete item pool is run.
#'    By default, only type-level reduction analyses are run (i.e., items of like-type go through
#'    the pipeline independent of the other items in the pool). When this flag is `TRUE`, reductions are made
#'    at the overall level (i.e., all items go through the reduction pipeline together, agnostic of item type).
#'    If only one item type is present, this argument will be ignored.
#'
#' @param plot A logical value (optional, default: TRUE). Controls whether visualizations
#'   are generated and displayed. When TRUE, generates EGA network comparison plots (before
#'   vs after reduction) for each item type and the sample overall. Plots are always saved
#'   and returned in the output object but can be suppressed from display for cleaner output.
#'
#' @param silently A logical value (optional, default: FALSE). Controls console output
#'   and messaging during function execution. When TRUE, suppresses progress statements
#'   about item generation, embedding, and pipeline reduction. Does not affect warnings
#'   or errors, only informational messages. Operates independently of the `plot` parameter.
#'
#' @return
#' The structure of the return value depends on the function flags.
#'
#' **Defaults:** `items.only = FALSE`, `embeddings.only = FALSE`,
#' `run.overall = FALSE`, `keep.org = FALSE`, `all.together = FALSE`.
#'
#' **When `items.only = TRUE`:**
#' Returns a `data.frame` of generated items with columns:
#' `ID`, `statement`, `type`, and `attribute`.
#'
#' **When `embeddings.only = TRUE`:**
#' Returns a named `list` with two elements:
#' \itemize{
#'   \item `embeddings` — an embedding matrix/list (columns or rownames correspond to item IDs).
#'   \item `items` — the items `data.frame` described above.
#' }
#'
#' **Default behaviour** (`items.only = FALSE`, `embeddings.only = FALSE`,
#' `run.overall = FALSE`, `keep.org = FALSE`, `all.together = FALSE`):
#' Returns a named `list` with two top-level elements:
#' \describe{
#'   \item{`item_type_level`}{A named list where each name is an item type and each element is a per-type named list containing:
#'     \describe{
#'       \item{`final_NMI`}{Numeric: final normalized mutual information after reduction.}
#'       \item{`initial_NMI`}{Numeric: initial NMI of the pre-reduced item pool.}
#'       \item{`embeddings`}{List or matrix of embeddings for this item type (see 'Notes on `embeddings`' below).}
#'       \item{`UVA`}{List from Unique Variable Analysis (contains at least `n_removed`, `n_sweeps`, `redundant_pairs` data.frame).}
#'       \item{`bootEGA`}{List with bootEGA results (e.g. `initial_boot`, `final_boot`, `n_removed`, `items_removed`, `initial_boot_with_redundancies`).}
#'       \item{`EGA.model_selected`}{Character: chosen EGA model (e.g. `"TMFG"` or `"Glasso"`).}
#'       \item{`final_items`}{`data.frame`: final items after reduction (columns include `ID`, `statement`, `attribute`, `type`, `EGA_com`).}
#'       \item{`final_EGA`}{EGA object (from EGAnet) after reduction.}
#'       \item{`initial_EGA`}{Initial EGA object computed on the pre-reduced item set.}
#'       \item{`start_N`}{Integer: initial number of items in this type.}
#'       \item{`final_N`}{Integer: final number of items in this type.}
#'       \item{`network_plot`}{`ggplot` / `patchwork` object comparing networks before vs after reduction.}
#'       \item{`stability_plot`}{`ggplot` / `patchwork` object showing item stability before vs after reduction.}
#'     }
#'   }
#'
#'   \item{`overall`}{Named list with aggregated results across all item types. Under the default this contains:
#'     \describe{
#'       \item{`final_items`}{`data.frame` of final items across all types (columns as above).}
#'       \item{`embeddings`}{Embeddings for the full reduced item set (see 'Notes on `embeddings`' below). Note: `overall$embeddings` does **not** include `selected`.}
#'     }
#'   }
#' }
#'
#' **When `keep.org = TRUE`** (in addition to defaults above):
#' The top-level shape remains (`item_type_level` and `overall`) but includes original (pre-reduction) information:
#' \describe{
#'   \item{`item_type_level`}{Each per-type sublist contains:
#'     `final_NMI`, `initial_NMI`, `embeddings`, `UVA`, `bootEGA`, `EGA.model_selected`, `final_items`, `initial_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`, `stability_plot`.}
#'   \item{`overall`}{Contains `final_items`, `initial_items`, and `embeddings` for the full item pool.}
#' }
#' For `keep.org = TRUE`, per-type `embeddings` contains at least: `full_org`, `sparse_org`, `selected`, `full`, and `sparse`. (`overall$embeddings` contains the same subcomponents **except** `selected` is omitted.)
#'
#' **When `run.overall = TRUE`** (`items.only = FALSE`, `embeddings.only = FALSE`):
#' \describe{
#'   \item{`item_type_level`}{Same per-type structure as the default (see above).}
#'   \item{`overall`}{A named list with aggregated results (not limited to `final_items` and `embeddings`) containing:
#'     `final_NMI`, `initial_NMI`, `embeddings`, `EGA.model_selected`, `final_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, and `network_plot`.}
#' }
#'
#' **When `all.together = TRUE`** (regardless of `run.overall`):
#' Results are **not** split into `item_type_level` and `overall`. Instead the function returns a single named list (applies to the full — possibly `keep.org` modified — result set) containing:
#' `final_NMI`, `initial_NMI`, `embeddings`, `UVA`, `bootEGA`, `EGA.model_selected`, `final_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`, and `stability_plot`.
#'
#'
#' @references
#' Golino, H. F., & Epskamp, S. (2017). Exploratory graph analysis: A new approach
#' for estimating the number of dimensions in psychological research.
#' \emph{PLOS ONE, 12}(6), e0174035.
#' \url{https://doi.org/10.1371/journal.pone.0174035}
#'
#' Christensen, A. P., Garrido, L. E., & Golino, H. (2023). Unique variable
#' analysis: A network psychometrics method to detect local dependence.
#' \emph{Multivariate Behavioral Research, 58}(6), 1165–1182.
#' \url{https://doi.org/10.1080/00273171.2023.2194606}
#'
#' Christensen, A. P., & Golino, H. (2021). Estimating the stability of
#' psychological dimensions via bootstrap exploratory graph analysis:
#' A Monte Carlo simulation and tutorial.
#' \emph{Psych, 3}(3), 479–500.
#' \url{https://doi.org/10.3390/psych3030032}
#'
#' Danon, L., Díaz-Guilera, A., Duch, J., & Arenas, A. (2005). Comparing
#' community structure identification.
#' \emph{Journal of Statistical Mechanics: Theory and Experiment, 2005}(9),
#' P09008.
#' \url{https://doi.org/10.1088/1742-5468/2005/09/P09008}
#'
#' Russell-Lasalandra, L. L., Christensen, A. P., & Golino, H. (2024).
#' Generative psychometrics via AI-GENIE: Automatic item generation and
#' validation via network-integrated evaluation.
#' \url{https://osf.io/preprints/psyarxiv/fgbj4_v2}.
#'
#' @examples
#' \dontrun{
#' ########################################################
#' #### Example 1: Using AI-GENIE with Default Prompts ####
#' ########################################################
#'
#' # Add an OpenAI API key
#' key <- "INSERT YOUR KEY HERE"
#'
#' # Item type definitions
#' trait.definitions <- list(
#'   neuroticism = paste0(
#'     "Neuroticism is a personality trait that describes one's ",
#'     "tendency to experience negative emotions like anxiety, ",
#'     "depression, irritability, anger, and self-consciousness."
#'   ),
#'   openness = paste0(
#'     "Openness is a personality trait that describes how ",
#'     "open-minded, creative, and imaginative a person is."
#'   ),
#'   extraversion = paste0(
#'     "Extraversion is a personality trait that describes people ",
#'     "who are more focused on the external world than their ",
#'     "internal experience."
#'   )
#' )
#'
#' # Item attributes
#' aspects.of.personality.traits <- list(
#'   neuroticism = c("anxious", "depressed", "insecure", "emotional"),
#'   openness = c("creative", "perceptual", "curious", "philosophical"),
#'   extraversion = c("friendly", "positive", "assertive", "energetic")
#' )
#'
#' # Name the field or specialty
#' domain <- "Personality Measurement"
#'
#' # Name the Inventory being created
#' scale.title <- "Three of 'Big Five:' A Streamlined Personality Inventory"
#'
#' # Run AI-GENIE to generate, validate, and redundancy-check an item pool for your new scale.
#' personality.inventory.results <- AIGENIE(
#'   item.attributes = aspects.of.personality.traits,
#'   openai.API = key,
#'   domain = domain,
#'   scale.title = scale.title,
#'   item.type.definitions = trait.definitions
#' )
#'
#' # View the final item pool
#' View(personality.inventory.results)
#'
#'
#' #######################################################
#' #### Example 2: Using AI-GENIE with Custom Prompts ####
#' #######################################################
#'
#'
#' # Define a custom system role
#' system.role <- paste0(
#'   "You are an expert methodologist who specializes in scale ",
#'   "development for personality measurement. You are especially ",
#'   "equipped to create novel personality items that mimic the ",
#'   "style of popular 'Big Five' assessments."
#' )
#'
#' # Define custom prompts for each personality trait
#' custom.personality.prompts <- list(
#'
#'   # Prompt for generating neuroticism traits
#'   neuroticism = paste0(
#'     "Generate unique, psychometrically robust single-statement items designed to assess ",
#'     "the Big Five personality trait neuroticism.",
#'     paste0(
#'       "Neuroticism has the following characteristics: anxious, ",
#'       "depressed, insecure, and emotional. "
#'     )
#'   ),
#'
#'   # Prompt for generating openness traits
#'   openness = paste0(
#'     "Generate unique, psychometrically robust single-statement items designed to assess ",
#'     "the Big Five personality trait openness.",
#'     paste0(
#'       "Openness has the following characteristics: creative, ",
#'       "perceptual, curious, and philosophical"
#'     )
#'   ),
#'
#'   # Prompt for generating extraversion traits
#'   extraversion = paste0(
#'     "Generate unique, psychometrically robust single-statement items designed to assess ",
#'     "the Big Five personality trait extraversion.",
#'     paste0(
#'       "Extraversion has the following characteristics: friendly, ",
#'       "positive, assertive, and energetic."
#'     )
#'   )
#'
#' )
#'
#' # Run AI-GENIE to generate, validate, and redundancy-check an item pool for your new scale.
#' personality.inventory.results.custom <- AIGENIE(
#'   item.attributes = aspects.of.personality.traits, # created in example 1
#'   main.prompts = custom.personality.prompts,
#'   system.role = system.role,
#'   openai.API = key, # created in example 1
#'   scale.title = scale.title # created in example 1
#' )
#'
#' # View the final item pool
#' View(personality.inventory.results.custom)
#'
#' ################################################################
#' ###### Or, Run AIGENIE with an Open Source Model via Groq ######
#' ################################################################
#'
#' # Add your API Key from Groq
#' groq.key <- "INSERT YOUR KEY HERE"
#'
#' # Chose an open-source model like 'DeepSeek' or 'GPT oss'
#' open.source.model <- "GPT oss 120b"
#'
#' # Use AIGENIE with an open source model via Groq
#' personality.inventory.results.gptoss <- AIGENIE(
#'   item.attributes = aspects.of.personality.traits, # created in example 1
#'   openai.API = key, # Created in example 1
#'   domain = domain, # Created in example 1
#'   scale.title = scale.title, # Created in example 1
#'   model = open.source.model, # Select a model available on Groq's API
#'   groq.API = groq.key
#' )
#'
#' # View the final item pool
#' View(personality.inventory.results.gptoss)
#'
#' ################################################################
#' ###### Or, Run AIGENIE with a Hugging Face Embedding Model #####
#' ################################################################
#'
#' # Chose a BAAI/bge series OR thenlper/gte series model
#' hf.embedding.model <- "BAAI/bge-large-en-v1.5"
#'
#' # Create a HF Token to access the best models. Moderate useage will still be FREE
#' hf.token <- "INSERT YOUR KEY HERE"
#'
#'
#' # Use AIGENIE with an open source model via Groq
#' personality.inventory.results.hf <- AIGENIE(
#'   item.attributes = aspects.of.personality.traits, # created in example 1
#'   # OpenAI API key is not needed for this example #
#'   domain = domain, # Created in example 1
#'   scale.title = scale.title, # Created in example 1
#'   model = open.source.model, # Select a model available on Groq's API
#'   groq.API = groq.key,
#'   embedding.model = hf.embedding.model,
#'   hf.token = hf.token
#' )
#'
#' # View the final item pool
#' View(personality.inventory.results.hf)
#'
#' ################################################################
#' #### Example 4: Using Anthropic Claude for Item Generation ####
#' ################################################################
#'
#' # Add your Anthropic API key
#' anthropic.key <- "INSERT YOUR KEY HERE"
#'
#' # Use Claude Sonnet (or "opus", "haiku", or full model names)
#' personality.inventory.claude <- AIGENIE(
#'   item.attributes = aspects.of.personality.traits,
#'   anthropic.API = anthropic.key,
#'   openai.API = key,  # Still needed for embeddings
#'   model = "sonnet",  # Alias for claude-sonnet-4-5-20250929
#'   domain = domain,
#'   scale.title = scale.title,
#'   item.type.definitions = trait.definitions
#' )
#'
#' # View the final item pool
#' View(personality.inventory.claude)
#'
#' ################################################################
#' #### Example 5: Using Jina AI Embeddings ####
#' ################################################################
#'
#' # Add your Jina API key (free tier available)
#' jina.key <- "INSERT YOUR KEY HERE"
#'
#' # Use Jina embeddings with Groq for generation
#' personality.inventory.jina <- AIGENIE(
#'   item.attributes = aspects.of.personality.traits,
#'   groq.API = groq.key,
#'   jina.API = jina.key,
#'   model = "llama-3.3-70b-versatile",
#'   embedding.model = "jina-embeddings-v3",
#'   domain = domain,
#'   scale.title = scale.title,
#'   item.type.definitions = trait.definitions
#' )
#'
#' # View the final item pool
#' View(personality.inventory.jina)
#'
#' ################################################################
#' #### Example 6: Anthropic + Jina (No OpenAI Required) ####
#' ################################################################
#'
#' # Full pipeline without OpenAI
#' personality.inventory.no.openai <- AIGENIE(
#'   item.attributes = aspects.of.personality.traits,
#'   anthropic.API = anthropic.key,
#'   jina.API = jina.key,
#'   model = "sonnet",
#'   embedding.model = "jina-embeddings-v3",
#'   domain = domain,
#'   scale.title = scale.title,
#'   item.type.definitions = trait.definitions
#' )
#'
#' # View the final item pool
#' View(personality.inventory.no.openai)
#'
#' }
#'
#' @export
AIGENIE <- function(item.attributes, openai.API=NULL, hf.token=NULL, # required parameters

                       # optional parameters --

                       # if using AIGENIE in custom mode, this should be set:
                       main.prompts = NULL,

                       # LLM parameters
                       groq.API = NULL, anthropic.API = NULL, jina.API = NULL,
                       model = "gpt4o", temperature = 1,
                       top.p = 1, embedding.model = "text-embedding-3-small",
                       target.N = NULL,

                       # Prompt parameters
                       domain = NULL, scale.title = NULL, item.examples = NULL,
                       audience = NULL, item.type.definitions = NULL,
                       response.options = NULL, prompt.notes = NULL, system.role = NULL,

                       # EGA parameters
                       EGA.model = NULL, EGA.algorithm = NULL, EGA.uni.method = NULL,
                       uva.cut.off = 0.20,

                       # Flags
                       keep.org = FALSE, items.only = FALSE, embeddings.only = FALSE,
                       adaptive = TRUE, run.overall = FALSE, all.together = FALSE,
                       plot = TRUE, silently = FALSE
                       ){


  # Validate uva.cut.off (kept separate from validate_user_input_AIGENIE to
  # avoid expanding the positional signature of the validator).
  uva.cut.off_validate(uva.cut.off)

  # Validate all params and reassign params
  validation <- validate_user_input_AIGENIE(item.attributes, openai.API, hf.token,
                                            main.prompts,
                                            groq.API, anthropic.API, jina.API,
                                            model, temperature,
                                            top.p, embedding.model, target.N,
                                            domain, scale.title, item.examples,
                                            audience, item.type.definitions,
                                            response.options, prompt.notes,
                                            system.role, EGA.model, EGA.algorithm,
                                            EGA.uni.method, keep.org, items.only,
                                            embeddings.only, adaptive, run.overall,
                                            all.together, plot, silently)


  target.N <- validation$target.N
  EGA.model <- validation$EGA.model
  EGA.uni.method <- validation$EGA.uni.method
  EGA.algorithm <- validation$EGA.algorithm
  model <- validation$model
  item.type.definitions <- validation$item.type.definitions
  item.examples <- validation$item.examples
  item.attributes <- validation$item.attributes
  prompt.notes <- validation$prompt.notes
  main.prompts <- validation$main.prompts
  custom <- validation$custom
  run.overall <- validation$run.overall
  all.together <- validation$all.together


  # Begin constructing the prompts
  # first, the system role if one was not provided
  system.role <- create_system.role(domain, scale.title, audience,
                                    response.options, system.role)


  # Create/Modify the prompts
  if(!custom){
    main.prompts <- create_main.prompts(item.attributes, item.type.definitions,
                                      domain, scale.title, prompt.notes,
                                      audience, item.examples)
  } else {
    main.prompts <- modify_main.prompts(main.prompts, item.attributes,
                                        item.type.definitions,
                                        domain, scale.title, prompt.notes,
                                        audience, item.examples)

  }


  # Generate the items for reduction analysis
  items_gen <- generate_items_via_llm(main.prompts, system.role, model, top.p, temperature,
                                  adaptive, silently, groq.API, openai.API,
                                  anthropic.API = anthropic.API, target.N = target.N)
  items <- items_gen$items
  success <- items_gen$successful

  if(is.data.frame(items)){
    items$ID <- 1:nrow(items) # create an ID variable
  }

  # return items if requested OR if the run was not a success
  if(items.only || !success){

    if(!success && !silently){
      message("Item generation failed before completion. Returning a data frame of items generated thus far.")
    }

    return(items)
  }


  # Now, generate item embeddings
  attempt_to_embed <- generate_embeddings(
    embedding.model = embedding.model,
    items = items,
    openai.API = openai.API,
    hf.token = hf.token,
    jina.API = jina.API,
    silently = silently
  )

  success <- attempt_to_embed$success
  embeddings <- attempt_to_embed$embeddings

  # Return partial results if failure or just the embeddings if requested
  if(!success || embeddings.only){
    if(!success && !silently){
      message("Embedding step has failed. Returning a data frame of items generated instead.")
    }

    if(!success){
      return(items)
    }

     return(list(embeddings = embeddings, items = items))
  }

  # Run as a single sample if applicable
  if(all.together){
    items_to_run <- run_all_together(items)

    # run the pipeline using the updated matrix
    try_item_level <- run_item_reduction_pipeline(embedding_matrix = embeddings,
                                                  items=items_to_run, EGA.model = EGA.model$overall,
                                                  EGA.algorithm = EGA.algorithm$overall,
                                                  EGA.uni.method = EGA.uni.method$overall,
                                                  uva.cut.off = uva.cut.off, keep.org = keep.org,
                                                  silently = silently, plot = plot)

    if(!try_item_level$success){

      if(!silently){
        message("AI-GENIE reduction failed. Returning partial results.")
      }

      return(try_item_level$item_level)
    }

    item_level <- try_item_level$item_level

    # Update the returned data frame appropriately
    IDs <- item_level[["All"]][["final_items"]]$ID
    item_level[["All"]][["final_items"]] <- items[items$ID %in% IDs,]
    if(keep.org){
      item_level[["All"]][["initial_items"]] <- items
    }

    return(item_level[["All"]])

  }



  # Generate item level results
  try_item_level <- run_item_reduction_pipeline(embedding_matrix = embeddings,
                    items=items, EGA.model = EGA.model$type,
                    EGA.algorithm = EGA.algorithm$type,
                    EGA.uni.method = EGA.uni.method$type,
                    uva.cut.off = uva.cut.off, keep.org = keep.org,
                    silently = silently, plot = plot)

  if(!try_item_level$success){
    if(!silently){
      message("AI-GENIE reduction failed. Returning partial results.")
    }
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level

  if(run.overall && length(names(item.attributes)) > 1) { # only run overall if you have to


    # If successful, generate results for items overall
    try_overall_result <- run_pipeline_for_all(item_level = item_level, items = items,
                            embeddings = embeddings, model = EGA.model$overall,
                            algorithm = EGA.algorithm$overall,
                            uni.method = EGA.uni.method$overall,
                            uva.cut.off = uva.cut.off,
                            keep.org = keep.org, silently = silently, plot = plot)

    if(!try_overall_result$success && !silently){
      message("Overall analyses has failed. Returning only type-level results.")
      return(item_level)
    }

    overall_result <- try_overall_result$overall_result
  } else {
    overall_result <- item_level
    try_overall_result <- list(success = TRUE)
  }


  if(!silently && try_overall_result$success && try_item_level$success){
    print_results(overall_result, item_level, run.overall)
  }

  return(build_return(item_level, overall_result,
                      run.overall, keep.org))

}




#' Generate and Validate Psychometric Scale Items Using Local Models
#'
#' @description
#' Local version of AI-GENIE that uses locally installed language models and
#' embeddings for complete privacy and offline operation. Generates items,
#' creates embeddings, and performs network psychometric reduction entirely
#' on the user's machine.
#'
#' @param item.attributes Named list of item types and their attributes (required)
#' @param model.path Path to local GGUF model file (required)
#' @param embedding.model Name or path to local embedding model (default: "bert-base-uncased")
#' @param main.prompts Custom prompts for item generation (optional)
#' @param temperature LLM temperature for randomness (0-2, default: 1)
#' @param top.p Top-p nucleus sampling parameter (0-1, default: 1)
#' @param target.N Number of items to generate per type (default: 60)
#' @param domain Content domain (e.g., "psychological")
#' @param scale.title Name of the scale
#' @param item.examples Data frame of example items
#' @param audience Target population
#' @param item.type.definitions Definitions for item types
#' @param response.options Response scale labels
#' @param prompt.notes Additional instructions for generation
#' @param system.role Custom system prompt
#' @param EGA.model Network model ("glasso", "TMFG", or NULL for auto)
#' @param EGA.algorithm Community detection algorithm (default: "walktrap" when there is one trait and "louvain" when there are multiple)
#' @param EGA.uni.method Unidimensionality method (default: "louvain")
#' @param uva.cut.off Numeric in `[0, 1)`. wTO threshold passed to `EGAnet::UVA` for the
#'   redundancy-reduction step (default: 0.20). Lower values remove more items.
#' @param n.ctx Context window size (default: 4096)
#' @param n.gpu.layers GPU layers to use (-1 for all, default: -1)
#' @param max.tokens Maximum tokens per generation (default: 1024)
#' @param device Device for embeddings ("auto", "cpu", "cuda", "mps")
#' @param batch.size Batch size for embeddings (default: 32)
#' @param pooling.strategy Pooling for embeddings ("mean", "cls", "max")
#' @param max.length Max sequence length for embeddings (default: 512)
#' @param keep.org Keep original items and embeddings (default: FALSE)
#' @param items.only Generate items only, skip reduction (default: FALSE)
#' @param embeddings.only Generate embeddings only (default: FALSE)
#' @param adaptive Use adaptive generation (default: TRUE)
#' @param run.overall A logical value (optional, default: FALSE). Controls whether a *fit* analysis
#'    on the complete item pool is run *post-reduction.*
#'    By default, only type-level reduction analyses are run (i.e., items of like-type go through
#'    the pipeline independent of the other items in the pool). When this flag is `TRUE`, an additional
#'    analysis is run on the overall sample, but no further reductions at the overall level are made.
#'    If only one item type is present, this argument will be ignored.
#'
#' @param all.together A logical value (optional, default: FALSE). Controls whether the *reduction* analysis
#'    on the complete item pool is run.
#'    By default, only type-level reduction analyses are run (i.e., items of like-type go through
#'    the pipeline independent of the other items in the pool). When this flag is `TRUE`, reductions are made
#'    at the overall level (i.e., all items go through the reduction pipeline together, agnostic of item type).
#'    If only one item type is present, this argument will be ignored.
#' @param plot Display network plots (default: TRUE)
#' @param silently Suppress progress messages (default: FALSE)
#'
#' @return
#' The structure of the return value depends on the function flags.
#'
#' **Defaults:** `items.only = FALSE`, `embeddings.only = FALSE`,
#' `run.overall = FALSE`, `keep.org = FALSE`, `all.together = FALSE`.
#'
#' **When `items.only = TRUE`:**
#' Returns a `data.frame` of generated items with columns:
#' `ID`, `statement`, `type`, and `attribute`.
#'
#' **When `embeddings.only = TRUE`:**
#' Returns a named `list` with two elements:
#' \itemize{
#'   \item `embeddings` — an embedding matrix/list (columns or rownames correspond to item IDs).
#'   \item `items` — the items `data.frame` described above.
#' }
#'
#' **Default behaviour** (`items.only = FALSE`, `embeddings.only = FALSE`,
#' `run.overall = FALSE`, `keep.org = FALSE`, `all.together = FALSE`):
#' Returns a named `list` with two top-level elements:
#' \describe{
#'   \item{`item_type_level`}{A named list where each name is an item type and each element is a per-type named list containing:
#'     \describe{
#'       \item{`final_NMI`}{Numeric: final normalized mutual information after reduction.}
#'       \item{`initial_NMI`}{Numeric: initial NMI of the pre-reduced item pool.}
#'       \item{`embeddings`}{List or matrix of embeddings for this item type (see 'Notes on `embeddings`' below).}
#'       \item{`UVA`}{List from Unique Variable Analysis (contains at least `n_removed`, `n_sweeps`, `redundant_pairs` data.frame).}
#'       \item{`bootEGA`}{List with bootEGA results (e.g. `initial_boot`, `final_boot`, `n_removed`, `items_removed`, `initial_boot_with_redundancies`).}
#'       \item{`EGA.model_selected`}{Character: chosen EGA model (e.g. `"TMFG"` or `"Glasso"`).}
#'       \item{`final_items`}{`data.frame`: final items after reduction (columns include `ID`, `statement`, `attribute`, `type`, `EGA_com`).}
#'       \item{`final_EGA`}{EGA object (from EGAnet) after reduction.}
#'       \item{`initial_EGA`}{Initial EGA object computed on the pre-reduced item set.}
#'       \item{`start_N`}{Integer: initial number of items in this type.}
#'       \item{`final_N`}{Integer: final number of items in this type.}
#'       \item{`network_plot`}{`ggplot` / `patchwork` object comparing networks before vs after reduction.}
#'       \item{`stability_plot`}{`ggplot` / `patchwork` object showing item stability before vs after reduction.}
#'     }
#'   }
#'
#'   \item{`overall`}{Named list with aggregated results across all item types. Under the default this contains:
#'     \describe{
#'       \item{`final_items`}{`data.frame` of final items across all types (columns as above).}
#'       \item{`embeddings`}{Embeddings for the full reduced item set (see 'Notes on `embeddings`' below). Note: `overall$embeddings` does **not** include `selected`.}
#'     }
#'   }
#' }
#'
#' **When `keep.org = TRUE`** (in addition to defaults above):
#' The top-level shape remains (`item_type_level` and `overall`) but includes original (pre-reduction) information:
#' \describe{
#'   \item{`item_type_level`}{Each per-type sublist contains:
#'     `final_NMI`, `initial_NMI`, `embeddings`, `UVA`, `bootEGA`, `EGA.model_selected`, `final_items`, `initial_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`, `stability_plot`.}
#'   \item{`overall`}{Contains `final_items`, `initial_items`, and `embeddings` for the full item pool.}
#' }
#' For `keep.org = TRUE`, per-type `embeddings` contains at least: `full_org`, `sparse_org`, `selected`, `full`, and `sparse`. (`overall$embeddings` contains the same subcomponents **except** `selected` is omitted.)
#'
#' **When `run.overall = TRUE`** (`items.only = FALSE`, `embeddings.only = FALSE`):
#' \describe{
#'   \item{`item_type_level`}{Same per-type structure as the default (see above).}
#'   \item{`overall`}{A named list with aggregated results (not limited to `final_items` and `embeddings`) containing:
#'     `final_NMI`, `initial_NMI`, `embeddings`, `EGA.model_selected`, `final_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, and `network_plot`.}
#' }
#'
#' **When `all.together = TRUE`** (regardless of `run.overall`):
#' Results are **not** split into `item_type_level` and `overall`. Instead the function returns a single named list (applies to the full — possibly `keep.org` modified — result set) containing:
#' `final_NMI`, `initial_NMI`, `embeddings`, `UVA`, `bootEGA`, `EGA.model_selected`, `final_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`, and `stability_plot`.
#'
#' @references
#' Golino, H. F., & Epskamp, S. (2017). Exploratory graph analysis: A new approach
#' for estimating the number of dimensions in psychological research.
#' \emph{PLOS ONE, 12}(6), e0174035.
#' \url{https://doi.org/10.1371/journal.pone.0174035}
#'
#' Christensen, A. P., Garrido, L. E., & Golino, H. (2023). Unique variable
#' analysis: A network psychometrics method to detect local dependence.
#' \emph{Multivariate Behavioral Research, 58}(6), 1165–1182.
#' \url{https://doi.org/10.1080/00273171.2023.2194606}
#'
#' Christensen, A. P., & Golino, H. (2021). Estimating the stability of
#' psychological dimensions via bootstrap exploratory graph analysis:
#' A Monte Carlo simulation and tutorial.
#' \emph{Psych, 3}(3), 479–500.
#' \url{https://doi.org/10.3390/psych3030032}
#'
#' Danon, L., Díaz-Guilera, A., Duch, J., & Arenas, A. (2005). Comparing
#' community structure identification.
#' \emph{Journal of Statistical Mechanics: Theory and Experiment, 2005}(9),
#' P09008.
#' \url{https://doi.org/10.1088/1742-5468/2005/09/P09008}
#'
#' Russell-Lasalandra, L. L., Christensen, A. P., & Golino, H. (2024).
#' Generative psychometrics via AI-GENIE: Automatic item generation and
#' validation via network-integrated evaluation.
#' \url{https://osf.io/preprints/psyarxiv/fgbj4_v2}.
#'
#' @examples
#' \dontrun{
#' ########################################################
#' #### Running AIGENIE with a downloaded LLM model ######
#' ########################################################
#'
#' # Item type definitions
#' trait.definitions <- list(
#'  neuroticism = paste0(
#'    "Neuroticism is a personality trait that describes one's ",
#'    "tendency to experience negative emotions like anxiety, ",
#'    "depression, irritability, anger, and self-consciousness."
#'  ),
#'  extraversion = paste0(
#'    "Extraversion is a personality trait that describes people ",
#'    "who are more focused on the external world than their ",
#'    "internal experience."
#'  )
#' )
#'
#' # Item attributes
#' aspects.of.personality.traits <- list(
#'  neuroticism = c("anxious", "depressed", "insecure", "emotional"),
#'  extraversion = c("friendly", "positive", "assertive", "energetic")
#' )
#'
#' # Name the field or specialty
#' domain <- "Personality Measurement"
#'
#' # Name the Inventory being created
#' scale.title <- "Two of 'Big Five:' A Streamlined Personality Inventory"
#'
#' # Add a file path name to a local text generation model downloaded on your computer
#' model.path <- "ADD FILE PATH TO DOWNLOADED MODEL HERE"
#'
#'
#' # Generate and validate items using a model installed on your machine
#' local_example <- local_AIGENIE(
#'  item.attributes = aspects.of.personality.traits,
#'  item.type.definitions = trait.definitions,
#'  domain = domain,
#'  model.path = model.path
#' )
#'
#' }
#' @export
local_AIGENIE <- function(
    # Required parameters
  item.attributes,
  model.path,
  embedding.model = "bert-base-uncased",

  # Optional content parameters
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

  # EGA parameters
  EGA.model = NULL,
  EGA.algorithm = NULL,
  EGA.uni.method = NULL,
  uva.cut.off = 0.20,

  # Local model parameters
  n.ctx = 4096,
  n.gpu.layers = -1,
  max.tokens = 1024,
  device = "auto",
  batch.size = 32,
  pooling.strategy = "mean",
  max.length = 512L,

  # Flags
  keep.org = FALSE,
  items.only = FALSE,
  embeddings.only = FALSE,
  adaptive = TRUE,
  run.overall = FALSE,
  all.together = FALSE,
  plot = TRUE,
  silently = FALSE
) {

  # Validate uva.cut.off (kept separate from the positional validator)
  uva.cut.off_validate(uva.cut.off)

  # Step 1: Validate all inputs
  validation <- validate_user_input_local_AIGENIE(
    item.attributes, model.path, embedding.model, main.prompts,
    temperature, top.p, target.N, domain, scale.title, item.examples,
    audience, item.type.definitions, response.options, prompt.notes,
    system.role, EGA.model, EGA.algorithm, EGA.uni.method, n.ctx,
    n.gpu.layers, max.tokens, device, batch.size, pooling.strategy,
    max.length, keep.org, items.only, embeddings.only, adaptive,
    run.overall, all.together, plot, silently
  )

  # Extract validated parameters
  target.N <- validation$target.N
  EGA.model <- validation$EGA.model
  EGA.uni.method <- validation$EGA.uni.method
  EGA.algorithm <- validation$EGA.algorithm
  item.type.definitions <- validation$item.type.definitions
  item.examples <- validation$item.examples
  item.attributes <- validation$item.attributes
  prompt.notes <- validation$prompt.notes
  main.prompts <- validation$main.prompts
  custom <- validation$custom
  model.path <- validation$model.path
  embedding.model <- validation$embedding.model
  n.ctx <- validation$n.ctx
  n.gpu.layers <- validation$n.gpu.layers
  max.tokens <- validation$max.tokens
  device <- validation$device
  batch.size <- validation$batch.size
  pooling.strategy <- validation$pooling.strategy
  max.length <- validation$max.length
  run.overall <- validation$run.overall
  all.together <- validation$all.together

  # Step 2: Check local setup
  setup_ok <- check_local_llm_setup(model.path, silently)
  if (!setup_ok) {
    stop("Local setup incomplete. Please run check_local_llm_setup() for details.")
  }

  # Step 3: Construct prompts (same as API version)
  system.role <- create_system.role(domain, scale.title, audience,
                                    response.options, system.role)

  if (!custom) {
    main.prompts <- create_main.prompts(item.attributes, item.type.definitions,
                                        domain, scale.title, prompt.notes,
                                        audience, item.examples)
  } else {
    main.prompts <- modify_main.prompts(main.prompts, item.attributes,
                                        item.type.definitions,
                                        domain, scale.title, prompt.notes,
                                        audience, item.examples)
  }

  # Step 4: Generate items using local LLM
  if (!silently) {
    cat("Generating items with local LLM\n")
    cat("----------------------------------------\n")
  }

  items_gen <- generate_items_via_local_llm(
    main.prompts, system.role, model.path,
    temperature, top.p, adaptive, silently,
    target.N, n.ctx, n.gpu.layers, max.tokens
  )

  items <- items_gen$items
  success <- items_gen$successful

  if (is.data.frame(items)) {
    items$ID <- 1:nrow(items)  # Add ID column
  }

  # Return if items only requested or generation failed
  if (items.only || !success) {
    if (!success && !silently) {
      message("Item generation failed. Returning partial results.")
    }
    return(items)
  }

  # Step 5: Generate embeddings using local model
  attempt_to_embed <- embed_items_local(
    embedding.model = embedding.model,
    items = items,
    pooling.strategy = pooling.strategy,
    device = device,
    batch.size = batch.size,
    max.length = max.length,
    silently = silently
  )

  success <- attempt_to_embed$success
  embeddings <- attempt_to_embed$embeddings

  # Return if embedding failed or embeddings only requested
  if (!success || embeddings.only) {
    if (!success && !silently) {
      message("Embedding generation failed. Returning items instead.")
      return(items)
    }
    if (embeddings.only) {
      return(list(embeddings = embeddings, items = items))
    }
  }

  # Step 6: Run reduction pipeline
  # Run as a single sample if applicable
  if(all.together){
    items_to_run <- run_all_together(items)

    # run the pipeline using the updated matrix
    try_item_level <- run_item_reduction_pipeline(embedding_matrix = embeddings,
                                                  items=items_to_run, EGA.model = EGA.model$overall,
                                                  EGA.algorithm = EGA.algorithm$overall,
                                                  EGA.uni.method = EGA.uni.method$overall,
                                                  uva.cut.off = uva.cut.off, keep.org = keep.org,
                                                  silently = silently, plot = plot)

    if(!try_item_level$success){

      if(!silently){
        message("AI-GENIE reduction failed. Returning partial results.")
      }

      return(try_item_level$item_level)
    }

    item_level <- try_item_level$item_level

    # Update the returned data frame appropriately
    IDs <- item_level[["All"]][["final_items"]]$ID
    item_level[["All"]][["final_items"]] <- items[items$ID %in% IDs,]
    if(keep.org){
      item_level[["All"]][["initial_items"]] <- items
    }

    return(item_level[["All"]])

  }



  # Item-level reduction
  try_item_level <- run_item_reduction_pipeline(
    embedding_matrix = embeddings, items=items,
    EGA.model = EGA.model$type, EGA.algorithm = EGA.algorithm$type,
    EGA.uni.method = EGA.uni.method$type, uva.cut.off = uva.cut.off,
    keep.org = keep.org, silently = silently,
    plot = plot
  )

  if (!try_item_level$success) {
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level

  if(run.overall && length(names(item.attributes)) > 1){
  # Overall reduction
    try_overall_result <- run_pipeline_for_all(
      item_level = item_level, items = items,
      embeddings = embeddings, model = EGA.model$overall,
      algorithm = EGA.algorithm$overall, uni.method = EGA.uni.method$overall,
      uva.cut.off = uva.cut.off,
      keep.org = keep.org, silently = silently, plot = plot
    )

    if (!try_overall_result$success) {
      return(item_level)
    }

    overall_result <- try_overall_result$overall_result
  } else {
    overall_result <- item_level
    try_overall_result <- list(success = TRUE)
  }

  # Step 7: Print results summary
  if(!silently && try_overall_result$success && try_item_level$success){
    print_results(overall_result, item_level, run.overall)
  }

  # Return results
  return(build_return(item_level, overall_result,
  run.overall, keep.org))
}


#' The use of the psychometric reduction component of AIGENIE on your pre-existing item pool
#'
#' @description
#' GENIE applies the psychometric reduction steps present in `AIGENIE` on user-supplied
#' items. Users provide their own items and optionally their own
#' embeddings, then `GENIE` performs redundancy reduction and
#' structural validation to assess item quality and dimensionality.
#'
#' @param items Data frame with columns: statement, attribute, type, ID.
#'   All columns must be character type except ID (numeric or character allowed).
#'   \itemize{
#'     \item \code{statement}: The actual item text
#'     \item \code{attribute}: The construct/attribute the item measures
#'     \item \code{type}: The item type/category
#'     \item \code{ID}: Unique identifier for each item
#'   }
#'
#' @param embedding.matrix Optional numeric matrix or data frame where:
#'   \itemize{
#'     \item Rows represent embedding dimensions
#'     \item Columns represent items (must match items$ID exactly)
#'     \item If `NULL`, embeddings will be generated using `embedding.model`
#'   }
#'
#' @param openai.API OpenAI API key (required if using OpenAI embedding models)
#' @param hf.token HuggingFace token (optional, improves rate limits for HF models)
#' @param jina.API Jina AI API key for using Jina embedding models (e.g., "jina-embeddings-v3").
#'   Free tier available at \url{https://jina.ai/}.
#' @param model Language model identifier (currently unused in GENIE)
#' @param embedding.model Embedding model to use if embedding.matrix not provided:
#'   \itemize{
#'     \item OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
#'     \item Jina AI: "jina-embeddings-v3", "jina-embeddings-v4", "jina-embeddings-v2-base-en" (requires jina.API)
#'     \item HuggingFace: "BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"
#'   }
#' @param EGA.model EGA network estimation model ("glasso", "TMFG", or NULL for auto-selection)
#' @param EGA.algorithm EGA community detection algorithm ("walktrap", "leiden", "louvain")
#' @param EGA.uni.method Unidimensionality assessment method ("louvain", "expand", "LE")
#' @param uva.cut.off Numeric in `[0, 1)`. wTO threshold passed to `EGAnet::UVA` for the
#'   redundancy-reduction step (default: 0.20). Lower values remove more items.
#' @param embeddings.only If `TRUE`, return embeddings and stop (skip network analysis)
#' @param run.overall A logical value (optional, default: FALSE). Controls whether a *fit* analysis
#'    on the complete item pool is run *post-reduction.*
#'    By default, only type-level reduction analyses are run (i.e., items of like-type go through
#'    the pipeline independent of the other items in the pool). When this flag is `TRUE`, an additional
#'    analysis is run on the overall sample, but no further reductions at the overall level are made.
#'    If only one item type is present, this argument will be ignored.
#'
#' @param all.together A logical value (optional, default: FALSE). Controls whether the *reduction* analysis
#'    on the complete item pool is run.
#'    By default, only type-level reduction analyses are run (i.e., items of like-type go through
#'    the pipeline independent of the other items in the pool). When this flag is `TRUE`, reductions are made
#'    at the overall level (i.e., all items go through the reduction pipeline together, agnostic of item type).
#'    If only one item type is present, this argument will be ignored.
#' @param plot If `TRUE`, display network comparison plots
#' @param silently If `TRUE`, suppress progress messages
#'
#' @return
#' The structure of the return value depends on the function flags.
#'
#' **Defaults:** `items.only = FALSE`, `embeddings.only = FALSE`,
#' `run.overall = FALSE`, `all.together = FALSE`.
#'
#' **When `items.only = TRUE`:**
#' Returns a `data.frame` of generated items with columns:
#' `ID`, `statement`, `type`, and `attribute`.
#'
#' **When `embeddings.only = TRUE`:**
#' Returns a named `list` with two elements:
#' \itemize{
#'   \item `embeddings` — an embedding matrix/list (columns or rownames correspond to item IDs).
#'   \item `items` — the items `data.frame` described above.
#' }
#'
#' **Default behaviour** (`items.only = FALSE`, `embeddings.only = FALSE`,
#' `run.overall = FALSE`, `all.together = FALSE`):
#' Returns a named `list` with two top-level elements:
#' \describe{
#'   \item{`item_type_level`}{A named list where each name is an item type and each element is a per-type named list containing:
#'     \describe{
#'       \item{`final_NMI`}{Numeric: final normalized mutual information after reduction.}
#'       \item{`initial_NMI`}{Numeric: initial NMI of the pre-reduced item pool.}
#'       \item{`embeddings`}{List or matrix of embeddings for this item type (see 'Notes on `embeddings`' below).}
#'       \item{`UVA`}{List from Unique Variable Analysis (contains at least `n_removed`, `n_sweeps`, `redundant_pairs` data.frame).}
#'       \item{`bootEGA`}{List with bootEGA results (e.g. `initial_boot`, `final_boot`, `n_removed`, `items_removed`, `initial_boot_with_redundancies`).}
#'       \item{`EGA.model_selected`}{Character: chosen EGA model (e.g. `"TMFG"` or `"Glasso"`).}
#'       \item{`final_items`}{`data.frame`: final items after reduction (columns include `ID`, `statement`, `attribute`, `type`, `EGA_com`).}
#'       \item{`final_EGA`}{EGA object (from EGAnet) after reduction.}
#'       \item{`initial_EGA`}{Initial EGA object computed on the pre-reduced item set.}
#'       \item{`start_N`}{Integer: initial number of items in this type.}
#'       \item{`final_N`}{Integer: final number of items in this type.}
#'       \item{`network_plot`}{`ggplot` / `patchwork` object comparing networks before vs after reduction.}
#'       \item{`stability_plot`}{`ggplot` / `patchwork` object showing item stability before vs after reduction.}
#'     }
#'   }
#'
#'   \item{`overall`}{Named list with aggregated results across all item types. Under the default this contains:
#'     \describe{
#'       \item{`final_items`}{`data.frame` of final items across all types (columns as above).}
#'       \item{`embeddings`}{Embeddings for the full reduced item set (see 'Notes on `embeddings`' below). Note: `overall$embeddings` does **not** include `selected`.}
#'     }
#'   }
#' }
#'
#' **When `run.overall = TRUE`** (`items.only = FALSE`, `embeddings.only = FALSE`):
#' \describe{
#'   \item{`item_type_level`}{Same per-type structure as the default (see above).}
#'   \item{`overall`}{A named list with aggregated results (not limited to `final_items` and `embeddings`) containing:
#'     `final_NMI`, `initial_NMI`, `embeddings`, `EGA.model_selected`, `final_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, and `network_plot`.}
#' }
#'
#' **When `all.together = TRUE`** (regardless of `run.overall`):
#' Results are **not** split into `item_type_level` and `overall`. Instead the function returns a single named list containing:
#' `final_NMI`, `initial_NMI`, `embeddings`, `UVA`, `bootEGA`, `EGA.model_selected`, `final_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`, and `stability_plot`.
#'
#'
#' @references
#' Golino, H. F., & Epskamp, S. (2017). Exploratory graph analysis: A new approach
#' for estimating the number of dimensions in psychological research.
#' \emph{PLOS ONE, 12}(6), e0174035.
#' \url{https://doi.org/10.1371/journal.pone.0174035}
#'
#' Christensen, A. P., Garrido, L. E., & Golino, H. (2023). Unique variable
#' analysis: A network psychometrics method to detect local dependence.
#' \emph{Multivariate Behavioral Research, 58}(6), 1165–1182.
#' \url{https://doi.org/10.1080/00273171.2023.2194606}
#'
#' Christensen, A. P., & Golino, H. (2021). Estimating the stability of
#' psychological dimensions via bootstrap exploratory graph analysis:
#' A Monte Carlo simulation and tutorial.
#' \emph{Psych, 3}(3), 479–500.
#' \url{https://doi.org/10.3390/psych3030032}
#'
#' Danon, L., Díaz-Guilera, A., Duch, J., & Arenas, A. (2005). Comparing
#' community structure identification.
#' \emph{Journal of Statistical Mechanics: Theory and Experiment, 2005}(9),
#' P09008.
#' \url{https://doi.org/10.1088/1742-5468/2005/09/P09008}
#'
#' Russell-Lasalandra, L. L., Christensen, A. P., & Golino, H. (2024).
#' Generative psychometrics via AI-GENIE: Automatic item generation and
#' validation via network-integrated evaluation.
#' \url{https://osf.io/preprints/psyarxiv/fgbj4_v2}.
#'
#' @examples
#'  \dontrun{
#' #####################################################################
#' #### GENIE with Bundled GPT-5.4 Items and Frozen Embeddings      ####
#' #####################################################################
#'
#' # Load the example item pool and its matching embedding matrix.
#' # No API key is required.
#' data("items.gpt5.4.example")
#' data("embeddings.gpt5.4.example")
#'
#' # Run GENIE using the pre-computed embeddings.
#' gpt5.4.example.results <- GENIE(
#'   items = items.gpt5.4.example,
#'   embedding.matrix = embeddings.gpt5.4.example,
#'   EGA.model = "glasso",
#'   EGA.algorithm = "walktrap",
#'   EGA.uni.method = "louvain",
#'   uva.cut.off = 0.20,
#'   run.overall = TRUE,
#'   plot = FALSE,
#'   silently = FALSE
#' )
#'
#' # Final retained items
#' gpt5.4.example.results$item_type_level$conscientiousness$final_items
#' gpt5.4.example.results$item_type_level$openness$final_items
#'
#' # Publication-ready audit of every filtered item:
#' # what was removed, why, the filtering statistic and cutoff,
#' # redundancy partner(s), item stability, and pre-reduction
#' # network-loading diagnostics.
#' gpt5.4.example.results$filtering_audit
#'
#' # Stage-by-stage NMI trajectories
#' gpt5.4.example.results$item_type_level$conscientiousness$reduction_summary
#' gpt5.4.example.results$item_type_level$openness$reduction_summary
#'
#' # Pooled post-reduction fit, when run.overall = TRUE
#' gpt5.4.example.results$overall
#'
#' ############################################################
#' #### Using GENIE with OpenAI's Embeddings (Recommended) ####
#' ############################################################
#'
#' # Add an OpenAI API Key
#' key <- "INSERT YOUR KEY HERE"
#'
#'
#' # Specify item statements that you already have written
#' statements <- c(
#'   "I find myself naturally initiating conversations with strangers at social gatherings.",
#'   "I enjoy creating a welcoming atmosphere for people I meet for the first time.",
#'   "I generally maintain a hopeful outlook, even when faced with challenges.",
#'   "I frequently find myself in a good mood, spreading cheer to those around me.",
#'   "I often have the drive to engage in exciting activities, even after a long day.",
#'   "I tend to tackle projects with enthusiasm and high energy from start to finish.",
#'   paste0(
#'     "I actively seek to include others in group activities, ",
#'     "making them feel part of the team."
#'   ),
#'   "I frequently reach out to new acquaintances to foster connections and friendships.",
#'   paste0(
#'     "I habitually focus on the silver lining in difficult ",
#'     "situations, maintaining an optimistic perspective."
#'   ),
#'   paste0(
#'     "I often express gratitude for the positive aspects of my ",
#'     "life, which enhances my overall mood."
#'   ),
#'   "I find joy in taking on new challenges that require a burst of energy and enthusiasm.",
#'   "I thrive in dynamic environments that keep me on my toes and invigorate my spirit.",
#'   "I take pleasure in introducing people to one another, acting as a social connector.",
#'   "I enjoy making others comfortable by engaging them in light-hearted conversation.",
#'   "I often set a positive tone in group settings with my upbeat demeanor.",
#'  "I approach each day with a sense of excitement and a positive mindset.",
#'  "I am drawn to fast-paced environments where I can express my high energy levels.",
#'  "I feel invigorated when working on multiple projects that demand my full attention.",
#'  "I take delight in meeting new people and quickly making them feel at ease.",
#'  paste0(
#'    "I find it rewarding to help shy or reserved individuals ",
#'    "become involved in group discussions."
#'  ),
#'  "I have a natural tendency to uplift others with my positive remarks and outlook.",
#'  paste0(
#'    "I find happiness in highlighting the successes of others, ",
#'    "contributing to a cheerful environment."
#'  ),
#'  "I eagerly immerse myself in activities that demand stamina and sustained energy.",
#'  "I often channel my vitality into hobbies and sports that require physical exertion.",
#'  "I feel rejuvenated when I bring people together to collaborate and share ideas.",
#'  "I often extend a genuine greeting to others, creating an inviting atmosphere.",
#'  "I regularly see challenges as opportunities for growth and learning.",
#'  "I commonly radiate positivity, influencing the mood of those around me.",
#'  "I approach mornings with anticipation and vigor, ready to embrace the day.",
#'  paste0(
#'    "I consistently infuse enthusiasm into group activities, ",
#'    "boosting collective energy levels."
#'  ),
#'  "I make an effort to connect with people by remembering details about their lives.",
#'  "I genuinely enjoy learning about people's diverse experiences and viewpoints.",
#'  "I have a habit of encouraging others to see the bright side of their situations.",
#'  "I believe in celebrating small victories, finding joy in daily accomplishments.",
#'  "I often find myself eager to start the day with ambitious plans and goals.",
#'  paste0(
#'    "I am known for sustaining high levels of energy during ",
#'    "extended work sessions or projects."
#'  ),
#'  "I make an effort to engage those around me in meaningful and enjoyable conversations.",
#'  "I often seek opportunities to bring people together, fostering a sense of community.",
#'  "I naturally inspire others with my optimistic outlook, even in uncertain times.",
#'  paste0(
#'    "I frequently look for the positive aspects in challenging ",
#'    "situations and share them with others."
#'  ),
#'  "I approach new experiences with an eagerness and fervor that motivates those around me.",
#'  "I thrive on maintaining high energy levels throughout demanding and fast-paced days.",
#'  paste0(
#'    "I take pleasure in initiating warm interactions in group ",
#'    "settings to make everyone comfortable."
#'  ),
#'  "I enjoy hosting gatherings that connect friends and encourage social bonding.",
#'  paste0(
#'    "I am skilled at turning setbacks into learning experiences ",
#'    "to maintain a positive outlook."
#'  ),
#'  "I always try to highlight the benefits in situations, enhancing a cheerful atmosphere.",
#'  "I find excitement in starting the day with a list of activities to energize my routine.",
#'  paste0(
#'    "I relish the challenge of keeping up with dynamic schedules ",
#'    "that require sustained energy."
#'  ),
#'  "I often find joy in making newcomers feel welcome and appreciated in group settings.",
#'  "I genuinely enjoy striking up conversations to learn more about the people I encounter.",
#'  "I have a knack for seeing potential in situations that others might overlook.",
#'  paste0(
#'    "I consistently try to uplift the mood in my surroundings ",
#'    "with hopeful and encouraging words."
#'  ),
#'  paste0(
#'    "I frequently harness my energy to inspire and motivate those ",
#'    "around me in team environments."
#'  ),
#'  "I often feel invigorated by challenges that require sustained focus and dynamic thinking.",
#'  "I often create environments where people feel encouraged to share their thoughts freely.",
#'  "I find it fulfilling to engage deeply with people, building lasting connections.",
#'  "I see potential in every day, believing it holds opportunities for something good.",
#'  "I actively focus on the pleasures of life, which naturally enhances my mood.",
#'  "I am invigorated by opportunities to engage in lively and spirited events.",
#'  "I tend to maintain momentum throughout the day, sustaining my energy levels.",
#'  paste0(
#'    "I frequently experience sudden shifts in my emotions even ",
#'    "when there is no apparent reason."
#'  ),
#'  "People often find it difficult to predict my emotional reactions to different situations.",
#'  "I often doubt my abilities and worry about whether I am meeting expectations.",
#'  "I frequently question my self-worth and tend to seek reassurance from others.",
#'  "I become annoyed easily over small inconveniences or delays.",
#'  paste0(
#'    "I often find myself feeling agitated or frustrated in ",
#'    "situations that don't bother most people."
#'  ),
#'  "My mood can change drastically over the course of a day, often without any clear reason.",
#'  "I tend to experience emotional highs and lows more intensely than those around me.",
#'  "I sometimes avoid taking on new challenges because I fear not being good enough.",
#'  "I often feel uncertain about my social standing and worry about being accepted by others.",
#'  "I find myself getting irritated quickly when things don't go my way.",
#'  "Minor annoyances often cause my patience to wear thin unusually fast.",
#'  "I frequently struggle to maintain a stable emotional state throughout the day.",
#'  "Unexpected events can cause me to experience drastic emotional swings.",
#'  "I often feel inadequate in comparison to others around me.",
#'  "I tend to second-guess my choices due to a lack of confidence in myself.",
#'  "I tend to become frustrated when things do not proceed as I have planned.",
#'  "I am prone to irritation when faced with unexpected changes to my routine.",
#'  paste0(
#'    "My emotional state is often unpredictable, shifting from ",
#'    "contentment to sadness with little warning."
#'  ),
#'  "People have commented that my emotions seem to fluctuate more than those of others.",
#'  "I frequently feel self-conscious about my achievements compared to those of my peers.",
#'  paste0(
#'    "I often worry excessively about making mistakes, even in ",
#'    "situations where it might be inconsequential."
#'  ),
#'  "Small disruptions in my daily routine can trigger strong feelings of annoyance.",
#'  "I find myself becoming irritated more quickly than others when under stress or pressure.",
#'  paste0(
#'    "I often find my emotional responses to be unpredictable, ",
#'    "feeling fine one moment and unsettled the next."
#'  ),
#'  "I experience strong emotions that can shift unexpectedly, often catching me off guard.",
#'  "I regularly feel uncertain about my ability to manage new responsibilities effectively.",
#'  "I often question my decisions, fearing they might not lead to the best outcomes.",
#'  paste0(
#'    "I frequently find myself reacting with impatience to ",
#'    "situations perceived as minor interruptions."
#'  ),
#'  "Even minor provocations can sometimes lead to an exaggerated sense of annoyance for me.",
#'  paste0(
#'    "My emotional state is often inconsistent, and I can feel ",
#'    "ecstatic or despondent within short timeframes."
#'  ),
#'  paste0(
#'    "I notice that my feelings can be quite volatile and intense, ",
#'    "affecting how I interact with others throughout the day."
#'  ),
#'  "I regularly doubt whether I am capable of achieving my personal or professional goals.",
#'  "I often seek validation from others to feel reassured about my self-worth.",
#'  paste0(
#'    "I am sensitive to disturbances and find my patience wearing ",
#'    "thin quickly when things aren't orderly."
#'  ),
#'  paste0(
#'    "I occasionally struggle to contain my annoyance over trivial ",
#'    "issues that disrupt my sense of calm."
#'  ),
#'  "I can go from feeling upbeat to being downcast without an obvious cause.",
#'  "My emotional responses can sometimes be unpredictable, shifting with little notice.",
#'  "I often feel the need for affirmation about my abilities from friends or colleagues.",
#'  "I tend to compare myself to others and feel uncertain about my achievements.",
#'  "I find myself easily bothered by noises or disturbances in my environment.",
#'  "I get easily flustered by situations that interrupt my planned activities.",
#'  paste0(
#'    "I find it challenging to maintain a consistent emotional ",
#'    "state, regardless of external situations."
#'  ),
#'  "My emotional reactions can be intense and differ significantly from moment to moment.",
#'  "I have a persistent fear of not measuring up to the expectations placed on me.",
#'  "I often feel anxious about others' perceptions of my capabilities and appearance.",
#'  "I am quick to express frustration at minor inconveniences in my daily routine.",
#'  paste0(
#'    "I find that small, unforeseen events often disrupt my sense ",
#'    "of calm and lead to irritation."
#'  ),
#'  paste0(
#'    "My emotional reactions can be strong and relentless, ",
#'    "impacting my behavior throughout the day."
#'  ),
#'  paste0(
#'    "I often find myself emotionally labile, with an inner ",
#'    "turbulence that others rarely perceive."
#'  ),
#'  "I frequently worry about my competence in areas where others seem confident.",
#'  paste0(
#'    "I have a tendency to second-guess myself and require ",
#'    "affirmation to feel reassured about my choices."
#'  ),
#'  "Small disruptions can ignite a lingering sense of agitation within me.",
#'  "I often catch myself feeling irritable even in relatively calm settings.",
#'  paste0(
#'    "I find myself swinging from happy to melancholic in a short ",
#'    "span of time, often surprising even myself."
#'  ),
#'  paste0(
#'    "Others often comment on how quickly my mood can change in ",
#'    "response to seemingly minor events."
#'  ),
#'  paste0(
#'    "I tend to feel apprehensive about presenting my opinions, ",
#'    "fearing they may be judged harshly."
#'  ),
#'  "I often require reassurance from peers to feel confident in my decisions and ideas.",
#'  "Interruptions during focused tasks often lead to an outpour of irritation from me.",
#'  "I struggle to keep my frustration in check when things do not unfold as expected."
#' )
#'
#'
#' # Create the item type and attribute labels
#' item.attributes <- c(
#'  rep(c("friendly", "positive", "energetic"), each = 2, times = 10),
#'  rep(c("moody", "insecure", "irritable"), each = 2, times = 10)
#' )
#' item.types <- c(
#'  rep("extraversion", 60),
#'  rep("neuroticism", 60)
#' )
#'
#'
#'
#'
#' # Build your data frame with the required columns: ID, statement, attribute, and type
#' items_df <- data.frame(
#'  ID = rep(as.factor(1:length(statements))),
#'  statement = statements,
#'  attribute = item.attributes,
#'  type = item.types
#' )
#'
#'
#' # Run GENIE with items you provide (embedding items via OpenAI)
#' example_reduction <- GENIE(items = items_df,
#'                           openai.API = key)
#'
#' # View the results
#' View(example_reduction)
#'
#'
#' ################################################################
#' ###### Or, Run GENIE with a Hugging Face Embedding Model #######
#' ################################################################
#'
#' # Chose a BAAI/bge series OR thenlper/gte series model
#' hf.embedding.model <- "BAAI/bge-large-en-v1.5"
#'
#' # Create a HF Token to access the best models. Moderate useage will still be FREE
#' hf.token <- "INSERT YOUR KEY HERE"
#'
#' # Run GENIE using the Hugging Face Embedding model
#' example_reduction_HF <- GENIE(items = items_df,
#'                              embedding.model = hf.embedding.model,
#'                              hf.token = hf.token)
#'
#'
#'
#'}
#' @export
GENIE <- function(
    items,                                    # Required: user items
    embedding.matrix = NULL,                  # Optional: user embeddings

    # API parameters
    openai.API = NULL,
    hf.token = NULL,
    jina.API = NULL,

    # Embedding parameters
    embedding.model = "text-embedding-3-small",

    # EGA parameters
    EGA.model = NULL,
    EGA.algorithm = NULL,
    EGA.uni.method = NULL,
    uva.cut.off = 0.20,

    # Control flags
    embeddings.only = FALSE,
    run.overall = FALSE,
    all.together = FALSE,
    plot = TRUE,
    silently = FALSE
) {

  # Validate uva.cut.off (kept separate from validate_user_input_GENIE)
  uva.cut.off_validate(uva.cut.off)

  # Step 1: Comprehensive input validation
  validation <- validate_user_input_GENIE(
    items = items,
    embedding.matrix = embedding.matrix,
    openai.API = openai.API,
    hf.token = hf.token,
    jina.API = jina.API,
    embedding.model = embedding.model,
    EGA.model = EGA.model,
    EGA.algorithm = EGA.algorithm,
    EGA.uni.method = EGA.uni.method,
    embeddings.only = embeddings.only,
    run.overall = run.overall,
    all.together = all.together,
    plot = plot,
    silently = silently
  )

  # Extract validated parameters
  items <- validation$items
  embedding.matrix <- validation$embedding.matrix
  item.attributes <- validation$item.attributes
  EGA.model <- validation$EGA.model
  EGA.algorithm <- validation$EGA.algorithm
  EGA.uni.method <- validation$EGA.uni.method
  embedding.model <- validation$embedding.model
  provider <- validation$provider
  openai.API <- validation$openai.API
  hf.token <- validation$hf.token
  jina.API <- validation$jina.API
  run.overall <- validation$run.overall
  all.together <- validation$all.together

  # Step 2: Handle embeddings (generate if not provided)
  if (is.null(embedding.matrix)) {
    if (!silently) {
      cat("Generating embeddings using", embedding.model, "\n")
    }

    # Generate embeddings using unified provider dispatch
    embedding_result <- generate_embeddings(
      embedding.model = embedding.model,
      items = items,
      openai.API = openai.API,
      hf.token = hf.token,
      jina.API = jina.API,
      silently = silently
    )

    if (!embedding_result$success) {
      stop("Failed to generate embeddings. Please check your API credentials and model selection.")
    }

    embeddings <- embedding_result$embeddings

  } else {
    if (!silently) {
      cat("Using provided embedding matrix\n")
    }
    embeddings <- embedding.matrix
  }

  # Step 3: Return embeddings if that's all that was requested
  if (embeddings.only) {
    return(embeddings)
  }

  # Step 4: Run the network psychometric pipeline (same as AIGENIE)

  # Run as a single sample if applicable
  if(all.together){
    items_to_run <- run_all_together(items)

    # run the pipeline using the updated matrix
    try_item_level <- run_item_reduction_pipeline(embedding_matrix = embeddings,
                                                  items=items_to_run, EGA.model = EGA.model$overall,
                                                  EGA.algorithm = EGA.algorithm$overall,
                                                  EGA.uni.method = EGA.uni.method$overall,
                                                  uva.cut.off = uva.cut.off, keep.org = FALSE,
                                                  silently = silently, plot = plot)

    if(!try_item_level$success){

      if(!silently){
        message("AI-GENIE reduction failed. Returning partial results.")
      }

      return(try_item_level$item_level)
    }

    item_level <- try_item_level$item_level

    # Update the returned data frame appropriately
    IDs <- item_level[["All"]][["final_items"]]$ID
    item_level[["All"]][["final_items"]] <- items[items$ID %in% IDs,]

    return(item_level[["All"]])

  }

  # Item-level analysis
  try_item_level <- run_item_reduction_pipeline(
    embedding_matrix = embeddings,
    items = items,
    EGA.model = EGA.model$type,
    EGA.algorithm = EGA.algorithm$type,
    EGA.uni.method = EGA.uni.method$type,
    uva.cut.off = uva.cut.off,
    keep.org = FALSE,  # GENIE doesn't need to keep original embeddings
    silently = silently,
    plot = plot
  )

  if (!try_item_level$success) {
    warning("GENIE: Item-level analysis failed. Returning partial results.")
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level

  # Overall analysis
  if(run.overall && length(names(item.attributes)) > 1){
  try_overall_result <- run_pipeline_for_all(
    item_level = item_level,
    items = items,
    embeddings = embeddings,
    model = EGA.model$overall,
    algorithm = EGA.algorithm$overall,
    uni.method = EGA.uni.method$overall,
    uva.cut.off = uva.cut.off,
    keep.org = FALSE,  # GENIE doesn't need to keep original data
    silently = silently,
    plot = plot
  )

  if (!try_overall_result$success) {
    warning("Overall analysis failed. Returning item-level results only.")
    return(item_level)
  }

  overall_result <- try_overall_result$overall_result
  } else {
    overall_result <- item_level
    try_overall_result <- list(success = TRUE)
  }

  # Step 5: Display results summary
  if(!silently && try_overall_result$success && try_item_level$success){
    print_results(overall_result, item_level, run.overall)
  }

  # Step 6: Return comprehensive results
  # prepare return object
  return(build_return(item_level, overall_result,
                      run.overall, keep.org = FALSE))
}





#' Local Generative Network-Integrated Evaluation (local_GENIE)
#'
#' @description
#' Local version of GENIE that uses locally installed embedding models for complete
#' privacy and offline operation. Provides the same psychometric validation and
#' quality assessment for user-supplied items as GENIE, but generates embeddings
#' locally using transformer models instead of API calls.
#'
#' @param items Data frame with columns: statement, attribute, type, ID.
#'   All columns must be character type except ID (numeric or character allowed).
#'   \itemize{
#'     \item \code{statement}: The actual item text
#'     \item \code{attribute}: The construct/attribute the item measures
#'     \item \code{type}: The item type/category
#'     \item \code{ID}: Unique identifier for each item
#'   }
#'
#' @param embedding.matrix Optional numeric matrix or data frame where:
#'   \itemize{
#'     \item Rows represent embedding dimensions
#'     \item Columns represent items (must match items$ID exactly)
#'     \item If `NULL`, embeddings will be generated using `embedding.model`
#'   }
#'
#' @param embedding.model Local embedding model identifier or path. Compatible models:
#'   \itemize{
#'     \item BERT variants: "bert-base-uncased", "bert-large-uncased"
#'     \item RoBERTa: "roberta-base", "roberta-large"
#'     \item DeBERTa: "microsoft/deberta-v3-base", "microsoft/deberta-v3-large"
#'     \item DistilBERT: "distilbert-base-uncased"
#'     \item Local paths: e.g., "/path/to/local/model"
#'   }
#'
#' @param device Device for embedding computation:
#'   \itemize{
#'     \item "auto": Automatically detect best available device
#'     \item "cpu": Force CPU usage
#'     \item "cuda": Use NVIDIA GPU (if available)
#'     \item "mps": Use Apple Silicon GPU (if available)
#'   }
#'
#' @param batch.size Number of items to process simultaneously (default: 32)
#' @param pooling.strategy Method for pooling token embeddings:
#'   \itemize{
#'     \item "mean": Average all token embeddings (default)
#'     \item "cls": Use only the CLS token embedding
#'     \item "max": Max pooling across tokens
#'   }
#' @param max.length Maximum sequence length for tokenization (default: 512)
#'
#' @param EGA.model Network estimation model ("glasso", "TMFG", or NULL for auto-selection)
#' @param EGA.algorithm Community detection algorithm ("walktrap", "leiden", "louvain")
#' @param EGA.uni.method Unidimensionality assessment method ("louvain", "expand", "LE")
#' @param uva.cut.off Numeric in `[0, 1)`. wTO threshold passed to `EGAnet::UVA` for the
#'   redundancy-reduction step (default: 0.20). Lower values remove more items.
#'
#' @param embeddings.only If `TRUE`, return embeddings and stop (skip network analysis)
#' @param run.overall A logical value (optional, default: FALSE). Controls whether a *fit* analysis
#'    on the complete item pool is run *post-reduction.*
#'    By default, only type-level reduction analyses are run (i.e., items of like-type go through
#'    the pipeline independent of the other items in the pool). When this flag is `TRUE`, an additional
#'    analysis is run on the overall sample, but no further reductions at the overall level are made.
#'    If only one item type is present, this argument will be ignored.
#'
#' @param all.together A logical value (optional, default: FALSE). Controls whether the *reduction* analysis
#'    on the complete item pool is run.
#'    By default, only type-level reduction analyses are run (i.e., items of like-type go through
#'    the pipeline independent of the other items in the pool). When this flag is `TRUE`, reductions are made
#'    at the overall level (i.e., all items go through the reduction pipeline together, agnostic of item type).
#'    If only one item type is present, this argument will be ignored.
#' @param plot If `TRUE`, display network comparison plots
#' @param silently If `TRUE`, suppress progress messages
#'
#' @return
#' **Defaults:** `items.only = FALSE`, `embeddings.only = FALSE`,
#' `run.overall = FALSE`, `all.together = FALSE`.
#'
#' **When `items.only = TRUE`:**
#' Returns a `data.frame` of generated items with columns:
#' `ID`, `statement`, `type`, and `attribute`.
#'
#' **When `embeddings.only = TRUE`:**
#' Returns a named `list` with two elements:
#' \itemize{
#'   \item `embeddings` — an embedding matrix/list (columns or rownames correspond to item IDs).
#'   \item `items` — the items `data.frame` described above.
#' }
#'
#' **Default behaviour** (`items.only = FALSE`, `embeddings.only = FALSE`,
#' `run.overall = FALSE`, `all.together = FALSE`):
#' Returns a named `list` with two top-level elements:
#' \describe{
#'   \item{`item_type_level`}{A named list where each name is an item type and each element is a per-type named list containing:
#'     \describe{
#'       \item{`final_NMI`}{Numeric: final normalized mutual information after reduction.}
#'       \item{`initial_NMI`}{Numeric: initial NMI of the pre-reduced item pool.}
#'       \item{`embeddings`}{List or matrix of embeddings for this item type (see 'Notes on `embeddings`' below).}
#'       \item{`UVA`}{List from Unique Variable Analysis (contains at least `n_removed`, `n_sweeps`, `redundant_pairs` data.frame).}
#'       \item{`bootEGA`}{List with bootEGA results (e.g. `initial_boot`, `final_boot`, `n_removed`, `items_removed`, `initial_boot_with_redundancies`).}
#'       \item{`EGA.model_selected`}{Character: chosen EGA model (e.g. `"TMFG"` or `"Glasso"`).}
#'       \item{`final_items`}{`data.frame`: final items after reduction (columns include `ID`, `statement`, `attribute`, `type`, `EGA_com`).}
#'       \item{`final_EGA`}{EGA object (from EGAnet) after reduction.}
#'       \item{`initial_EGA`}{Initial EGA object computed on the pre-reduced item set.}
#'       \item{`start_N`}{Integer: initial number of items in this type.}
#'       \item{`final_N`}{Integer: final number of items in this type.}
#'       \item{`network_plot`}{`ggplot` / `patchwork` object comparing networks before vs after reduction.}
#'       \item{`stability_plot`}{`ggplot` / `patchwork` object showing item stability before vs after reduction.}
#'     }
#'   }
#'
#'   \item{`overall`}{Named list with aggregated results across all item types. Under the default this contains:
#'     \describe{
#'       \item{`final_items`}{`data.frame` of final items across all types (columns as above).}
#'       \item{`embeddings`}{Embeddings for the full reduced item set (see 'Notes on `embeddings`' below). Note: `overall$embeddings` does **not** include `selected`.}
#'     }
#'   }
#' }
#'
#' **When `run.overall = TRUE`** (`items.only = FALSE`, `embeddings.only = FALSE`):
#' \describe{
#'   \item{`item_type_level`}{Same per-type structure as the default (see above).}
#'   \item{`overall`}{A named list with aggregated results (not limited to `final_items` and `embeddings`) containing:
#'     `final_NMI`, `initial_NMI`, `embeddings`, `EGA.model_selected`, `final_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, and `network_plot`.}
#' }
#'
#' **When `all.together = TRUE`** (regardless of `run.overall`):
#' Results are **not** split into `item_type_level` and `overall`. Instead the function returns a single named list containing:
#' `final_NMI`, `initial_NMI`, `embeddings`, `UVA`, `bootEGA`, `EGA.model_selected`, `final_items`, `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`, and `stability_plot`.
#'
#' @examples
#' \dontrun{
#' ###################################################
#' #### Using GENIE with a Local Embedding Model  ####
#' ###################################################
#'
#' # First, ensure that your machine is configured to compute local generation
#' install_local_llm_support()
#' # Once ready, continue to run GENIE on your data frame
#'
#'
#' # Specify item statements that you already have written
#' statements <- c(
#'   "I find myself naturally initiating conversations with strangers at social gatherings.",
#'   "I enjoy creating a welcoming atmosphere for people I meet for the first time.",
#'   "I generally maintain a hopeful outlook, even when faced with challenges.",
#'   "I frequently find myself in a good mood, spreading cheer to those around me.",
#'   "I often have the drive to engage in exciting activities, even after a long day.",
#'   "I tend to tackle projects with enthusiasm and high energy from start to finish.",
#'   paste0(
#'     "I actively seek to include others in group activities, ",
#'     "making them feel part of the team."
#'   ),
#'   "I frequently reach out to new acquaintances to foster connections and friendships.",
#'   paste0(
#'     "I habitually focus on the silver lining in difficult ",
#'     "situations, maintaining an optimistic perspective."
#'   ),
#'   paste0(
#'     "I often express gratitude for the positive aspects of my ",
#'     "life, which enhances my overall mood."
#'   ),
#'   "I find joy in taking on new challenges that require a burst of energy and enthusiasm.",
#'   "I thrive in dynamic environments that keep me on my toes and invigorate my spirit.",
#'   "I take pleasure in introducing people to one another, acting as a social connector.",
#'   "I enjoy making others comfortable by engaging them in light-hearted conversation.",
#'   "I often set a positive tone in group settings with my upbeat demeanor.",
#'   "I approach each day with a sense of excitement and a positive mindset.",
#'   "I am drawn to fast-paced environments where I can express my high energy levels.",
#'   "I feel invigorated when working on multiple projects that demand my full attention.",
#'   "I take delight in meeting new people and quickly making them feel at ease.",
#'   paste0(
#'     "I find it rewarding to help shy or reserved individuals ",
#'     "become involved in group discussions."
#'   ),
#'   "I have a natural tendency to uplift others with my positive remarks and outlook.",
#'   paste0(
#'     "I find happiness in highlighting the successes of others, ",
#'     "contributing to a cheerful environment."
#'   ),
#'   "I eagerly immerse myself in activities that demand stamina and sustained energy.",
#'   "I often channel my vitality into hobbies and sports that require physical exertion.",
#'   "I feel rejuvenated when I bring people together to collaborate and share ideas.",
#'   "I often extend a genuine greeting to others, creating an inviting atmosphere.",
#'   "I regularly see challenges as opportunities for growth and learning.",
#'   "I commonly radiate positivity, influencing the mood of those around me.",
#'   "I approach mornings with anticipation and vigor, ready to embrace the day.",
#'   paste0(
#'     "I consistently infuse enthusiasm into group activities, ",
#'     "boosting collective energy levels."
#'   ),
#'   "I make an effort to connect with people by remembering details about their lives.",
#'   "I genuinely enjoy learning about people's diverse experiences and viewpoints.",
#'   "I have a habit of encouraging others to see the bright side of their situations.",
#'   "I believe in celebrating small victories, finding joy in daily accomplishments.",
#'   "I often find myself eager to start the day with ambitious plans and goals.",
#'   paste0(
#'     "I am known for sustaining high levels of energy during ",
#'     "extended work sessions or projects."
#'   ),
#'   "I make an effort to engage those around me in meaningful and enjoyable conversations.",
#'   "I often seek opportunities to bring people together, fostering a sense of community.",
#'   "I naturally inspire others with my optimistic outlook, even in uncertain times.",
#'   paste0(
#'     "I frequently look for the positive aspects in challenging ",
#'     "situations and share them with others."
#'   ),
#'   "I approach new experiences with an eagerness and fervor that motivates those around me.",
#'   "I thrive on maintaining high energy levels throughout demanding and fast-paced days.",
#'   paste0(
#'     "I take pleasure in initiating warm interactions in group ",
#'     "settings to make everyone comfortable."
#'   ),
#'   "I enjoy hosting gatherings that connect friends and encourage social bonding.",
#'   paste0(
#'     "I am skilled at turning setbacks into learning experiences ",
#'     "to maintain a positive outlook."
#'   ),
#'   "I always try to highlight the benefits in situations, enhancing a cheerful atmosphere.",
#'   "I find excitement in starting the day with a list of activities to energize my routine.",
#'   paste0(
#'     "I relish the challenge of keeping up with dynamic schedules ",
#'     "that require sustained energy."
#'   ),
#'   "I often find joy in making newcomers feel welcome and appreciated in group settings.",
#'   "I genuinely enjoy striking up conversations to learn more about the people I encounter.",
#'   "I have a knack for seeing potential in situations that others might overlook.",
#'   paste0(
#'     "I consistently try to uplift the mood in my surroundings ",
#'     "with hopeful and encouraging words."
#'   ),
#'   paste0(
#'     "I frequently harness my energy to inspire and motivate those ",
#'     "around me in team environments."
#'   ),
#'   paste0(
#'     "I often feel invigorated by challenges that require ",
#'     "sustained focus and dynamic thinking."
#'   ),
#'   "I often create environments where people feel encouraged to share their thoughts freely.",
#'   "I find it fulfilling to engage deeply with people, building lasting connections.",
#'   "I see potential in every day, believing it holds opportunities for something good.",
#'   "I actively focus on the pleasures of life, which naturally enhances my mood.",
#'   "I am invigorated by opportunities to engage in lively and spirited events.",
#'   "I tend to maintain momentum throughout the day, sustaining my energy levels.",
#'   paste0(
#'     "I frequently experience sudden shifts in my emotions even ",
#'     "when there is no apparent reason."
#'   ),
#'   paste0(
#'     "People often find it difficult to predict my emotional ",
#'     "reactions to different situations."
#'   ),
#'   "I often doubt my abilities and worry about whether I am meeting expectations.",
#'   "I frequently question my self-worth and tend to seek reassurance from others.",
#'   "I become annoyed easily over small inconveniences or delays.",
#'   paste0(
#'     "I often find myself feeling agitated or frustrated in ",
#'     "situations that don't bother most people."
#'   ),
#'   "My mood can change drastically over the course of a day, often without any clear reason.",
#'   "I tend to experience emotional highs and lows more intensely than those around me.",
#'   "I sometimes avoid taking on new challenges because I fear not being good enough.",
#'   paste0(
#'     "I often feel uncertain about my social standing and worry ",
#'     "about being accepted by others."
#'   ),
#'   "I find myself getting irritated quickly when things don't go my way.",
#'   "Minor annoyances often cause my patience to wear thin unusually fast.",
#'   "I frequently struggle to maintain a stable emotional state throughout the day.",
#'   "Unexpected events can cause me to experience drastic emotional swings.",
#'   "I often feel inadequate in comparison to others around me.",
#'   "I tend to second-guess my choices due to a lack of confidence in myself.",
#'   "I tend to become frustrated when things do not proceed as I have planned.",
#'   "I am prone to irritation when faced with unexpected changes to my routine.",
#'   paste0(
#'     "My emotional state is often unpredictable, shifting from ",
#'     "contentment to sadness with little warning."
#'   ),
#'   "People have commented that my emotions seem to fluctuate more than those of others.",
#'   "I frequently feel self-conscious about my achievements compared to those of my peers.",
#'   paste0(
#'     "I often worry excessively about making mistakes, even in ",
#'     "situations where it might be inconsequential."
#'   ),
#'   "Small disruptions in my daily routine can trigger strong feelings of annoyance.",
#'   "I find myself becoming irritated more quickly than others when under stress or pressure.",
#'   paste0(
#'     "I often find my emotional responses to be unpredictable, ",
#'     "feeling fine one moment and unsettled the next."
#'   ),
#'   "I experience strong emotions that can shift unexpectedly, often catching me off guard.",
#'   "I regularly feel uncertain about my ability to manage new responsibilities effectively.",
#'   "I often question my decisions, fearing they might not lead to the best outcomes.",
#'   paste0(
#'     "I frequently find myself reacting with impatience to ",
#'     "situations perceived as minor interruptions."
#'   ),
#'   "Even minor provocations can sometimes lead to an exaggerated sense of annoyance for me.",
#'   paste0(
#'     "My emotional state is often inconsistent, and I can feel ",
#'     "ecstatic or despondent within short timeframes."
#'   ),
#'   paste0(
#'     "I notice that my feelings can be quite volatile and intense, ",
#'     "affecting how I interact with others throughout the day."
#'   ),
#'   "I regularly doubt whether I am capable of achieving my personal or professional goals.",
#'   "I often seek validation from others to feel reassured about my self-worth.",
#'   paste0(
#'     "I am sensitive to disturbances and find my patience wearing ",
#'     "thin quickly when things aren't orderly."
#'   ),
#'   paste0(
#'     "I occasionally struggle to contain my annoyance over trivial ",
#'     "issues that disrupt my sense of calm."
#'   ),
#'   "I can go from feeling upbeat to being downcast without an obvious cause.",
#'   "My emotional responses can sometimes be unpredictable, shifting with little notice.",
#'   "I often feel the need for affirmation about my abilities from friends or colleagues.",
#'   "I tend to compare myself to others and feel uncertain about my achievements.",
#'   "I find myself easily bothered by noises or disturbances in my environment.",
#'   "I get easily flustered by situations that interrupt my planned activities.",
#'   paste0(
#'     "I find it challenging to maintain a consistent emotional ",
#'     "state, regardless of external situations."
#'   ),
#'   "My emotional reactions can be intense and differ significantly from moment to moment.",
#'   "I have a persistent fear of not measuring up to the expectations placed on me.",
#'   "I often feel anxious about others' perceptions of my capabilities and appearance.",
#'   "I am quick to express frustration at minor inconveniences in my daily routine.",
#'   paste0(
#'     "I find that small, unforeseen events often disrupt my sense ",
#'     "of calm and lead to irritation."
#'   ),
#'   paste0(
#'     "My emotional reactions can be strong and relentless, ",
#'     "impacting my behavior throughout the day."
#'   ),
#'   paste0(
#'     "I often find myself emotionally labile, with an inner ",
#'     "turbulence that others rarely perceive."
#'   ),
#'   "I frequently worry about my competence in areas where others seem confident.",
#'   paste0(
#'     "I have a tendency to second-guess myself and require ",
#'     "affirmation to feel reassured about my choices."
#'   ),
#'   "Small disruptions can ignite a lingering sense of agitation within me.",
#'   "I often catch myself feeling irritable even in relatively calm settings.",
#'   paste0(
#'     "I find myself swinging from happy to melancholic in a short ",
#'     "span of time, often surprising even myself."
#'   ),
#'   paste0(
#'     "Others often comment on how quickly my mood can change in ",
#'     "response to seemingly minor events."
#'   ),
#'   paste0(
#'     "I tend to feel apprehensive about presenting my opinions, ",
#'     "fearing they may be judged harshly."
#'   ),
#'   "I often require reassurance from peers to feel confident in my decisions and ideas.",
#'   "Interruptions during focused tasks often lead to an outpour of irritation from me.",
#'   "I struggle to keep my frustration in check when things do not unfold as expected."
#' )
#'
#'
#' # Create the item type and attribute labels
#' item.attributes <- c(
#'   rep(c("friendly", "positive", "energetic"), each = 2, times = 10),
#'   rep(c("moody", "insecure", "irritable"), each = 2, times = 10)
#' )
#' item.types <- c(
#'   rep("extraversion", 60),
#'   rep("neuroticism", 60)
#' )
#'
#'
#' # Build your data frame with the required columns: ID, statement, attribute, and type
#' items_df <- data.frame(
#'   ID = rep(as.factor(1:length(statements))),
#'   statement = statements,
#'   attribute = item.attributes,
#'   type = item.types
#' )
#'
#'
#' # Run GENIE with items you provide with the default local embedding model ("bert-base-uncased")
#' example_reduction <- local_GENIE(items = items_df)
#'
#' # View the results
#' View(example_reduction)
#'
#'
#' ########################################################################################
#' ###### Or, Run local_GENIE with a locally-install Embedding Model of your Choice #######
#' #######################################################################################
#'
#' # Provide the path to a compatible locally installed embedding model
#' # Note that this package will not install the model for you, it should already be
#' # installed and ready to go
#' embedding.model <- "ADD YOUR PATH HERE"
#'
#' # Run GENIE using the Hugging Face Embedding model
#' example_reduction_your_model <- local_GENIE(items = items_df,
#'                               embedding.model = embedding.model)
#'
#' }
#'
#' @references
#' Golino, H. F., & Epskamp, S. (2017). Exploratory graph analysis: A new approach
#' for estimating the number of dimensions in psychological research.
#' \emph{PLOS ONE, 12}(6), e0174035.
#' \url{https://doi.org/10.1371/journal.pone.0174035}
#'
#' Christensen, A. P., Garrido, L. E., & Golino, H. (2023). Unique variable
#' analysis: A network psychometrics method to detect local dependence.
#' \emph{Multivariate Behavioral Research, 58}(6), 1165–1182.
#' \url{https://doi.org/10.1080/00273171.2023.2194606}
#'
#' Christensen, A. P., & Golino, H. (2021). Estimating the stability of
#' psychological dimensions via bootstrap exploratory graph analysis:
#' A Monte Carlo simulation and tutorial.
#' \emph{Psych, 3}(3), 479–500.
#' \url{https://doi.org/10.3390/psych3030032}
#'
#' Danon, L., Díaz-Guilera, A., Duch, J., & Arenas, A. (2005). Comparing
#' community structure identification.
#' \emph{Journal of Statistical Mechanics: Theory and Experiment, 2005}(9),
#' P09008.
#' \url{https://doi.org/10.1088/1742-5468/2005/09/P09008}
#'
#' Russell-Lasalandra, L. L., Christensen, A. P., & Golino, H. (2024).
#' Generative psychometrics via AI-GENIE: Automatic item generation and
#' validation via network-integrated evaluation.
#' \url{https://osf.io/preprints/psyarxiv/fgbj4_v2}.
#'
#' @export
local_GENIE <- function(
    # Required parameter
  items,
  embedding.matrix = NULL,

  # Local embedding parameters
  embedding.model = "bert-base-uncased",
  device = "auto",
  batch.size = 32,
  pooling.strategy = "mean",
  max.length = 512,

  # EGA parameters
  EGA.model = NULL,
  EGA.algorithm = NULL,
  EGA.uni.method = NULL,
  uva.cut.off = 0.20,

  # Control flags
  embeddings.only = FALSE,
  run.overall = FALSE,
  all.together = FALSE,
  plot = TRUE,
  silently = FALSE
) {

  # Validate uva.cut.off (kept separate from validate_user_input_local_GENIE)
  uva.cut.off_validate(uva.cut.off)

  # Step 1: Comprehensive input validation
  validation <- validate_user_input_local_GENIE(
    items = items,
    embedding.matrix = embedding.matrix,
    embedding.model = embedding.model,
    device = device,
    batch.size = batch.size,
    pooling.strategy = pooling.strategy,
    max.length = max.length,
    EGA.model = EGA.model,
    EGA.algorithm = EGA.algorithm,
    EGA.uni.method = EGA.uni.method,
    embeddings.only = embeddings.only,
    run.overall = run.overall,
    all.together = all.together,
    plot = plot,
    silently = silently
  )

  # Extract validated parameters
  items <- validation$items
  embedding.matrix <- validation$embedding.matrix
  item.attributes <- validation$item.attributes
  embedding.model <- validation$embedding.model
  device <- validation$device
  batch.size <- validation$batch.size
  pooling.strategy <- validation$pooling.strategy
  max.length <- validation$max.length
  EGA.model <- validation$EGA.model
  EGA.algorithm <- validation$EGA.algorithm
  EGA.uni.method <- validation$EGA.uni.method
  all.together <- validation$all.together
  run.overall <- validation$run.overall

  if(is.null(embedding.matrix)){ # the user needs us to generate embeddings

    embedding_result <- embed_items_local(
      embedding.model = embedding.model,
      items = items,
      pooling.strategy = pooling.strategy,
      device = device,
      batch.size = batch.size,
      max.length = max.length,
      silently = silently
    )

    if (!embedding_result$success) {
      stop("Failed to generate embeddings locally. Please check your model setup and system requirements.")
    }

    embeddings <- embedding_result$embeddings

  } else {
    if (!silently) {
      cat("Using provided embedding matrix\n")
    }
    embeddings <- embedding.matrix # the user provided embeddings
  }

  # Step 3: Return embeddings if that's all that was requested
  if (embeddings.only) {
    if (!silently) {
      cat("Local embeddings generated successfully. Returning embeddings and items.\n")
    }
    return(list(
      embeddings = embeddings,
      items = items
    ))
  }

  # Step 4: Run the network psychometric pipeline (same as regular GENIE)
  if (!silently) {
    cat("Running network psychometric analysis...\n")
  }

  # Run as a single sample if applicable
  if(all.together){
    items_to_run <- run_all_together(items)

    # run the pipeline using the updated matrix
    try_item_level <- run_item_reduction_pipeline(embedding_matrix = embeddings,
                                                  items=items_to_run, EGA.model = EGA.model$overall,
                                                  EGA.algorithm = EGA.algorithm$overall,
                                                  EGA.uni.method = EGA.uni.method$overall,
                                                  uva.cut.off = uva.cut.off, keep.org = FALSE,
                                                  silently = silently, plot = plot)

    if(!try_item_level$success){

      if(!silently){
        message("AI-GENIE reduction failed. Returning partial results.")
      }

      return(try_item_level$item_level)
    }

    item_level <- try_item_level$item_level

    # Update the returned data frame appropriately
    IDs <- item_level[["All"]][["final_items"]]$ID
    item_level[["All"]][["final_items"]] <- items[items$ID %in% IDs,]

    return(item_level[["All"]])

  }

  # Item-level analysis
  try_item_level <- run_item_reduction_pipeline(
    embedding_matrix = embeddings,
    items = items,
    EGA.model = EGA.model$type,
    EGA.algorithm = EGA.algorithm$type,
    EGA.uni.method = EGA.uni.method$type,
    uva.cut.off = uva.cut.off,
    keep.org = FALSE,  # local_GENIE doesn't need to keep original embeddings
    silently = silently,
    plot = plot
  )

  if (!try_item_level$success) {
    warning("Item-level analysis failed. Returning partial results.")
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level

  if(run.overall && length(names(item.attributes)) > 1){
  # Overall analysis
  try_overall_result <- run_pipeline_for_all(
    item_level = item_level,
    items = items,
    embeddings = embeddings,
    model = EGA.model$overall,
    algorithm = EGA.algorithm$overall,
    uni.method = EGA.uni.method$overall,
    uva.cut.off = uva.cut.off,
    keep.org = FALSE,  # local_GENIE doesn't need to keep original data
    silently = silently,
    plot = plot
  )

  if (!try_overall_result$success) {
    warning("Overall analysis failed. Returning item-level results only.")
    return(item_level)
  }

  overall_result <- try_overall_result$overall_result
  } else {
    overall_result <- item_level
    try_overall_result <- list(success = TRUE)
  }

  # Step 5: Display results summary
  if(!silently && try_overall_result$success && try_item_level$success){
    print_results(overall_result, item_level, run.overall)
  }

  # Step 6: Return comprehensive results
  # prepare return object
  return(build_return(item_level, overall_result,
                      run.overall, keep.org = FALSE))
}



#' Chat with an LLM via API Calls
#'
#' Send one or more prompts to a remote large-language model (LLM) using the
#' appropriate provider API (OpenAI, Hugging Face, Groq, or Anthropic). A valid
#' API key for at least one provider is required. To use a local model
#' (no API call), see `local_chat()`.
#'
#' @param prompts A character string or character vector. The main prompt(s)
#'   given to the model. If multiple prompts are supplied, each will be sent
#'   separately to the model.
#' @param model A character string specifying the LLM model name (e.g.,
#'   `"gpt4o"`). The model must correspond to the API key provided.
#' @param system.role A character string or character vector, default `NULL`.
#'   The system role(s) (model persona). If only one system role is provided
#'   and multiple prompts are supplied, the same role will be used for each
#'   prompt. If multiple system roles are provided, they should align with
#'   the prompts.
#' @param openai.API A character string, default `NULL`. Your OpenAI API
#'   key (required when using an OpenAI model).
#' @param hf.token A character string, default `NULL`. Your Hugging Face
#'   token (required when using a Hugging Face-hosted model).
#' @param groq.API A character string, default `NULL`. Your Groq API key
#'   (required when using a Groq-hosted model).
#' @param anthropic.API A character string, default `NULL`. Your
#'   Anthropic API key (required when using an Anthropic model).
#' @param reps Integer, default `1`. The number of times each prompt will be
#'   given to the model.
#' @param top.p Numeric, default `1`. Top-p (nucleus) sampling parameter.
#' @param temperature Numeric, default `1`. Sampling temperature controlling
#'   response randomness.
#' @param max.tokens Integer, default `2048L`. Maximum number of tokens
#'   requested from the model.
#' @param silently Logical, default `FALSE`. If `FALSE`, progress messages
#'   are printed. If `TRUE`, the function runs quietly.
#'
#' @return A `data.frame` with one row per API call (i.e., per prompt × repetition)
#'   containing:
#'   \itemize{
#'     \item `rep` — repetition index
#'     \item `prompt` — the prompt text sent to the model
#'     \item `response` — the raw text response returned by the model
#'   }
#'
#' @details
#' The function includes a retry mechanism (up to 5 attempts) for transient API
#' failures. If all attempts fail, the function stops with an informative error.
#'
#' @section Important:
#' This function requires a valid API key corresponding to the selected model.
#' Network access is required. For local (non-API) models, use `local_chat()`.
#'
#' @examples
#' \dontrun{
#' #################################################
#' ### Example 1: Writing a Very Basic Prompt  #####
#' #################################################
#'
#' # First, define your API key
#' key <- "INSERT YOUR KEY HERE"
#' # Then, define your LLM model. This model should correspond to your API key.
#' model <- "gpt4o" # in this example, you would need an OpenAI API key.
#'
#' # Then, write your prompt. This will be given to the model directly.
#' prompt <- "Why does the planet Saturn have rings? Give a 100 word explanation."
#'
#' # Optionally, add a system role (a model persona)
#' system.role <- "You specialize in tutoring astronomy for high school students."
#'
#' # Add the number of prompt repetitions. By default, this is set to 1. But it
#' # may bu useful to increase the number of repetitions to get a sense of how
#' # consistent your output might be.
#' reps <- 3
#'
#' # Now you are ready to chat with an LLM
#' first_chat <- chat(
#'   # Set your own API key. If you are not using OpenAI, change the 1st
#'   # argument to match your API key. Choices are `hf.token`, `groq.API`,
#'   # `anthropic.API`, and `openai.API`. In this example, I'm using
#'   # `openai.API` since I want to chat with a GPT model.
#'   openai.API = key,
#'   model = model, # Ensure your model corresponds to your API key.
#'   prompts = prompt,
#'   system.role = system.role,
#'   reps = reps
#' )
#'
#' # Check how the output changes from iteration to iteration
#' first_chat$response[[1]] # first iteration output
#' first_chat$response[[2]] # second iteration output
#' first_chat$response[[2]] # third iteration output
#'
#'
#' ####################################################################
#' ### Example 2: Send multiple prompts in a single function call #####
#' ####################################################################
#'
#' # You are also able to send more than one prompt in a single call
#' # Let's pull the first prompt from Example 1:
#' prompt1 <- "Why does the planet Saturn have rings? Give a 100 word explanation."
#' prompt2 <- "Which planet is the hottest in our solar system? How do we know?"
#'
#' # Aggregate the prompts in a single object
#' prompts <- c(prompt1, prompt2)
#'
#' # Ask the model the questions
#' second_chat <- chat(
#'   openai.API = key, # defined in Example 1
#'   model = model, # defined in Example 1
#'   prompts = prompts, # NEW
#'   system.role = system.role, # defined in Example 1
#'   reps = reps # defined in Example 1
#' )
#'
#' # The outputted data frame for this example will have 6 rows
#' # since the number of prompts (2) times the number of reps (3)
#' # gives a total of 6 API calls.
#' second_chat$response[second_chat$prompt==prompt1] # the responses from prompt 1
#' second_chat$response[second_chat$prompt==prompt2] # the responses from prompt 2
#'
#'
#' ####################################################################
#' ### Example 3: Send multiple prompts with different System Roles ###
#' ####################################################################
#'
#' # Perhaps your prompts are not related. In that case, you would probably want
#' # to set a different system role for each prompt.
#' # Let's change `prompt 2` to be entirely unrelated to astronomy.
#' prompt2 <- "What is the difference between eukaryotes and prokaryotes? Why?"
#'
#' # This new second prompt does not fit with the astronomy tutor persona. Let's
#' # write a persona to match this new prompt topic.
#' system.role2 <- "You specialize in tutoring biology for middle school students."
#'
#' # Now, let's combine the system roles into a single object
#' system.role <- c(system.role, # defined in Example 1: the astronomy tutor
#'                  system.role2 # defined above: the biology tutor
#'                  )
#'
#' # Aggregate our prompts in a single object again
#' prompts <- c(prompt1, # Asks about Saturn's rings (needs astronomy tutor)
#'              prompt2 # Asks about types of cells (needs biology tutor)
#'              )
#'
#' # Ask the model the questions
#' third_chat <- chat(
#'   openai.API = key, # defined in Example 1
#'   model = model, # defined in Example 1
#'   prompts = prompts, # NEW
#'   system.role = system.role, # NEW
#'   reps = reps # defined in Example 1
#' )
#'
#' # View the outputted data frame to examine the responses
#' View(third_chat)
#' }
#'
#' @seealso \code{\link{local_chat}}
#' @export
chat <- function(prompts, model,
                 system.role = NULL,
                 openai.API = NULL,
                 hf.token = NULL,
                 groq.API = NULL,
                 anthropic.API = NULL,
                 reps = 1,
                 top.p = 1,
                 temperature = 1,
                 max.tokens = 2048L,
                 silently = FALSE) {

  validation <- validate_chat_params(prompts, model,
                                     system.role,
                                     openai.API,
                                     hf.token,
                                     groq.API,
                                     anthropic.API,
                                     reps,
                                     top.p,
                                     temperature,
                                     max.tokens,
                                     silently)

  prompts <- validation$prompts
  system.role <- validation$system.role
  reps <- validation$reps
  max.tokens <- validation$max.tokens
  model <- validation$model

  ensure_aigenie_python()

  provider_info <- detect_llm_provider(
    model, groq.API, openai.API,
    anthropic.API = anthropic.API
  )

  provider <- provider_info$provider
  model <- provider_info$model

  # Preallocate list (one entry per generation)
  total_generations <- length(prompts) * reps
  results_list <- vector("list", total_generations)
  counter <- 1L

  for (i in seq_along(prompts)) {

    if (!silently) {
      cat(sprintf("Generating response(s) using prompt %d\n", i))
      cat("----------------------------------------\n")
    }

    for (j in seq_len(reps)) {

      if (!silently) {
        cat(sprintf("Generating response %d of %d... ", j, reps))
      }

      max_attempts <- 5L
      wait_seconds <- 3
      attempt <- 1L
      success <- FALSE
      last_error_msg <- NULL
      raw_text <- NULL

      while (attempt <= max_attempts && !success) {

        res <- tryCatch({
          generate_text_llm(
            prompt = prompts[[i]],
            system.role = system.role[[i]],
            model = model,
            temperature = temperature,
            top.p = top.p,
            max_tokens = max.tokens,
            openai.API = openai.API,
            groq.API = groq.API,
            anthropic.API = anthropic.API
          )
        }, error = function(e) {
          structure(
            list(message = conditionMessage(e)),
            class = "llm_error"
          )
        })

        if (inherits(res, "llm_error")) {

          last_error_msg <- res$message

          if (!silently && attempt < max_attempts) {
            cat(sprintf(
              "\nAttempt %d failed: %s. Retrying in %ds...\n",
              attempt, last_error_msg, wait_seconds
            ))
          }

          attempt <- attempt + 1L
          if (attempt <= max_attempts) Sys.sleep(wait_seconds)

        } else {
          raw_text <- res
          success <- TRUE
        }
      }

      if (!success) {
        stop(
          sprintf(
            "API call failed after %d attempts.\nLast error: %s",
            max_attempts,
            ifelse(is.null(last_error_msg), "unknown error", last_error_msg)
          ),
          call. = FALSE
        )
      }

      # Store result as one-row data frame
      results_list[[counter]] <- data.frame(
        rep = j,
        prompt = prompts[[i]],
        response = raw_text,
        stringsAsFactors = FALSE
      )

      counter <- counter + 1L

      if (!silently) cat("Done.\n")
    }

    if (!silently) cat("\n")
  }

  # Combine once
  results_df <- do.call(rbind, results_list)

  return(results_df)
}




#' Chat with a local LLM (no API calls)
#'
#' Send one or more prompts to a locally available large-language model (LLM)
#' without making remote API calls. The local model must be installed/available
#' on the machine as a local model directory. This function is intended for
#' fully local inference (no API key required).
#'
#' @param prompts A character string or character vector. The main prompt(s)
#'   given to the model. If multiple prompts are supplied, each will be sent
#'   separately to the model.
#' @param model.path A character string. Path for the local model file.
#'    The function does not download models; ensure the model is present locally
#'    before using this function.
#' @param n.ctx Integer, default `4096`. The context window (number of tokens)
#'   available to the model for a single generation.
#' @param n.gpu.layers Integer, default `-1`. Number of model layers to place on
#'   GPU (if supported). Use `-1` to let the runtime choose automatically.
#' @param max.tokens Integer, default `1024L`. Maximum number of tokens requested
#'   from the local model for a single generation.
#' @param system.role A character string or character vector, default `NULL`.
#'   The system role(s) (model persona). If only one system role is provided
#'   and multiple prompts are supplied, the same role will be used for each
#'   prompt. If multiple system roles are provided, they should align with
#'   the prompts.
#' @param reps Integer, default `1`. The number of times each prompt will be
#'   given to the model (independent generations).
#' @param temperature Numeric, default `1`. Sampling temperature controlling
#'   response randomness.
#' @param top.p Numeric, default `1`. Top-p (nucleus) sampling parameter.
#' @param silently Logical, default `FALSE`. If `FALSE`, progress messages
#'   are printed to the console. If `TRUE`, the function runs quietly.
#'
#' @return A `data.frame` with one row per generation (i.e., per prompt × repetition)
#'   containing:
#'   \itemize{
#'     \item `rep` — repetition index
#'     \item `prompt` — the prompt text sent to the model
#'     \item `response` — the raw text response returned by the model
#'   }
#'
#' @details
#' Before running this function check that you are able to run the local
#' environment via `check_local_llm_setup()`.
#'
#' For each prompt × repetition the function constructs a `full_prompt` that
#' includes the `system.role` and the user prompt, sets a deterministic
#' generation seed per generation, and calls the local model. A retry loop
#' (up to 5 attempts, with brief waits) handles transient failures; if all
#' attempts fail the function aborts with an informative error.
#'
#' @section Important / Warnings:
#' * Local model inference can be resource intensive. Large models may require
#'   substantial disk space, RAM, and (optionally) GPU support. Performance and
#'   feasibility depend on model size and hardware.
#' * `model.path` must point to a model already present; this function will not
#'   download remote models.
#'
#' @examples
#' \dontrun{
#' #################################################
#' ### Example 1: Writing a Very Basic Prompt  #####
#' #################################################
#'
#' # For local_chat you do NOT need an API key, but you DO need a text generation
#' # model available locally.
#' model <- "path/to/local-model" # replace with your local model path
#'
#' # Then, write your prompt. This will be given to the model directly.
#' prompt <- "Why does the planet Saturn have rings? Give a 100 word explanation."
#'
#' # Optionally, add a system role (a model persona)
#' system.role <- "You specialize in tutoring astronomy for high school students."
#'
#' # Add the number of prompt repetitions. By default, this is set to 1. But it
#' # may be useful to increase the number of repetitions to get a sense of how
#' # consistent your output might be.
#' reps <- 3
#'
#' # Now you are ready to chat with the local model
#' first_chat <- local_chat(
#'   model.path = model, # local model identifier or path
#'   prompts = prompt,
#'   system.role = system.role,
#'   reps = reps
#' )
#'
#' # Check how the output changes from iteration to iteration
#' first_chat$response[[1]] # first iteration output
#' first_chat$response[[2]] # second iteration output
#' first_chat$response[[3]] # third iteration output
#'
#'
#' ####################################################################
#' ### Example 2: Send multiple prompts in a single function call #####
#' ####################################################################
#'
#' # You are able to send more than one prompt in a single call
#' prompt1 <- "Why does the planet Saturn have rings? Give a 100 word explanation."
#' prompt2 <- "Which planet is the hottest in our solar system? How do we know?"
#'
#' # Aggregate the prompts in a single object
#' prompts <- c(prompt1, prompt2)
#'
#' # Ask the model the questions
#' second_chat <- local_chat(
#'   model.path = model, # defined above
#'   prompts = prompts, # NEW
#'   system.role = system.role, # defined above
#'   reps = reps # defined above
#' )
#'
#' # The outputted data frame for this example will have 6 rows
#' # since the number of prompts (2) times the number of reps (3)
#' # gives a total of 6 generations.
#' second_chat$response[second_chat$prompt == prompt1] # the responses from prompt 1
#' second_chat$response[second_chat$prompt == prompt2] # the responses from prompt 2
#'
#'
#' ####################################################################
#' ### Example 3: Send multiple prompts with different System Roles ###
#' ####################################################################
#'
#' # Perhaps your prompts are not related. In that case, you would probably want
#' # to set a different system role for each prompt.
#' prompt2 <- "What is the difference between eukaryotes and prokaryotes? Why?"
#'
#' # This new second prompt does not fit with the astronomy tutor persona. Let's
#' # write a persona to match this new prompt topic.
#' system.role2 <- "You specialize in tutoring biology for middle school students."
#'
#' # Now, let's combine the system roles into a single object
#' system.role <- c(system.role, # defined earlier: the astronomy tutor
#'                  system.role2 # defined above: the biology tutor
#'                  )
#'
#' # Aggregate our prompts in a single object again
#' prompts <- c(prompt1, # Asks about Saturn's rings (needs astronomy tutor)
#'              prompt2 # Asks about types of cells (needs biology tutor)
#'              )
#'
#' # Ask the model the questions
#' third_chat <- local_chat(
#'   model.path = model,
#'   prompts = prompts,
#'   system.role = system.role,
#'   reps = reps
#' )
#'
#' # View the outputted data frame to examine the responses
#' View(third_chat)
#' }
#'
#' @seealso \code{\link{chat}}
#' @export
local_chat <- function(prompts, model.path,
                       n.ctx = 4096,
                       n.gpu.layers = -1,
                       max.tokens = 1024,
                       system.role = NULL,
                       reps = 1,
                       temperature = 1,
                       top.p = 1,
                       silently = FALSE) {

  validation <- validate_local_chat_params(prompts, model.path,
                                           n.ctx,
                                           n.gpu.layers,
                                           max.tokens,
                                           system.role,
                                           reps,
                                           temperature,
                                           top.p,
                                           silently)

  prompts <- validation$prompts
  system.role <- validation$system.role
  reps <- validation$reps
  max.tokens <- validation$max.tokens
  model.path <- validation$model.path

  setup_ok <- check_local_llm_setup(model.path, silently)
  if (!setup_ok) {
    stop("Local setup incomplete. Please run check_local_llm_setup() for details.")
  }

  ensure_llama_cpp_python(silently = silently)

  # ---- Load model ----
  llama_cpp <- tryCatch(
    reticulate::import("llama_cpp"),
    error = function(e) {
      stop("Failed to import llama_cpp: ", conditionMessage(e), call. = FALSE)
    }
  )

  if (!silently) cat("Loading local model...\n")

  llm <- tryCatch(
    llama_cpp$Llama(
      model_path = model.path,
      n_ctx = as.integer(n.ctx),
      n_gpu_layers = as.integer(n.gpu.layers),
      seed = -1L,  # generation-level seed controls determinism
      verbose = FALSE
    ),
    error = function(e) {
      stop("Failed to load local model: ", conditionMessage(e), call. = FALSE)
    }
  )

  if (!silently) cat("Model loaded successfully.\n\n")

  # ---- Preallocate result list (one entry per generation) ----
  total_generations <- length(prompts) * reps
  results_list <- vector("list", total_generations)
  counter <- 1L

  # ---- Main loops ----
  for (i in seq_along(prompts)) {

    if (!silently) {
      cat(sprintf("Generating response(s) using prompt %d\n", i))
      cat("----------------------------------------\n")
    }

    for (j in seq_len(reps)) {

      if (!silently) {
        cat(sprintf("Generating response %d of %d... ", j, reps))
      }

      max_attempts <- 5L
      wait_seconds <- 3
      attempt <- 1L
      success <- FALSE
      last_error_msg <- NULL
      raw_text <- NULL

      while (attempt <= max_attempts && !success) {

        full_prompt <- paste0(
          "System: ", system.role[[i]], "\n\n",
          "User: ", prompts[[i]], "\n\n",
          "Assistant:"
        )

        generation_seed <- as.integer(
          (123L + i * 1000L + j) %% .Machine$integer.max
        )

        # Shared arguments minus the token parameter (resolved below)
        base_args <- list(
          prompt      = full_prompt,
          temperature = temperature,
          top_p       = top.p,
          seed        = generation_seed,
          echo        = FALSE,
          stop        = list("User:", "System:")
        )

        # Generate — retry with max_completion_tokens if model rejects max_tokens
        raw_text <- tryCatch({
          response <- do.call(llm, c(base_args,
                                     list(max_tokens = as.integer(max.tokens))))
          response[["choices"]][[1]][["text"]]
        }, error = function(e) {
          if (grepl("max_tokens", conditionMessage(e), fixed = TRUE) &&
              grepl("max_completion_tokens", conditionMessage(e), fixed = TRUE)) {
            if (!silently) cat("Retrying with 'max_completion_tokens' parameter...\n")
            tryCatch({
              response <- do.call(llm, c(base_args,
                                         list(max_completion_tokens = as.integer(max.tokens))))
              response[["choices"]][[1]][["text"]]
            }, error = function(e2) {
              structure(
                list(message = conditionMessage(e2)),
                class = "llm_error"
              )
            })
          } else {
            structure(
              list(message = conditionMessage(e)),
              class = "llm_error"
            )
          }
        })

        if (inherits(raw_text, "llm_error")) {

          last_error_msg <- raw_text$message

          if (!silently && attempt < max_attempts) {
            cat(sprintf(
              "\nAttempt %d failed: %s. Retrying in %ds...\n",
              attempt, last_error_msg, wait_seconds
            ))
          }

          attempt <- attempt + 1L
          if (attempt <= max_attempts) Sys.sleep(wait_seconds)

        } else {
          success <- TRUE
        }
      }

      if (!success) {
        stop(
          sprintf(
            "Generation failed after %d attempts.\nLast error: %s",
            max_attempts,
            ifelse(is.null(last_error_msg), "unknown error", last_error_msg)
          ),
          call. = FALSE
        )
      }

      # Store result as one-row data frame
      results_list[[counter]] <- data.frame(
        rep = j,
        prompt = prompts[[i]],
        response = raw_text,
        stringsAsFactors = FALSE
      )

      counter <- counter + 1L

      if (!silently) cat("Done.\n")
    }

    if (!silently) cat("\n")
  }

  # ---- Combine once ----
  results_df <- do.call(rbind, results_list)

  return(results_df)
}



