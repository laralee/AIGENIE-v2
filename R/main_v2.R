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
#' The return object varies based on parameter settings:
#'
#' **When `items.only = TRUE`:** Returns a data frame with columns `ID`, `statement`,
#' `type`, and `attribute` containing the generated items.
#'
#' **When `embeddings.only = TRUE`:** Returns a named list with two elements:
#' `embeddings` (the embedding matrix with column names corresponding to item IDs) and
#' `items` (the items data frame described above).
#'
#' **Default behavior (`items.only = FALSE`, `embeddings.only = FALSE`):** Returns a
#' complex list with two primary components:
#'
#' \describe{
#'   \item{`overall`}{Results for all items considered agnostic of item type, containing:
#'     \describe{
#'       \item{`final_NMI`}{Final normalized mutual information (NMI) after all reduction steps}
#'       \item{`initial_NMI`}{Initial NMI of the pre-reduced, full item pool}
#'       \item{`embeddings`}{List with `full` (full embedding matrix of reduced items),
#'         `sparse` (sparsified embeddings of reduced items), `selected` (string specifying
#'         which embeddings were used). If `keep.org = TRUE`, also includes `full_org`
#'         and `sparse_org` for complete pre-reduction item set}
#'       \item{`EGA.model_selected`}{EGA model used for analysis (TMFG or Glasso)}
#'       \item{`final_items`}{Data frame with final items after reduction (columns: ID,
#'         statement, attribute, type, EGA_com)}
#'       \item{`final_EGA`}{Final EGA object from EGAnet package post-reduction}
#'       \item{`initial_EGA`}{Initial EGA object of all items in original pool}
#'       \item{`start_N`}{Initial number of items pre-reduction}
#'       \item{`final_N`}{Final number of items in reduced pool}
#'       \item{`network_plot`}{ggplot/patchwork comparison plot showing EGA network before vs after reduction}
#'     }
#'   }
#'   \item{`item_type_level`}{Named list where results are displayed at the item type level. Each element contains results
#'     for items of only one type:
#'     \describe{
#'       \item{`final_NMI`, `initial_NMI`, `embeddings`, `EGA.model_selected`, `final_items`,
#'         `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`}{Same as overall object}
#'       \item{`UVA`}{List with `n_removed` (number of redundant items removed by the Unique Variable Analysis (UVA) step), `n_sweeps`
#'         (number of UVA iterations), `redundant_pairs` (data frame with sweep, items, keep, remove columns)}
#'       \item{`bootEGA`}{List with `initial_boot` (first bootEGA object), `final_boot`
#'         (final bootEGA object with all stable items), `n_removed` (number of unstable items),
#'         `items_removed` (data frame of removed items with boot run information),
#'         `initial_boot_with_redundancies` (boot EGA object for original item pool)}
#'       \item{`stability_plot`}{ggplot/patchwork stability plot showing item stability before vs after reduction}
#'     }
#'   }
#' }
#'
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
#'   neuroticism = "Neuroticism is a personality trait that describes one's tendency to experience negative emotions like anxiety, depression, irritability, anger, and self-consciousness.",
#'   openness = "Openness is a personality trait that describes how open-minded, creative, and imaginative a person is.",
#'   extraversion = "Extraversion is a personality trait that describes people who are more focused on the external world than their internal experience."
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
#' system.role <- "You are an expert methodologist who specializes in scale development for personality measurement. You are especially equipped to create novel personality items that mimic the style of popular 'Big Five' assessments."
#'
#' # Define custom prompts for each personality trait
#' custom.personality.prompts <- list(
#'
#'   # Prompt for generating neuroticism traits
#'   neuroticism = paste0(
#'     "Generate unique, psychometrically robust single-statement items designed to assess ",
#'     "the Big Five personality trait neuroticism.",
#'     "Neuroticism has the following characteristics: anxious, depressed, insecure, and emotional. "
#'   ),
#'
#'   # Prompt for generating openness traits
#'   openness = paste0(
#'     "Generate unique, psychometrically robust single-statement items designed to assess ",
#'     "the Big Five personality trait openness.",
#'     "Openness has the following characteristics: creative, perceptual, curious, and philosophical"
#'   ),
#'
#'   # Prompt for generating extraversion traits
#'   extraversion = paste0(
#'     "Generate unique, psychometrically robust single-statement items designed to assess ",
#'     "the Big Five personality trait extraversion.",
#'     "Extraversion has the following characteristics: friendly, positive, assertive, and energetic."
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
                       EGA.model = NULL, EGA.algorithm = NULL, EGA.uni.method = "louvain",

                       # Flags
                       keep.org = FALSE, items.only = FALSE, embeddings.only = FALSE,
                       adaptive = TRUE, plot = TRUE, silently = FALSE
                       ){


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
                                            embeddings.only, adaptive, plot,
                                            silently)


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


  # Generate item level results
  try_item_level <- run_item_reduction_pipeline(embedding_matrix = embeddings,
                    items=items, EGA.model = EGA.model, EGA.algorithm = EGA.algorithm,
                    EGA.uni.method = EGA.uni.method, keep.org = keep.org,
                    silently = silently, plot = plot)

  if(!try_item_level$success){
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level

  if(length(names(item.attributes)) > 1) { # only run overall if you have to
    # If successful, generate results for items overall
    try_overall_result <- run_pipeline_for_all(item_level = item_level, items = items,
                            embeddings = embeddings, EGA.model = EGA.model,
                            EGA.algorithm = EGA.algorithm, EGA.uni.method = EGA.uni.method,
                            keep.org = keep.org, silently = silently, plot = plot)

    if(!try_overall_result$success){
      return(item_level)
    }

    overall_result <- try_overall_result$overall_result
    only.one <- FALSE
  } else {
    overall_result <- item_level
    try_overall_result <- list(success = TRUE)
    only.one <- TRUE
  }


  if(!silently && try_overall_result$success && try_item_level$success){
    print_results(overall_result, item_level, only.one)
  }

  return( list(overall = overall_result,
               item_type_level = item_level))



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
#' @param plot Display network plots (default: TRUE)
#' @param silently Suppress progress messages (default: FALSE)
#'
#' **Default behavior (`items.only = FALSE`, `embeddings.only = FALSE`):** Returns a
#' complex list with two primary components:
#'
#' \describe{
#'   \item{`overall`}{Results for all items considered agnostic of item type, containing:
#'     \describe{
#'       \item{`final_NMI`}{Final normalized mutual information after all reduction steps}
#'       \item{`initial_NMI`}{Initial NMI of the pre-reduced, full item pool}
#'       \item{`embeddings`}{List with `full` (full embedding matrix of reduced items),
#'         `sparse` (sparsified embeddings of reduced items), `selected` (string specifying
#'         which embeddings were used). If `keep.org = TRUE`, also includes `full_org`
#'         and `sparse_org` for complete pre-reduction item set}
#'       \item{`EGA.model_selected`}{EGA model used for analysis (TMFG or Glasso)}
#'       \item{`final_items`}{Data frame with final items after reduction (columns: ID,
#'         statement, attribute, type, EGA_com)}
#'       \item{`final_EGA`}{Final EGA object from EGAnet package post-reduction}
#'       \item{`initial_EGA`}{Initial EGA object of all items in original pool}
#'       \item{`start_N`}{Initial number of items pre-reduction}
#'       \item{`final_N`}{Final number of items in reduced pool}
#'       \item{`network_plot`}{ggplot/patchwork comparison plot showing EGA network before vs after reduction}
#'     }
#'   }
#'   \item{`item_type_level`}{Named list where results are displayed at the item type level. Each element contains results
#'     for items of only one type:
#'     \describe{
#'       \item{`final_NMI`, `initial_NMI`, `embeddings`, `EGA.model_selected`, `final_items`,
#'         `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`}{Same as overall object}
#'       \item{`UVA`}{List with `n_removed` (number of redundant items removed), `n_sweeps`
#'         (number of UVA iterations), `redundant_pairs` (data frame with sweep, items, keep, remove columns)}
#'       \item{`bootEGA`}{List with `initial_boot` (first bootEGA object), `final_boot`
#'         (final bootEGA object with all stable items), `n_removed` (number of unstable items),
#'         `items_removed` (data frame of removed items with boot run information),
#'         `initial_boot_with_redundancies` (boot EGA object for original item pool)}
#'       \item{`stability_plot`}{ggplot/patchwork stability plot showing item stability before vs after reduction}
#'     }
#'   }
#' }
#'
#' @examples
#' \dontrun{
#' ########################################################
#' #### Running AIGENIE with a downloaded LLM model ######
#' ########################################################
#'
#' # Item type definitions
#' trait.definitions <- list(
#'  neuroticism = "Neuroticism is a personality trait that describes one's tendency to experience negative emotions like anxiety, depression, irritability, anger, and self-consciousness.",
#'  extraversion = "Extraversion is a personality trait that describes people who are more focused on the external world than their internal experience."
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
#' scale.title <- "Three of 'Big Five:' A Streamlined Personality Inventory"
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
  EGA.uni.method = "louvain",

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
  plot = TRUE,
  silently = FALSE
) {

  # Step 1: Validate all inputs
  validation <- validate_user_input_local_AIGENIE(
    item.attributes, model.path, embedding.model,
    main.prompts, temperature, top.p, target.N,
    domain, scale.title, item.examples, audience,
    item.type.definitions, response.options, prompt.notes,
    system.role, EGA.model, EGA.algorithm, EGA.uni.method,
    n.ctx, n.gpu.layers, max.tokens,
    device, batch.size, pooling.strategy, max.length,
    keep.org, items.only, embeddings.only, adaptive, plot, silently
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

  # Item-level reduction
  try_item_level <- run_item_reduction_pipeline(
    embedding_matrix = embeddings, items=items,
    EGA.model = EGA.model, EGA.algorithm = EGA.algorithm,
    EGA.uni.method = EGA.uni.method, keep.org = keep.org, silently = silently,
    plot = plot
  )

  if (!try_item_level$success) {
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level

  if(length(names(item.attributes)) > 1){
  # Overall reduction
    try_overall_result <- run_pipeline_for_all(
      item_level = item_level, items = items, embeddings = embeddings,
      EGA.model = EGA.model, EGA.algorithm = EGA.algorithm,
      EGA.uni.method = EGA.uni.method, keep.org = keep.org,
      silently = silently, plot = plot
    )

    if (!try_overall_result$success) {
      return(item_level)
    }

    overall_result <- try_overall_result$overall_result
    only.one <- FALSE
  } else {
    overall_result <- item_level
    try_overall_result <- list(success = TRUE)
    only.one <- TRUE
  }

  # Step 7: Print results summary
  if(!silently && try_overall_result$success && try_item_level$success){
    print_results(overall_result, item_level, only.one)
  }

  # Return results
  return(list(
    overall = overall_result,
    item_type_level = item_level
  ))
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
#' @param groq.API Groq API key (currently unused in GENIE)
#' @param jina.API Jina AI API key for using Jina embedding models (e.g., "jina-embeddings-v3").
#'   Free tier available at \url{https://jina.ai/}.
#' @param model Language model identifier (currently unused in GENIE)
#' @param temperature LLM temperature (currently unused in GENIE)
#' @param top.p LLM top-p parameter (currently unused in GENIE)
#' @param embedding.model Embedding model to use if embedding.matrix not provided:
#'   \itemize{
#'     \item OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
#'     \item Jina AI: "jina-embeddings-v3", "jina-embeddings-v4", "jina-embeddings-v2-base-en" (requires jina.API)
#'     \item HuggingFace: "BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"
#'   }
#' @param EGA.model EGA network estimation model ("glasso", "TMFG", or NULL for auto-selection)
#' @param EGA.algorithm EGA community detection algorithm ("walktrap", "leiden", "louvain")
#' @param EGA.uni.method Unidimensionality assessment method ("louvain", "expand", "LE")
#' @param embeddings.only If `TRUE`, return embeddings and stop (skip network analysis)
#' @param plot If `TRUE`, display network comparison plots
#' @param silently If `TRUE`, suppress progress messages
#'
#' **Default behavior (`embeddings.only = FALSE`):** Returns a
#' complex list with two primary components:
#'
#' \describe{
#'   \item{`overall`}{Results for all items considered agnostic of item type, containing:
#'     \describe{
#'       \item{`final_NMI`}{Final normalized mutual information after all reduction steps}
#'       \item{`initial_NMI`}{Initial NMI of the pre-reduced, full item pool}
#'       \item{`embeddings`}{List with `full` (full embedding matrix of reduced items),
#'         `sparse` (sparsified embeddings of reduced items), `selected` (string specifying
#'         which embeddings were used). If `keep.org = TRUE`, also includes `full_org`
#'         and `sparse_org` for complete pre-reduction item set}
#'       \item{`EGA.model_selected`}{EGA model used for analysis (TMFG or Glasso)}
#'       \item{`final_items`}{Data frame with final items after reduction (columns: ID,
#'         statement, attribute, type, EGA_com)}
#'       \item{`final_EGA`}{Final EGA object from EGAnet package post-reduction}
#'       \item{`initial_EGA`}{Initial EGA object of all items in original pool}
#'       \item{`start_N`}{Initial number of items pre-reduction}
#'       \item{`final_N`}{Final number of items in reduced pool}
#'       \item{`network_plot`}{ggplot/patchwork comparison plot showing EGA network before vs after reduction}
#'     }
#'   }
#'   \item{`item_type_level`}{Named list where results are displayed at the item type level. Each element contains results
#'     for items of only one type:
#'     \describe{
#'       \item{`final_NMI`, `initial_NMI`, `embeddings`, `EGA.model_selected`, `final_items`,
#'         `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`}{Same as overall object}
#'       \item{`UVA`}{List with `n_removed` (number of redundant items removed), `n_sweeps`
#'         (number of UVA iterations), `redundant_pairs` (data frame with sweep, items, keep, remove columns)}
#'       \item{`bootEGA`}{List with `initial_boot` (first bootEGA object), `final_boot`
#'         (final bootEGA object with all stable items), `n_removed` (number of unstable items),
#'         `items_removed` (data frame of removed items with boot run information),
#'         `initial_boot_with_redundancies` (boot EGA object for original item pool)}
#'       \item{`stability_plot`}{ggplot/patchwork stability plot showing item stability before vs after reduction}
#'     }
#'   }
#' }
#'
#' @examples
#'  \dontrun{
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
#'   "I actively seek to include others in group activities, making them feel part of the team.",
#'   "I frequently reach out to new acquaintances to foster connections and friendships.",
#'   "I habitually focus on the silver lining in difficult situations, maintaining an optimistic perspective.",
#'   "I often express gratitude for the positive aspects of my life, which enhances my overall mood.",
#'   "I find joy in taking on new challenges that require a burst of energy and enthusiasm.",
#'   "I thrive in dynamic environments that keep me on my toes and invigorate my spirit.",
#'   "I take pleasure in introducing people to one another, acting as a social connector.",
#'   "I enjoy making others comfortable by engaging them in light-hearted conversation.",
#'   "I often set a positive tone in group settings with my upbeat demeanor.",
#'  "I approach each day with a sense of excitement and a positive mindset.",
#'  "I am drawn to fast-paced environments where I can express my high energy levels.",
#'  "I feel invigorated when working on multiple projects that demand my full attention.",
#'  "I take delight in meeting new people and quickly making them feel at ease.",
#'  "I find it rewarding to help shy or reserved individuals become involved in group discussions.",
#'  "I have a natural tendency to uplift others with my positive remarks and outlook.",
#'  "I find happiness in highlighting the successes of others, contributing to a cheerful environment.",
#'  "I eagerly immerse myself in activities that demand stamina and sustained energy.",
#'  "I often channel my vitality into hobbies and sports that require physical exertion.",
#'  "I feel rejuvenated when I bring people together to collaborate and share ideas.",
#'  "I often extend a genuine greeting to others, creating an inviting atmosphere.",
#'  "I regularly see challenges as opportunities for growth and learning.",
#'  "I commonly radiate positivity, influencing the mood of those around me.",
#'  "I approach mornings with anticipation and vigor, ready to embrace the day.",
#'  "I consistently infuse enthusiasm into group activities, boosting collective energy levels.",
#'  "I make an effort to connect with people by remembering details about their lives.",
#'  "I genuinely enjoy learning about people's diverse experiences and viewpoints.",
#'  "I have a habit of encouraging others to see the bright side of their situations.",
#'  "I believe in celebrating small victories, finding joy in daily accomplishments.",
#'  "I often find myself eager to start the day with ambitious plans and goals.",
#'  "I am known for sustaining high levels of energy during extended work sessions or projects.",
#'  "I make an effort to engage those around me in meaningful and enjoyable conversations.",
#'  "I often seek opportunities to bring people together, fostering a sense of community.",
#'  "I naturally inspire others with my optimistic outlook, even in uncertain times.",
#'  "I frequently look for the positive aspects in challenging situations and share them with others.",
#'  "I approach new experiences with an eagerness and fervor that motivates those around me.",
#'  "I thrive on maintaining high energy levels throughout demanding and fast-paced days.",
#'  "I take pleasure in initiating warm interactions in group settings to make everyone comfortable.",
#'  "I enjoy hosting gatherings that connect friends and encourage social bonding.",
#'  "I am skilled at turning setbacks into learning experiences to maintain a positive outlook.",
#'  "I always try to highlight the benefits in situations, enhancing a cheerful atmosphere.",
#'  "I find excitement in starting the day with a list of activities to energize my routine.",
#'  "I relish the challenge of keeping up with dynamic schedules that require sustained energy.",
#'  "I often find joy in making newcomers feel welcome and appreciated in group settings.",
#'  "I genuinely enjoy striking up conversations to learn more about the people I encounter.",
#'  "I have a knack for seeing potential in situations that others might overlook.",
#'  "I consistently try to uplift the mood in my surroundings with hopeful and encouraging words.",
#'  "I frequently harness my energy to inspire and motivate those around me in team environments.",
#'  "I often feel invigorated by challenges that require sustained focus and dynamic thinking.",
#'  "I often create environments where people feel encouraged to share their thoughts freely.",
#'  "I find it fulfilling to engage deeply with people, building lasting connections.",
#'  "I see potential in every day, believing it holds opportunities for something good.",
#'  "I actively focus on the pleasures of life, which naturally enhances my mood.",
#'  "I am invigorated by opportunities to engage in lively and spirited events.",
#'  "I tend to maintain momentum throughout the day, sustaining my energy levels.",
#'  "I frequently experience sudden shifts in my emotions even when there is no apparent reason.",
#'  "People often find it difficult to predict my emotional reactions to different situations.",
#'  "I often doubt my abilities and worry about whether I am meeting expectations.",
#'  "I frequently question my self-worth and tend to seek reassurance from others.",
#'  "I become annoyed easily over small inconveniences or delays.",
#'  "I often find myself feeling agitated or frustrated in situations that don't bother most people.",
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
#'  "My emotional state is often unpredictable, shifting from contentment to sadness with little warning.",
#'  "People have commented that my emotions seem to fluctuate more than those of others.",
#'  "I frequently feel self-conscious about my achievements compared to those of my peers.",
#'  "I often worry excessively about making mistakes, even in situations where it might be inconsequential.",
#'  "Small disruptions in my daily routine can trigger strong feelings of annoyance.",
#'  "I find myself becoming irritated more quickly than others when under stress or pressure.",
#'  "I often find my emotional responses to be unpredictable, feeling fine one moment and unsettled the next.",
#'  "I experience strong emotions that can shift unexpectedly, often catching me off guard.",
#'  "I regularly feel uncertain about my ability to manage new responsibilities effectively.",
#'  "I often question my decisions, fearing they might not lead to the best outcomes.",
#'  "I frequently find myself reacting with impatience to situations perceived as minor interruptions.",
#'  "Even minor provocations can sometimes lead to an exaggerated sense of annoyance for me.",
#'  "My emotional state is often inconsistent, and I can feel ecstatic or despondent within short timeframes.",
#'  "I notice that my feelings can be quite volatile and intense, affecting how I interact with others throughout the day.",
#'  "I regularly doubt whether I am capable of achieving my personal or professional goals.",
#'  "I often seek validation from others to feel reassured about my self-worth.",
#'  "I am sensitive to disturbances and find my patience wearing thin quickly when things aren't orderly.",
#'  "I occasionally struggle to contain my annoyance over trivial issues that disrupt my sense of calm.",
#'  "I can go from feeling upbeat to being downcast without an obvious cause.",
#'  "My emotional responses can sometimes be unpredictable, shifting with little notice.",
#'  "I often feel the need for affirmation about my abilities from friends or colleagues.",
#'  "I tend to compare myself to others and feel uncertain about my achievements.",
#'  "I find myself easily bothered by noises or disturbances in my environment.",
#'  "I get easily flustered by situations that interrupt my planned activities.",
#'  "I find it challenging to maintain a consistent emotional state, regardless of external situations.",
#'  "My emotional reactions can be intense and differ significantly from moment to moment.",
#'  "I have a persistent fear of not measuring up to the expectations placed on me.",
#'  "I often feel anxious about others' perceptions of my capabilities and appearance.",
#'  "I am quick to express frustration at minor inconveniences in my daily routine.",
#'  "I find that small, unforeseen events often disrupt my sense of calm and lead to irritation.",
#'  "My emotional reactions can be strong and relentless, impacting my behavior throughout the day.",
#'  "I often find myself emotionally labile, with an inner turbulence that others rarely perceive.",
#'  "I frequently worry about my competence in areas where others seem confident.",
#'  "I have a tendency to second-guess myself and require affirmation to feel reassured about my choices.",
#'  "Small disruptions can ignite a lingering sense of agitation within me.",
#'  "I often catch myself feeling irritable even in relatively calm settings.",
#'  "I find myself swinging from happy to melancholic in a short span of time, often surprising even myself.",
#'  "Others often comment on how quickly my mood can change in response to seemingly minor events.",
#'  "I tend to feel apprehensive about presenting my opinions, fearing they may be judged harshly.",
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
    groq.API = NULL,                         # Unused but kept for consistency
    jina.API = NULL,

    # LLM parameters (unused but kept for consistency)
    model = "gpt4o",
    temperature = 1,
    top.p = 1,

    # Embedding parameters
    embedding.model = "text-embedding-3-small",

    # EGA parameters
    EGA.model = NULL,
    EGA.algorithm = NULL,
    EGA.uni.method = "louvain",

    # Control flags
    embeddings.only = FALSE,
    plot = TRUE,
    silently = FALSE
) {

  # Step 1: Comprehensive input validation
  validation <- validate_user_input_GENIE(
    items = items,
    embedding.matrix = embedding.matrix,
    openai.API = openai.API,
    hf.token = hf.token,
    groq.API = groq.API,
    jina.API = jina.API,
    model = model,
    temperature = temperature,
    top.p = top.p,
    embedding.model = embedding.model,
    EGA.model = EGA.model,
    EGA.algorithm = EGA.algorithm,
    EGA.uni.method = EGA.uni.method,
    embeddings.only = embeddings.only,
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

  # Item-level analysis
  try_item_level <- run_item_reduction_pipeline(
    embedding_matrix = embeddings,
    items = items,
    EGA.model = EGA.model,
    EGA.algorithm = EGA.algorithm,
    EGA.uni.method = EGA.uni.method,
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
  if(length(names(item.attributes)) > 1){
  try_overall_result <- run_pipeline_for_all(
    item_level = item_level,
    items = items,
    embeddings = embeddings,
    model = EGA.model,
    algorithm = EGA.algorithm,
    uni.method = EGA.uni.method,
    keep.org = FALSE,  # GENIE doesn't need to keep original data
    silently = silently,
    plot = plot
  )

  if (!try_overall_result$success) {
    warning("Overall analysis failed. Returning item-level results only.")
    return(item_level)
  }

  overall_result <- try_overall_result$overall_result
  only.one <- FALSE
  } else {
    overall_result <- item_level
    try_overall_result <- list(success = TRUE)
    only.one <- TRUE
  }

  # Step 5: Display results summary
  if(!silently && try_overall_result$success && try_item_level$success){
    print_results(overall_result, item_level, only.one)
  }

  # Step 6: Return comprehensive results
  return(list(
    overall = overall_result,
    item_type_level = item_level
  ))
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
#'
#' @param embeddings.only If `TRUE`, return embeddings and stop (skip network analysis)
#' @param plot If `TRUE`, display network comparison plots
#' @param silently If `TRUE`, suppress progress messages
#'
#' **Default behavior (`embeddings.only = FALSE`):** Returns a
#' complex list with two primary components:
#'
#' \describe{
#'   \item{`overall`}{Results for all items considered agnostic of item type, containing:
#'     \describe{
#'       \item{`final_NMI`}{Final normalized mutual information after all reduction steps}
#'       \item{`initial_NMI`}{Initial NMI of the pre-reduced, full item pool}
#'       \item{`embeddings`}{List with `full` (full embedding matrix of reduced items),
#'         `sparse` (sparsified embeddings of reduced items), `selected` (string specifying
#'         which embeddings were used). If `keep.org = TRUE`, also includes `full_org`
#'         and `sparse_org` for complete pre-reduction item set}
#'       \item{`EGA.model_selected`}{EGA model used for analysis (TMFG or Glasso)}
#'       \item{`final_items`}{Data frame with final items after reduction (columns: ID,
#'         statement, attribute, type, EGA_com)}
#'       \item{`final_EGA`}{Final EGA object from EGAnet package post-reduction}
#'       \item{`initial_EGA`}{Initial EGA object of all items in original pool}
#'       \item{`start_N`}{Initial number of items pre-reduction}
#'       \item{`final_N`}{Final number of items in reduced pool}
#'       \item{`network_plot`}{ggplot/patchwork comparison plot showing EGA network before vs after reduction}
#'     }
#'   }
#'   \item{`item_type_level`}{Named list where results are displayed at the item type level. Each element contains results
#'     for items of only one type:
#'     \describe{
#'       \item{`final_NMI`, `initial_NMI`, `embeddings`, `EGA.model_selected`, `final_items`,
#'         `final_EGA`, `initial_EGA`, `start_N`, `final_N`, `network_plot`}{Same as overall object}
#'       \item{`UVA`}{List with `n_removed` (number of redundant items removed), `n_sweeps`
#'         (number of UVA iterations), `redundant_pairs` (data frame with sweep, items, keep, remove columns)}
#'       \item{`bootEGA`}{List with `initial_boot` (first bootEGA object), `final_boot`
#'         (final bootEGA object with all stable items), `n_removed` (number of unstable items),
#'         `items_removed` (data frame of removed items with boot run information),
#'         `initial_boot_with_redundancies` (boot EGA object for original item pool)}
#'       \item{`stability_plot`}{ggplot/patchwork stability plot showing item stability before vs after reduction}
#'     }
#'   }
#' }
#'
#' @export
local_GENIE <- function(
    # Required parameter
  items,

  # Local embedding parameters
  embedding.model = "bert-base-uncased",
  device = "auto",
  batch.size = 32,
  pooling.strategy = "mean",
  max.length = 512,

  # EGA parameters
  EGA.model = NULL,
  EGA.algorithm = NULL,
  EGA.uni.method = "louvain",

  # Control flags
  embeddings.only = FALSE,
  plot = TRUE,
  silently = FALSE
) {

  # Step 1: Comprehensive input validation
  validation <- validate_user_input_local_GENIE(
    items = items,
    embedding.model = embedding.model,
    device = device,
    batch.size = batch.size,
    pooling.strategy = pooling.strategy,
    max.length = max.length,
    EGA.model = EGA.model,
    EGA.algorithm = EGA.algorithm,
    EGA.uni.method = EGA.uni.method,
    embeddings.only = embeddings.only,
    plot = plot,
    silently = silently
  )

  # Extract validated parameters
  items <- validation$items
  item.attributes <- validation$item.attributes
  embedding.model <- validation$embedding.model
  device <- validation$device
  batch.size <- validation$batch.size
  pooling.strategy <- validation$pooling.strategy
  max.length <- validation$max.length
  EGA.model <- validation$EGA.model
  EGA.algorithm <- validation$EGA.algorithm
  EGA.uni.method <- validation$EGA.uni.method

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

  # Item-level analysis
  try_item_level <- run_item_reduction_pipeline(
    embedding_matrix = embeddings,
    items = items,
    EGA.model = EGA.model,
    EGA.algorithm = EGA.algorithm,
    EGA.uni.method = EGA.uni.method,
    keep.org = FALSE,  # local_GENIE doesn't need to keep original embeddings
    silently = silently,
    plot = plot
  )

  if (!try_item_level$success) {
    warning("Item-level analysis failed. Returning partial results.")
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level

  if(length(names(item.attributes)) > 1){
  # Overall analysis
  try_overall_result <- run_pipeline_for_all(
    item_level = item_level,
    items = items,
    embeddings = embeddings,
    model = EGA.model,
    algorithm = EGA.algorithm,
    uni.method = EGA.uni.method,
    keep.org = FALSE,  # local_GENIE doesn't need to keep original data
    silently = silently,
    plot = plot
  )

  if (!try_overall_result$success) {
    warning("Overall analysis failed. Returning item-level results only.")
    return(item_level)
  }

  overall_result <- try_overall_result$overall_result
  only.one <- FALSE
  } else {
    overall_result <- item_level
    try_overall_result <- list(success = TRUE)
    only.one <- TRUE
  }

  # Step 5: Display results summary
  if(!silently && try_overall_result$success && try_item_level$success){
    print_results(overall_result, item_level, only.one)
  }

  # Step 6: Return comprehensive results
  return(list(
    overall = overall_result,
    item_type_level = item_level
  ))
}
