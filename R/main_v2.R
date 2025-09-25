#' @export
AIGENIE <- function(item.attributes, openai.API=NULL, hf.token=NULL, # required parameters

                       # optional parameters --

                       # if using AIGENIE in custom mode, this should be set:
                       main.prompts = NULL,

                       # LLM parameters
                       groq.API = NULL, model = "gpt4o", temperature = 1,
                       top.p = 1, embedding.model = "text-embedding-3-small",
                       target.N = NULL,

                       # Prompt parameters
                       domain = NULL, scale.title = NULL, item.examples = NULL,
                       audience = NULL, item.type.definitions = NULL,
                       response.options = NULL, prompt.notes = NULL, system.role = NULL,

                       # EGA parameters
                       EGA.model = NULL, EGA.algorithm = "walktrap", EGA.uni.method = "louvain",

                       # Flags
                       keep.org = FALSE, items.only = FALSE, embeddings.only = FALSE,
                       adaptive = TRUE, plot = TRUE, silently = FALSE
                       ){


  # Validate all params and reassign params
  validation <- validate_user_input_AIGENIE(item.attributes, openai.API, hf.token,
                                            main.prompts,
                                            groq.API, model, temperature,
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
                                  adaptive, silently, groq.API, openai.API, target.N)
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
  if(validation$provider == "openai"){
    attempt_to_embed <- embed_items(embedding.model, openai.API, items, silently)
  } else {
    attempt_to_embed <- embed_items_huggingface(embedding.model, hf.token, items, silently)
  }

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
  try_item_level <- run_item_reduction_pipeline(embeddings,
                                         items,
                                         EGA.model,
                                         EGA.algorithm,
                                         EGA.uni.method,
                                         keep.org,
                                         silently,
                                         plot)

  if(!try_item_level$success){
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level


  # If successful, generate results for items overall
  try_overall_result <- run_pipeline_for_all(item_level,
                                         items,
                                         embeddings,
                                         EGA.model,
                                         EGA.algorithm,
                                         EGA.uni.method,
                                         keep.org,
                                         silently,
                                         plot)

  if(!try_overall_result$success){
    return(item_level)
  }

  overall_result <- try_overall_result$overall_result

  if(!silently){
    print_results(overall_result, item_level)
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
#' @param EGA.algorithm Community detection algorithm (default: "walktrap")
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
#' @return Depending on flags:
#'   - Full pipeline: List with overall and item-type level results
#'   - items.only: Data frame of generated items
#'   - embeddings.only: List with embeddings matrix and items
#'
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
  EGA.algorithm = "walktrap",
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
    embeddings, items, EGA.model, EGA.algorithm,
    EGA.uni.method, keep.org, silently, plot
  )

  if (!try_item_level$success) {
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level

  # Overall reduction
  try_overall_result <- run_pipeline_for_all(
    item_level, items, embeddings, EGA.model,
    EGA.algorithm, EGA.uni.method, keep.org,
    silently, plot
  )

  if (!try_overall_result$success) {
    return(item_level)
  }

  overall_result <- try_overall_result$overall_result

  # Step 7: Print results summary
  if (!silently) {
    print_results(overall_result, item_level)
  }

  # Return results
  return(list(
    overall = overall_result,
    item_type_level = item_level
  ))
}


#' Generative Network-Integrated Evaluation (GENIE)
#'
#' @description
#' GENIE provides psychometric validation and quality assessment for user-supplied
#' items using the same network psychometric pipeline as AI-GENIE, but without the
#' item generation phase. Users provide their own items and optionally their own
#' embeddings, then GENIE performs redundancy reduction, community detection, and
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
#'     \item If NULL, embeddings will be generated using embedding.model
#'   }
#'
#' @param openai.API OpenAI API key (required if using OpenAI embedding models)
#' @param hf.token HuggingFace token (optional, improves rate limits for HF models)
#' @param groq.API Groq API key (currently unused in GENIE)
#' @param model Language model identifier (currently unused in GENIE)
#' @param temperature LLM temperature (currently unused in GENIE)
#' @param top.p LLM top-p parameter (currently unused in GENIE)
#' @param embedding.model Embedding model to use if embedding.matrix not provided:
#'   \itemize{
#'     \item OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
#'     \item HuggingFace: "BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5", etc.
#'   }
#' @param EGA.model Network estimation model ("glasso", "TMFG", or NULL for auto-selection)
#' @param EGA.algorithm Community detection algorithm ("walktrap", "leiden", "louvain")
#' @param EGA.uni.method Unidimensionality assessment method ("louvain", "expand", "LE")
#' @param embeddings.only If TRUE, return embeddings and stop (skip network analysis)
#' @param plot If TRUE, display network comparison plots
#' @param silently If TRUE, suppress progress messages
#'
#' @return Depending on embeddings.only flag:
#'   \itemize{
#'     \item If embeddings.only = TRUE: List with embeddings matrix and items
#'     \item If embeddings.only = FALSE: List with overall and item-type level results
#'   }
#'
#' @details
#' GENIE workflow:
#' 1. Validate and clean user-provided items
#' 2. Generate embeddings (if not provided) using specified embedding model
#' 3. Run network psychometric pipeline: redundancy reduction, community detection, stability analysis
#' 4. Return comprehensive results with network plots and quality metrics
#'
#' Unlike AI-GENIE which generates items from scratch, GENIE evaluates existing items,
#' making it ideal for researchers who want to validate their own item pools using
#' modern network psychometric methods.
#'
#' @export
GENIE <- function(
    items,                                    # Required: user items
    embedding.matrix = NULL,                  # Optional: user embeddings

    # API parameters
    openai.API = NULL,
    hf.token = NULL,
    groq.API = NULL,                         # Unused but kept for consistency

    # LLM parameters (unused but kept for consistency)
    model = "gpt4o",
    temperature = 1,
    top.p = 1,

    # Embedding parameters
    embedding.model = "text-embedding-3-small",

    # EGA parameters
    EGA.model = NULL,
    EGA.algorithm = "walktrap",
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

    # Generate embeddings using the same functions as AIGENIE
    if (provider == "openai") {
      embedding_result <- embed_items(
        embedding.model = embedding.model,
        openai.API = openai.API,
        items = items,
        silently = silently
      )
    } else {
      embedding_result <- embed_items_huggingface(
        embedding.model = embedding.model,
        hf.token = hf.token,
        items = items,
        silently = silently
      )
    }

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
    if (!silently) {
      cat("Embeddings generated successfully. Returning embeddings and items.\n")
    }
    return(list(
      embeddings = embeddings,
      items = items
    ))
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

  # Step 5: Display results summary
  if (!silently) {
    print_results(overall_result, item_level)
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
#'     \item Local paths: "/path/to/local/model"
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
#' @param embeddings.only If TRUE, return embeddings and stop (skip network analysis)
#' @param plot If TRUE, display network comparison plots
#' @param silently If TRUE, suppress progress messages
#'
#' @return Depending on embeddings.only flag:
#'   \itemize{
#'     \item If embeddings.only = TRUE: List with embeddings matrix and items
#'     \item If embeddings.only = FALSE: List with overall and item-type level results
#'   }
#'
#' @details
#' local_GENIE workflow:
#' 1. Validate and clean user-provided items
#' 2. Generate embeddings locally using specified transformer model
#' 3. Run network psychometric pipeline: redundancy reduction, community detection, stability analysis
#' 4. Return comprehensive results with network plots and quality metrics
#'
#' Unlike regular GENIE which uses API-based embeddings, local_GENIE processes everything
#' on the user's machine for complete privacy and offline capability. This makes it
#' ideal for sensitive data or environments without internet access.
#'
#' @note
#' Requirements:
#' - Python environment with transformers, torch, and numpy
#' - Local transformer model (downloaded automatically on first use)
#' - Sufficient RAM/VRAM for model and batch processing
#'
#' @examples
#' \dontrun{
#' # Create sample items
#' items <- data.frame(
#'   statement = c("I enjoy social gatherings", "I prefer working alone",
#'                 "I feel anxious in crowds", "I am comfortable speaking publicly"),
#'   attribute = c("extraversion", "introversion", "social_anxiety", "confidence"),
#'   type = c("personality", "personality", "anxiety", "anxiety"),
#'   ID = 1:4,
#'   stringsAsFactors = FALSE
#' )
#'
#' # Run local_GENIE with default BERT model
#' results <- local_GENIE(
#'   items = items,
#'   embedding.model = "bert-base-uncased"
#' )
#'
#' # Run with GPU acceleration and larger batch size
#' results <- local_GENIE(
#'   items = items,
#'   embedding.model = "roberta-base",
#'   device = "cuda",
#'   batch.size = 64
#' )
#'
#' # Generate embeddings only for external use
#' embeddings_result <- local_GENIE(
#'   items = items,
#'   embedding.model = "distilbert-base-uncased",
#'   embeddings.only = TRUE
#' )
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
  EGA.algorithm = "walktrap",
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

  # Step 5: Display results summary
  if (!silently) {
    print_results(overall_result, item_level)
  }

  # Step 6: Return comprehensive results
  return(list(
    overall = overall_result,
    item_type_level = item_level
  ))
}
