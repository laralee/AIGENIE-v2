#' @export
AIGENIE_v2 <- function(item.attributes, openai.API=NULL, hf.token=NULL, # required parameters

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
#' creates embeddings, and performs network psychometric validation entirely
#' on the user's machine.
#'
#' @param item.attributes Named list of item types and their attributes (required)
#' @param model.path Path to local GGUF model file (required)
#' @param embedding.model Name or path to local embedding model (default: "bert-base-uncased")
#'
#' @param main.prompts Custom prompts for item generation (optional)
#' @param temperature LLM temperature for randomness (0-2, default: 1)
#' @param top.p Top-p sampling parameter (0-1, default: 1)
#' @param target.N Number of items to generate per type (default: 60)
#'
#' @param domain Content domain (e.g., "psychological")
#' @param scale.title Name of the scale
#' @param item.examples Data frame of example items
#' @param audience Target population
#' @param item.type.definitions Definitions for item types
#' @param response.options Response scale labels
#' @param prompt.notes Additional instructions for generation
#' @param system.role Custom system prompt
#'
#' @param EGA.model Network model ("glasso", "TMFG", or NULL for auto)
#' @param EGA.algorithm Community detection algorithm (default: "walktrap")
#' @param EGA.uni.method Unidimensionality method (default: "louvain")
#'
#' @param n.ctx Context window size (default: 4096)
#' @param n.gpu.layers GPU layers to use (-1 for all, default: -1)
#' @param max.tokens Maximum tokens per generation (default: 1024)
#' @param device Device for embeddings ("auto", "cpu", "cuda", "mps")
#' @param batch.size Batch size for embeddings (default: 32)
#' @param pooling.strategy Pooling for embeddings ("mean", "cls", "max")
#' @param max.length Max sequence length for embeddings (default: 512)
#'
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
