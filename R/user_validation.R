#### Validate User Input ####

# AIGENIE ----
#' Validate All User Inputs for AI-GENIE
#'
#' This function performs comprehensive validation and normalization of all user-supplied
#' inputs to the AI-GENIE package. It checks logical flags, strings, model names, item
#' attribute structures, and ensures consistency across all interdependent components.
#'
#' If any input is invalid or misaligned with the package’s expected structure,
#' informative errors or warnings are raised. Cleaned and normalized objects are returned
#' for use downstream.
#'
#' @param item.attributes A named list of attributes and item types. Must be validated via
#'   \code{item.attributes_validate()}.
#' @param openai.API A string. OpenAI API key.
#' @param hf.token A string. HuggingFace API key.
#' @param main.prompts A named list of custom prompts that the user specifies (if desired)
#' @param groq.API A string or NULL. Groq API key.
#' @param model A string. The user-specified language model. Will be resolved to a
#'   canonical model name using \code{normalize_model_name()}.
#' @param temperature A numeric value between 0 and 2.
#' @param top.p A numeric value between 0 and 1.
#' @param embedding.model A string or NULL. Must be one of the accepted OpenAI embedding models.
#' @param target.N Either a scalar integer, NULL, or a named list/vector of integers
#'   corresponding to each attribute. Used for synthetic item generation.
#' @param domain A string describing the domain of the assessment.
#' @param scale.title A string naming the scale.
#' @param item.examples A data frame containing `type`, `attribute`, and `statement` columns.
#'   All values must be strings. Optional.
#' @param audience A string or NULL. The intended audience of the assessment.
#' @param item.type.definitions A named list mapping item types to their descriptions.
#'   Optional.
#' @param response.options An atomic vector of strings listing the response options users will have.
#'   Optional.
#' @param prompt.notes A named list or string that gives the LLM additional instructions to be appended to the prompt.
#'   Optional.
#' @param system.role A string or NULL. Used to customize the system prompt.
#' @param EGA.model A string or NULL. One of `"BGGM"`, `"glasso"`, or `"TMFG"`.
#' @param EGA.algorithm A string. One of `"leiden"`, `"louvain"`, or `"walktrap"`.
#' @param EGA.uni.method A string. One of `"expand"`, `"LE"`, or `"louvain"`.
#' @param keep.org A boolean. If TRUE, preserve original inputs in the output.
#' @param items.only A boolean. Whether to generate only items.
#' @param embeddings.only A boolean. Whether to run in embedding-only mode.
#' @param adaptive A boolean. Whether adaptive design logic should be applied.
#' @param plot A boolean. Whether to display plots for visual diagnostics.
#' @param silently A boolean. If TRUE, suppresses warning messages.
#'
#' @return A named list containing:
#' \describe{
#'   \item{target.N}{A named list of integers, aligned with `item.attributes`}
#'   \item{EGA.model}{Canonical model string or NULL}
#'   \item{EGA.uni.method}{Canonical unidimensionality method}
#'   \item{EGA.algorithm}{Canonical community detection algorithm}
#'   \item{model}{Resolved model string for text generation}
#'   \item{item.type.definitions}{Cleaned item type definitions (if provided)}
#'   \item{item.examples}{Cleaned item examples (if provided)}
#'   \item{item.attributes}{Cleaned and normalized item attributes}
#'   \item{prompt.notes}{Cleaned and normalized prompt notes (if provided)}
#'   \item{main.prompts}{Cleaned and normalized main prompts (if provided)}
#'   \item{custom}{A flag signaling whether we are in custom mode or not}
#' }
#'
validate_user_input_AIGENIE <- function(item.attributes, openai.API, hf.token,
                                        main.prompts,
                                        groq.API, anthropic.API, jina.API,
                                        model, temperature,
                                        top.p, embedding.model, target.N,
                                        domain, scale.title, item.examples,
                                        audience, item.type.definitions, response.options,
                                        prompt.notes,
                                        system.role, EGA.model, EGA.algorithm,
                                        EGA.uni.method, keep.org, items.only,
                                        embeddings.only, adaptive, run.overall,
                                        all.together,
                                        plot, silently) {

  # Ensure all "TRUEs and FALSEs" are specified accordingly
  validate_booleans(items.only, adaptive, plot, keep.org, silently,
                    run.overall, embeddings.only, all.together)

  # Ensure all string objects are actually strings (or set to NULL)
  validate_strings(openai.API, groq.API, anthropic.API, jina.API, hf.token,
                   audience, scale.title,
                   system.role, domain, model,
                   embedding.model)

  # Check if the user forgot to add their API keys if using example code
  check_for_default_APIs(hf.token, groq.API, openai.API, anthropic.API, jina.API)

  # Validate the `item.attributes` object
  item.attributes <- items.attributes_validate(item.attributes)

  # Validate the `item.examples` object based on `item.attributes`
  if(!is.null(item.examples)){ # only run the check if user provided
    item.examples <- item.examples_validate(item.examples, item.attributes)
  }

  # Validate the `item.type.definitions` object based on `item.attributes`
  if(!is.null(item.type.definitions)){ # only run the check if user provided
    item.type.definitions <- item.type.definitions_validate(item.type.definitions, item.attributes)
  }

  # Validate the `model` string and replace it with a valid model string if necessary
  model_info <- normalize_model_name(model, groq.API, openai.API,
                                 anthropic.API = anthropic.API, silently = silently)
  model <- model_info$model

  # Validate the embedding model
  provider <- embedding.model_validate(embedding.model)

  # Validate the Run flags
  run_flags <- run_flags_validate(run.overall, all.together, item.attributes,
                                  silently)
  run.overall <- run_flags$run.overall
  all.together <- run_flags$all.together

  # Validate the parameters to be passed to EGA
  EGA_params <- validate_ega_params(EGA.algorithm, EGA.uni.method, EGA.model)
  EGA.algorithm <- EGA_params$EGA.algorithm
  EGA.uni.method <- EGA_params$EGA.uni.method
  EGA.model <- EGA_params$EGA.model


  # Validate target N
  target.N <- target.N_validate(target.N, item.attributes, items.only, embeddings.only, silently)


  # Validate LLM parameters
  top.p_validate(top.p)
  temperature_validate(temperature)

  # Validate prompt components
  response.options_validate(response.options)
  prompt.notes <- validate_prompt.notes(prompt.notes, item.attributes)

  # Check to see if the user is in custom mode
  if(!is.null(main.prompts)){
    main.prompts_validation <- main.prompts_validate(main.prompts, item.attributes, silently)
    custom <- main.prompts_validation$custom
    main.prompts <- main.prompts_validation$out

  } else {
    custom <- FALSE
  }

  # Return
  return(list(
    target.N = target.N,
    EGA.model = EGA.model,
    EGA.uni.method = EGA.uni.method,
    EGA.algorithm = EGA.algorithm,
    model = model,
    item.type.definitions = item.type.definitions,
    item.examples = item.examples,
    item.attributes = item.attributes,
    prompt.notes = prompt.notes,
    main.prompts = main.prompts,
    custom = custom,
    provider = provider,
    all.together = all.together,
    run.overall = run.overall

  ))

}

#' Validate All User Inputs for Local AI-GENIE
#'
#' @description
#' Comprehensive validation of all inputs for local model execution.
#' Reuses existing validators where applicable and adds local-specific validations.
#'
#' @param item.attributes Named list of attributes (same as API version)
#' @param model.path Path to local GGUF model
#' @param embedding.model Local embedding model identifier
#' @param main.prompts Optional custom prompts
#' @param temperature LLM temperature
#' @param top.p LLM top-p sampling
#' @param target.N Target number of items
#' @param domain Assessment domain
#' @param scale.title Scale name
#' @param item.examples Example items
#' @param audience Target audience
#' @param item.type.definitions Type definitions
#' @param response.options Response scale options
#' @param prompt.notes Additional prompt instructions
#' @param system.role System prompt
#' @param EGA.model EGA model type
#' @param EGA.algorithm EGA algorithm
#' @param EGA.uni.method EGA unidimensionality method
#' @param n.ctx Context window size
#' @param n.gpu.layers GPU layers
#' @param max.tokens Maximum generation tokens
#' @param device Embedding computation device
#' @param batch.size Embedding batch size
#' @param pooling.strategy Embedding pooling strategy
#' @param max.length Embedding max sequence length
#' @param keep.org Keep original data
#' @param items.only Generate items only
#' @param embeddings.only Generate embeddings only
#' @param adaptive Use adaptive generation
#' @param plot Show plots
#' @param silently Suppress messages
#'
#' @return A list of all validated parameters
#'
validate_user_input_local_AIGENIE <- function(
    item.attributes, model.path, embedding.model,
    main.prompts, temperature, top.p, target.N,
    domain, scale.title, item.examples, audience,
    item.type.definitions, response.options, prompt.notes,
    system.role, EGA.model, EGA.algorithm, EGA.uni.method,
    n.ctx, n.gpu.layers, max.tokens,
    device, batch.size, pooling.strategy, max.length,
    keep.org, items.only, embeddings.only, adaptive, run.overall, run.together,
    plot, silently
) {

  # 1. Validate booleans
  validate_booleans(items.only, adaptive, plot, keep.org, silently,
                    run.overall, run.together, embeddings.only)

  # 2. Validate strings
  validate_strings(audience, scale.title, system.role, domain,
                   EGA.model, EGA.algorithm, EGA.uni.method)

  # 3. Validate local-specific paths and models
  model.path <- validate_model.path(model.path, silently)
  embedding.model <- validate_local_embedding_model(embedding.model, silently)

  # 4. Validate item.attributes
  item.attributes <- items.attributes_validate(item.attributes)

  # 5. Validate optional data structures
  if (!is.null(item.examples)) {
    item.examples <- item.examples_validate(item.examples, item.attributes)
  }

  if (!is.null(item.type.definitions)) {
    item.type.definitions <- item.type.definitions_validate(item.type.definitions, item.attributes)
  }

  # 6. Validate EGA parameters (and run flags)
  run_flags <- run_flags_validate(run.overall, all.together, item.attributes, silently)
  run.overall <- run_flags$run.overall
  all.together <- run_flags$all.together

  EGA_params <- validate_ega_params(EGA.algorithm, EGA.uni.method, EGA.model)
  EGA.algorithm <- EGA_params$EGA.algorithm
  EGA.uni.method <- EGA_params$EGA.uni.method
  EGA.model <- EGA_params$EGA.model

  # 7. Validate target.N
  target.N <- target.N_validate(target.N, item.attributes, items.only, embeddings.only, silently)

  # 8. Validate LLM parameters
  top.p_validate(top.p)
  temperature_validate(temperature)

  # 9. Validate local LLM specific parameters
  llm_params <- validate_local_llm_params(n.ctx, n.gpu.layers, max.tokens)

  # 10. Validate local embedding parameters
  embed_params <- validate_local_embedding_params(device, batch.size, pooling.strategy, max.length)

  # 11. Validate prompt components
  response.options_validate(response.options)
  prompt.notes <- validate_prompt.notes(prompt.notes, item.attributes)

  # 12. Check custom prompts mode
  custom <- FALSE
  if (!is.null(main.prompts)) {
    main.prompts_validation <- main.prompts_validate(main.prompts, item.attributes, silently)
    custom <- main.prompts_validation$custom
    main.prompts <- main.prompts_validation$out
  }

  # Return all validated parameters
  return(list(
    # Core parameters
    item.attributes = item.attributes,
    model.path = model.path,
    embedding.model = embedding.model,

    # LLM parameters
    temperature = temperature,
    top.p = top.p,
    n.ctx = llm_params$n.ctx,
    n.gpu.layers = llm_params$n.gpu.layers,
    max.tokens = llm_params$max.tokens,

    # Embedding parameters
    device = embed_params$device,
    batch.size = embed_params$batch.size,
    pooling.strategy = embed_params$pooling.strategy,
    max.length = embed_params$max.length,

    # EGA parameters
    EGA.model = EGA.model,
    EGA.uni.method = EGA.uni.method,
    EGA.algorithm = EGA.algorithm,

    # Content parameters
    target.N = target.N,
    item.type.definitions = item.type.definitions,
    item.examples = item.examples,
    prompt.notes = prompt.notes,
    main.prompts = main.prompts,
    custom = custom,

    # Flags
    items.only = items.only,
    embeddings.only = embeddings.only,
    keep.org = keep.org,
    adaptive = adaptive,
    plot = plot,
    silently = silently,
    run.overall = run.overall,
    all.together = all.together
  ))
}


#' Validate All User Inputs for GENIE
#'
#' @param items A data frame with columns: statement, attribute, type, ID
#' @param embedding.matrix Optional numeric matrix/data frame with items as columns
#' @param openai.API OpenAI API key (string or NULL)
#' @param hf.token HuggingFace token (string or NULL)
#' @param groq.API Groq API key (string or NULL)
#' @param model Language model identifier (string)
#' @param temperature LLM temperature parameter (numeric, 0-2)
#' @param top.p LLM top-p parameter (numeric, 0-1)
#' @param embedding.model Embedding model identifier (string)
#' @param EGA.model EGA network model (string or NULL)
#' @param EGA.algorithm EGA algorithm (string)
#' @param EGA.uni.method EGA unidimensionality method (string)
#' @param embeddings.only Whether to stop after embeddings (boolean)
#' @param plot Whether to show plots (boolean)
#' @param silently Whether to suppress messages (boolean)
#'
#' @return A named list containing all validated and normalized parameters
#'
validate_user_input_GENIE <- function(
    items,
    embedding.matrix,
    openai.API,
    hf.token,
    groq.API,
    jina.API,
    model,
    temperature,
    top.p,
    embedding.model,
    EGA.model,
    EGA.algorithm,
    EGA.uni.method,
    embeddings.only,
    run.overall,
    all.together,
    plot,
    silently
) {

  # 1. Validate boolean parameters
  validate_booleans(embeddings.only, plot, silently)

  # 2. Validate string parameters (allowing NULL where appropriate)
  validate_strings(openai.API, groq.API, jina.API, hf.token, model,
                   EGA.model, EGA.algorithm, EGA.uni.method, embedding.model)

  # Check if the user forgot to add their API keys if using example code
  check_for_default_APIs(hf.token, groq.API, openai.API, jina.API = jina.API)

  # 3. Validate and clean the items data frame (GENIE-specific)
  items <- items_validate_GENIE(items)

  # 4. Build item.attributes from the validated items (GENIE-specific)
  item.attributes <- build_item_attributes_from_items(items)

  # 5. Validate embedding matrix if provided (GENIE-specific)
  embedding.matrix <- embedding_matrix_validate_GENIE(
    embedding.matrix,
    items,
    silently
  )

  # 6. Validate model string and resolve to canonical form
  model_info <- normalize_model_name(model, silently = silently)
  model <- model_info$model

  # 7. Validate embedding model and detect provider
  provider <- embedding.model_validate(embedding.model)

  # 8. Validate EGA parameters (and run flags)
  run_flags <- run_flags_validate(run.overall, all.together, item.attributes, silently)
  run.overall <- run_flags$run.overall
  all.together <- run_flags$all.together

  EGA_params <- validate_ega_params(EGA.algorithm, EGA.uni.method, EGA.model)
  EGA.algorithm <- EGA_params$EGA.algorithm
  EGA.uni.method <- EGA_params$EGA.uni.method
  EGA.model <- EGA_params$EGA_model

  # 9. Validate LLM parameters
  temperature_validate(temperature)
  top.p_validate(top.p)

  # 10. Check API key availability based on what will be needed
  if (is.null(embedding.matrix)) {
    # Will need to generate embeddings
    if (provider == "openai" && is.null(openai.API)) {
      stop(
        paste0(
          "GENIE requires an OpenAI API key to generate embeddings with model '",
          embedding.model, "'.\n",
          "Please provide openai.API or use a HuggingFace or Jina model instead."
        ),
        call. = FALSE
      )
    }

    if (provider == "jina" && is.null(jina.API) &&
        nchar(Sys.getenv("JINA_API_KEY", unset = "")) == 0) {
      stop(
        paste0(
          "GENIE requires a Jina AI API key to generate embeddings with model '",
          embedding.model, "'.\n",
          "Please provide jina.API or set the JINA_API_KEY environment variable."
        ),
        call. = FALSE
      )
    }

    if (provider == "huggingface" && is.null(hf.token) && !silently) {
      warning(
        paste0(
          "No HuggingFace token provided. This may result in lower rate limits.\n",
          "Consider providing hf.token for better performance."
        ),
        call. = FALSE,
        immediate. = TRUE
      )
    }
  }

  # 11. Validate that we have sufficient data for meaningful analysis
  if (!embeddings.only) {
    # Count items per type-attribute combination
    type_attr_counts <- table(items$type, items$attribute)
    min_count <- min(type_attr_counts[type_attr_counts > 0])

    if (min_count < 10) {
      warning(
        paste0(
          "GENIE detected type-attribute combinations with very few items (minimum: ",
          min_count, ").\n",
          "Network analysis may be unreliable with fewer than 10 items per combination."
        ),
        call. = FALSE,
        immediate. = TRUE
      )
    }
  }

  # 12. Return all validated parameters
  return(list(
    # GENIE-specific validated parameters
    items = items,
    embedding.matrix = embedding.matrix,
    item.attributes = item.attributes,

    # Shared parameters (validated)
    model = model,
    temperature = temperature,
    top.p = top.p,
    embedding.model = embedding.model,
    provider = provider,
    EGA.model = EGA.model,
    EGA.algorithm = EGA.algorithm,
    EGA.uni.method = EGA.uni.method,

    # API keys (as-is, may be NULL)
    openai.API = openai.API,
    hf.token = hf.token,
    groq.API = groq.API,

    # Flags
    embeddings.only = embeddings.only,
    plot = plot,
    silently = silently,
    run.overall = run.overall,
    all.together = all.together
  ))
}



#' Validate All User Inputs for Local GENIE
#'
#' @param items Data frame with columns: statement, attribute, type, ID
#' @param embedding.model Local embedding model identifier or path
#' @param device Device for embeddings ("auto", "cpu", "cuda", "mps")
#' @param batch.size Batch size for embedding generation
#' @param pooling.strategy Pooling strategy ("mean", "cls", "max")
#' @param max.length Maximum sequence length for embeddings
#' @param EGA.model EGA network model ("glasso", "TMFG", or NULL)
#' @param EGA.algorithm EGA algorithm ("walktrap", "leiden", "louvain")
#' @param EGA.uni.method EGA unidimensionality method ("louvain", "expand", "LE")
#' @param embeddings.only Whether to stop after embeddings
#' @param plot Whether to show plots
#' @param silently Whether to suppress messages
#'
#' @return A list of all validated parameters ready for local GENIE execution
#'
validate_user_input_local_GENIE <- function(
    items,
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
) {

  # 1. Validate boolean parameters
  validate_booleans(embeddings.only, plot, silently, run.overall, all.together)

  # 2. Validate string parameters
  validate_strings(embedding.model, EGA.model, EGA.algorithm, EGA.uni.method)

  # 3. Validate and clean the items data frame (GENIE-specific)
  items <- items_validate_GENIE(items)

  # 4. Build item.attributes from the validated items (GENIE-specific)
  item.attributes <- build_item_attributes_from_items(items)

  # 5. Validate local embedding model
  embedding.model <- validate_local_embedding_model(embedding.model, silently)

  # 6. Validate local embedding parameters
  embed_params <- validate_local_embedding_params(
    device,
    batch.size,
    pooling.strategy,
    max.length
  )

  # 7. Validate EGA parameters (and run flags)
  run_flags <- run_flags_validate(run.overall, all.together, item.attributes, silently)
  run.overall <- run_flags$run.overall
  all.together <- run_flags$all.together

  EGA_params <- validate_ega_params(EGA.algorithm, EGA.uni.method, EGA.model)
  EGA.algorithm <- EGA_params$EGA.algorithm
  EGA.uni.method <- EGA_params$EGA.uni.method
  EGA.model <- EGA_params$EGA_model

  # 8. Validate that we have sufficient data for meaningful analysis
  if (!embeddings.only) {
    # Count items per type-attribute combination
    type_attr_counts <- table(items$type, items$attribute)
    min_count <- min(type_attr_counts[type_attr_counts > 0])

    if (min_count < 10) {
      warning(
        paste0(
          "Local GENIE detected type-attribute combinations with very few items (minimum: ",
          min_count, ").\n",
          "Network analysis may be unreliable with fewer than 10 items per combination."
        ),
        call. = FALSE,
        immediate. = TRUE
      )
    }
  }

  # 9. Inform user about local embedding setup
  if (!silently) {
    message("Local GENIE will generate embeddings using: ", embedding.model)
    message("Embedding generation device: ", embed_params$device)
  }

  # 10. Return all validated parameters
  return(list(
    # GENIE-specific validated parameters
    items = items,
    item.attributes = item.attributes,

    # Local embedding parameters
    embedding.model = embedding.model,
    device = embed_params$device,
    batch.size = embed_params$batch.size,
    pooling.strategy = embed_params$pooling.strategy,
    max.length = embed_params$max.length,

    # EGA parameters
    EGA.model = EGA.model,
    EGA.algorithm = EGA.algorithm,
    EGA.uni.method = EGA.uni.method,

    # Control flags
    embeddings.only = embeddings.only,
    plot = plot,
    silently = silently,
    run.overall = run.overall,
    all.together = all.together
  ))
}
