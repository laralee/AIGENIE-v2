#' Validate Local Model Path
#'
#' @param model_path Path to local GGUF model file
#' @param silently Logical. Suppress warnings
#'
#' @return The expanded, validated path
#'
validate_model_path <- function(model_path, silently = FALSE) {

  # Check if provided
  if (is.null(model_path) || !is.character(model_path) || length(model_path) != 1) {
    stop("AI-GENIE LOCAL expects model_path to be a path to a GGUF model file.",
         call. = FALSE)
  }

  # Expand path (handle ~ etc.)
  model_path <- path.expand(model_path)

  # Check if file exists
  if (!file.exists(model_path)) {
    stop(paste0("Model file not found: ", model_path, "\n",
                "Please provide a valid path to a GGUF model file."),
         call. = FALSE)
  }

  # Check file extension
  if (!grepl("\\.gguf$", tolower(model_path))) {
    stop(paste0("Model file must have .gguf extension. Found: ", model_path),
         call. = FALSE)
  }

  # Check file size (warn if suspiciously small)
  file_size_gb <- file.info(model_path)$size / 1024^3
  if (file_size_gb < 0.5) {
    if (!silently) {
      warning("Model file seems unusually small (< 0.5 GB). ",
              "Ensure this is a valid GGUF model.")
    }
  }

  return(model_path)
}


#' Validate Local Embedding Model
#'
#' @description
#' Validates that the embedding model is appropriate for local raw embeddings.
#' Checks for BERT-family models that provide raw feature extraction.
#'
#' @param embedding.model Character string specifying model identifier or path
#' @param silently Logical. Suppress informational messages
#'
#' @return The validated model identifier or path
#'
validate_local_embedding_model <- function(embedding.model, silently = FALSE) {

  # Check basic type requirements
  if (!is.character(embedding.model) || length(embedding.model) != 1 || is.na(embedding.model)) {
    stop(
      "AI-GENIE LOCAL expects embedding.model to be a string specifying a model name or path.",
      call. = FALSE
    )
  }

  # If it's a local path, check it exists
  if (file.exists(path.expand(embedding.model))) {
    expanded_path <- path.expand(embedding.model)
    if (!silently) {
      message("Using local embedding model from: ", expanded_path)
    }
    return(expanded_path)
  }

  # Known compatible models for raw embeddings (not sentence-similarity)
  compatible_models <- c(
    # BERT variants
    "bert-base-uncased",
    "bert-base-cased",
    "bert-large-uncased",
    "bert-large-cased",

    # RoBERTa
    "roberta-base",
    "roberta-large",
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",

    # DeBERTa
    "microsoft/deberta-base",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",

    # ELECTRA
    "google/electra-base-discriminator",
    "google/electra-small-discriminator",

    # DistilBERT
    "distilbert-base-uncased",
    "distilbert-base-cased",

    # ALBERT
    "albert-base-v2",
    "albert-large-v2"
  )

  # Models that are explicitly incompatible (sentence-similarity optimized)
  incompatible_models <- c(
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-distilroberta-v1",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "intfloat/e5-small-v2"
  )

  # Check if it's an incompatible model
  if (embedding.model %in% incompatible_models ||
      grepl("^sentence-transformers/", embedding.model)) {
    stop(
      paste0(
        "Model '", embedding.model, "' is NOT compatible with AI-GENIE LOCAL.\n",
        "This model is optimized for sentence similarity, not raw embeddings.\n\n",
        "Please use a BERT-family model for raw feature extraction:\n",
        "  - 'bert-base-uncased' (recommended, 768 dims)\n",
        "  - 'roberta-base' (768 dims)\n",
        "  - 'distilbert-base-uncased' (lighter, 768 dims)\n",
        "  - 'microsoft/deberta-v3-base' (768 dims)"
      ),
      call. = FALSE
    )
  }

  # Check if it's a known compatible model
  if (embedding.model %in% compatible_models) {
    if (!silently) {
      message("Using embedding model: ", embedding.model)
    }
    return(embedding.model)
  }

  # If unknown, check for BERT-like patterns
  bert_patterns <- c(
    "bert", "roberta", "deberta", "electra",
    "distilbert", "albert", "camembert", "xlm"
  )

  is_likely_bert <- any(sapply(bert_patterns, function(p) {
    grepl(p, tolower(embedding.model))
  }))

  if (is_likely_bert) {
    if (!silently) {
      message(
        "Model '", embedding.model, "' appears to be BERT-based.\n",
        "Proceeding, but ensure it provides raw embeddings, not similarity scores."
      )
    }
    return(embedding.model)
  }

  # Unknown model - warn but allow
  if (!silently) {
    warning(
      paste0(
        "Model '", embedding.model, "' is not recognized.\n",
        "AI-GENIE LOCAL expects BERT-family models for raw embeddings.\n",
        "Recommended models:\n",
        "  - 'bert-base-uncased'\n",
        "  - 'roberta-base'\n",
        "  - 'distilbert-base-uncased'\n",
        "Proceeding anyway - ensure this model provides raw feature embeddings."
      ),
      call. = FALSE,
      immediate. = TRUE
    )
  }

  return(embedding.model)
}


#' Validate Local LLM Generation Parameters
#'
#' @description
#' Validates parameters specific to local LLM generation
#'
#' @param n_ctx Context window size
#' @param n_gpu_layers Number of layers to offload to GPU
#' @param max_tokens Maximum tokens for generation
#'
#' @return A list of validated parameters
#'
validate_local_llm_params <- function(n_ctx, n_gpu_layers, max_tokens) {

  # Validate n_ctx (context window)
  if (!is.numeric(n_ctx) || length(n_ctx) != 1 || is.na(n_ctx)) {
    stop("AI-GENIE LOCAL expects n_ctx to be a numeric value.", call. = FALSE)
  }

  n_ctx <- as.integer(n_ctx)

  if (n_ctx < 128) {
    stop("AI-GENIE LOCAL expects n_ctx to be at least 128.", call. = FALSE)
  }

  if (n_ctx > 32768) {
    warning("n_ctx > 32768 may cause memory issues. Consider reducing if problems occur.")
  }

  # Validate n_gpu_layers
  if (!is.numeric(n_gpu_layers) || length(n_gpu_layers) != 1 || is.na(n_gpu_layers)) {
    stop("AI-GENIE LOCAL expects n_gpu_layers to be a numeric value.", call. = FALSE)
  }

  n_gpu_layers <- as.integer(n_gpu_layers)

  if (n_gpu_layers < -1) {
    stop("AI-GENIE LOCAL expects n_gpu_layers to be -1 (all) or >= 0.", call. = FALSE)
  }

  # Validate max_tokens
  if (!is.numeric(max_tokens) || length(max_tokens) != 1 || is.na(max_tokens)) {
    stop("AI-GENIE LOCAL expects max_tokens to be a numeric value.", call. = FALSE)
  }

  max_tokens <- as.integer(max_tokens)

  if (max_tokens < 1) {
    stop("AI-GENIE LOCAL expects max_tokens to be at least 1.", call. = FALSE)
  }

  if (max_tokens > n_ctx) {
    warning("max_tokens exceeds n_ctx. Setting max_tokens to n_ctx - 100.")
    max_tokens <- n_ctx - 100
  }

  return(list(
    n_ctx = n_ctx,
    n_gpu_layers = n_gpu_layers,
    max_tokens = max_tokens
  ))
}


#' Validate Local Embedding Parameters
#'
#' @description
#' Validates parameters specific to local embedding generation
#'
#' @param device Device for computation ("auto", "cpu", "cuda", "mps")
#' @param batch_size Number of items to process simultaneously
#' @param pooling_strategy Strategy for pooling token embeddings
#' @param max_length Maximum sequence length for tokenization
#'
#' @return A list of validated parameters
#'
validate_local_embedding_params <- function(device, batch_size, pooling_strategy, max_length) {

  # Validate device
  valid_devices <- c("auto", "cpu", "cuda", "mps")

  if (!is.character(device) || length(device) != 1 || is.na(device)) {
    stop("AI-GENIE LOCAL expects device to be a string.", call. = FALSE)
  }

  device <- tolower(trimws(device))

  if (!device %in% valid_devices) {
    stop(
      paste0(
        "AI-GENIE LOCAL expects device to be one of: ",
        paste(sprintf("'%s'", valid_devices), collapse = ", "),
        ". Received: '", device, "'"
      ),
      call. = FALSE
    )
  }

  # Validate batch_size
  if (!is.numeric(batch_size) || length(batch_size) != 1 || is.na(batch_size)) {
    stop("AI-GENIE LOCAL expects batch_size to be a numeric value.", call. = FALSE)
  }

  batch_size <- as.integer(batch_size)

  if (batch_size < 1) {
    stop("AI-GENIE LOCAL expects batch_size to be at least 1.", call. = FALSE)
  }

  if (batch_size > 128) {
    warning("Large batch_size (>128) may cause memory issues. Consider reducing if problems occur.")
  }

  # Validate pooling_strategy
  valid_strategies <- c("mean", "cls", "max")

  if (!is.character(pooling_strategy) || length(pooling_strategy) != 1 || is.na(pooling_strategy)) {
    stop("AI-GENIE LOCAL expects pooling_strategy to be a string.", call. = FALSE)
  }

  pooling_strategy <- tolower(trimws(pooling_strategy))

  if (!pooling_strategy %in% valid_strategies) {
    stop(
      paste0(
        "AI-GENIE LOCAL expects pooling_strategy to be one of: ",
        paste(sprintf("'%s'", valid_strategies), collapse = ", "),
        ". Received: '", pooling_strategy, "'"
      ),
      call. = FALSE
    )
  }

  # Validate max_length
  if (!is.numeric(max_length) || length(max_length) != 1 || is.na(max_length)) {
    stop("AI-GENIE LOCAL expects max_length to be a numeric value.", call. = FALSE)
  }

  max_length <- as.integer(max_length)

  if (max_length < 10) {
    stop("AI-GENIE LOCAL expects max_length to be at least 10.", call. = FALSE)
  }

  if (max_length > 512) {
    warning(
      paste0(
        "max_length > 512 exceeds typical BERT model limits. ",
        "Most models max out at 512 tokens. Setting to 512."
      )
    )
    max_length <- 512
  }

  return(list(
    device = device,
    batch_size = batch_size,
    pooling_strategy = pooling_strategy,
    max_length = max_length
  ))
}
