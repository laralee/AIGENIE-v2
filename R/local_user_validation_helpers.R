#' Validate Local Model Path
#'
#' @param model.path Path to local GGUF model file
#' @param silently Logical. Suppress warnings
#'
#' @return The expanded, validated path
#'
validate_model.path <- function(model.path, silently = FALSE) {

  # Check if provided
  if (is.null(model.path) || !is.character(model.path) || length(model.path) != 1) {
    stop("AI-GENIE LOCAL expects model.path to be a path to a GGUF model file.",
         call. = FALSE)
  }

  # Expand path (handle ~ etc.)
  model.path <- path.expand(model.path)

  # Check if file exists
  if (!file.exists(model.path)) {
    stop(paste0("Model file not found: ", model.path, "\n",
                "Please provide a valid path to a GGUF model file."),
         call. = FALSE)
  }

  # Check file extension
  if (!grepl("\\.gguf$", tolower(model.path))) {
    stop(paste0("Model file must have .gguf extension. Found: ", model.path),
         call. = FALSE)
  }

  # Check file size (warn if suspiciously small)
  file_size_gb <- file.info(model.path)$size / 1024^3
  if (file_size_gb < 0.5) {
    if (!silently) {
      warning("Model file seems unusually small (< 0.5 GB). ",
              "Ensure this is a valid GGUF model.")
    }
  }

  return(model.path)
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
#' @param n.ctx Context window size
#' @param n.gpu.layers Number of layers to offload to GPU
#' @param max.tokens Maximum tokens for generation
#'
#' @return A list of validated parameters
#'
validate_local_llm_params <- function(n.ctx, n.gpu.layers, max.tokens) {

  # Validate n.ctx (context window)
  if (!is.numeric(n.ctx) || length(n.ctx) != 1 || is.na(n.ctx)) {
    stop("AI-GENIE LOCAL expects n.ctx to be a numeric value.", call. = FALSE)
  }

  n.ctx <- as.integer(n.ctx)

  if (n.ctx < 128) {
    stop("AI-GENIE LOCAL expects n.ctx to be at least 128.", call. = FALSE)
  }

  if (n.ctx > 32768) {
    warning("n.ctx > 32768 may cause memory issues. Consider reducing if problems occur.")
  }

  # Validate n.gpu.layers
  if (!is.numeric(n.gpu.layers) || length(n.gpu.layers) != 1 || is.na(n.gpu.layers)) {
    stop("AI-GENIE LOCAL expects n.gpu.layers to be a numeric value.", call. = FALSE)
  }

  n.gpu.layers <- as.integer(n.gpu.layers)

  if (n.gpu.layers < -1) {
    stop("AI-GENIE LOCAL expects n.gpu.layers to be -1 (all) or >= 0.", call. = FALSE)
  }

  # Validate max.tokens
  if (!is.numeric(max.tokens) || length(max.tokens) != 1 || is.na(max.tokens)) {
    stop("AI-GENIE LOCAL expects max.tokens to be a numeric value.", call. = FALSE)
  }

  max.tokens <- as.integer(max.tokens)

  if (max.tokens < 1) {
    stop("AI-GENIE LOCAL expects max.tokens to be at least 1.", call. = FALSE)
  }

  if (max.tokens > n.ctx) {
    warning("max.tokens exceeds n.ctx. Setting max.tokens to n.ctx - 100.")
    max.tokens <- n.ctx - 100
  }

  return(list(
    n.ctx = n.ctx,
    n.gpu.layers = n.gpu.layers,
    max.tokens = max.tokens
  ))
}


#' Validate Local Embedding Parameters
#'
#' @description
#' Validates parameters specific to local embedding generation
#'
#' @param device Device for computation ("auto", "cpu", "cuda", "mps")
#' @param batch.size Number of items to process simultaneously
#' @param pooling.strategy Strategy for pooling token embeddings
#' @param max.length Maximum sequence length for tokenization
#'
#' @return A list of validated parameters
#'
validate_local_embedding_params <- function(device, batch.size, pooling.strategy, max.length) {

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

  # Validate batch.size
  if (!is.numeric(batch.size) || length(batch.size) != 1 || is.na(batch.size)) {
    stop("AI-GENIE LOCAL expects batch.size to be a numeric value.", call. = FALSE)
  }

  batch.size <- as.integer(batch.size)

  if (batch.size < 1) {
    stop("AI-GENIE LOCAL expects batch.size to be at least 1.", call. = FALSE)
  }

  if (batch.size > 128) {
    warning("Large batch.size (>128) may cause memory issues. Consider reducing if problems occur.")
  }

  # Validate pooling.strategy
  valid_strategies <- c("mean", "cls", "max")

  if (!is.character(pooling.strategy) || length(pooling.strategy) != 1 || is.na(pooling.strategy)) {
    stop("AI-GENIE LOCAL expects pooling.strategy to be a string.", call. = FALSE)
  }

  pooling.strategy <- tolower(trimws(pooling.strategy))

  if (!pooling.strategy %in% valid_strategies) {
    stop(
      paste0(
        "AI-GENIE LOCAL expects pooling.strategy to be one of: ",
        paste(sprintf("'%s'", valid_strategies), collapse = ", "),
        ". Received: '", pooling.strategy, "'"
      ),
      call. = FALSE
    )
  }

  # Validate max.length
  if (!is.numeric(max.length) || length(max.length) != 1 || is.na(max.length)) {
    stop("AI-GENIE LOCAL expects max.length to be a numeric value.", call. = FALSE)
  }

  max.length <- as.integer(max.length)

  if (max.length < 10) {
    stop("AI-GENIE LOCAL expects max.length to be at least 10.", call. = FALSE)
  }

  if (max.length > 512) {
    warning(
      paste0(
        "max.length > 512 exceeds typical BERT model limits. ",
        "Most models max out at 512 tokens. Setting to 512."
      )
    )
    max.length <- 512
  }

  return(list(
    device = device,
    batch.size = batch.size,
    pooling.strategy = pooling.strategy,
    max.length = max.length
  ))
}
