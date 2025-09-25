#' Generate Raw Embeddings Using Local Models
#'
#' @description
#' Generates raw feature embeddings for text items using locally installed
#' transformer models. These are raw embeddings from the model's encoder,
#' not similarity-optimized embeddings.
#'
#' @param embedding.model Character string specifying either:
#'   - A HuggingFace model identifier for BERT-style models:
#'     - "bert-base-uncased" (768 dims)
#'     - "roberta-base" (768 dims)
#'     - "microsoft/deberta-v3-base" (768 dims)
#'     - "BAAI/bge-base-en-v1.5" (768 dims) - if loaded for raw extraction
#'   - A local path to a saved model
#'
#' @param items Data frame containing the items to embed. Must have two columns:
#'   \itemize{
#'     \item \code{statement}: Character vector of text to embed
#'     \item \code{ID}: Unique identifiers for each statement
#'   }
#'
#' @param pooling.strategy Character string for pooling token embeddings:
#'   - "mean": Average all token embeddings (default)
#'   - "cls": Use only the CLS token embedding
#'   - "max": Max pooling across tokens
#'
#' @param device Character string specifying computation device
#' @param batch.size Integer. Number of items to process simultaneously
#' @param max.length Integer. Maximum sequence length (default 512)
#' @param silently Logical. If FALSE, displays progress messages
#'
#' @return A list with embeddings matrix and success flag
#'
#' Generate Raw Embeddings Using Local Models
#'
#' @description
#' Generates raw feature embeddings for text items using locally installed
#' transformer models. These are raw embeddings from the model's encoder,
#' not similarity-optimized embeddings.
#'
#' @param embedding.model Character string specifying either:
#'   - A HuggingFace model identifier for BERT-style models:
#'     - "bert-base-uncased" (768 dims)
#'     - "roberta-base" (768 dims)
#'     - "microsoft/deberta-v3-base" (768 dims)
#'     - "BAAI/bge-base-en-v1.5" (768 dims) - if loaded for raw extraction
#'   - A local path to a saved model
#'
#' @param items Data frame containing the items to embed. Must have two columns:
#'   \itemize{
#'     \item \code{statement}: Character vector of text to embed
#'     \item \code{ID}: Unique identifiers for each statement
#'   }
#'
#' @param pooling.strategy Character string for pooling token embeddings:
#'   - "mean": Average all token embeddings (default)
#'   - "cls": Use only the CLS token embedding
#'   - "max": Max pooling across tokens
#'
#' @param device Character string specifying computation device
#' @param batch.size Integer. Number of items to process simultaneously
#' @param max.length Integer. Maximum sequence length (default 512)
#' @param silently Logical. If FALSE, displays progress messages
#'
#' @return A list with embeddings matrix and success flag
#'
#' Generate Raw Embeddings Using Local Models
#'
#' @description
#' Generates raw feature embeddings for text items using locally installed
#' transformer models. These are raw embeddings from the model's encoder,
#' not similarity-optimized embeddings.
#'
#' @param embedding.model Character string specifying either:
#'   - A HuggingFace model identifier for BERT-style models:
#'     - "bert-base-uncased" (768 dims)
#'     - "roberta-base" (768 dims)
#'     - "microsoft/deberta-v3-base" (768 dims)
#'     - "BAAI/bge-base-en-v1.5" (768 dims) - if loaded for raw extraction
#'   - A local path to a saved model
#'
#' @param items Data frame containing the items to embed. Must have two columns:
#'   \itemize{
#'     \item \code{statement}: Character vector of text to embed
#'     \item \code{ID}: Unique identifiers for each statement
#'   }
#'
#' @param pooling.strategy Character string for pooling token embeddings:
#'   - "mean": Average all token embeddings (default)
#'   - "cls": Use only the CLS token embedding
#'   - "max": Max pooling across tokens
#'
#' @param device Character string specifying computation device
#' @param batch.size Integer. Number of items to process simultaneously
#' @param max.length Integer. Maximum sequence length (default 512)
#' @param silently Logical. If FALSE, displays progress messages
#'
#' @return A list with embeddings matrix and success flag
#'
embed_items_local <- function(embedding.model,
                              items,
                              pooling.strategy = "mean",
                              device = "auto",
                              batch.size = 32,
                              max.length = 512,
                              silently = FALSE) {

  # Set tokenizers parallelism to avoid forking warnings
  Sys.setenv(TOKENIZERS_PARALLELISM = "false")

  # Ensure Python environment is ready
  ensure_aigenie_python_local(silently = silently)

  if(!silently){
    cat("\n")
    cat("Generating raw embeddings locally...")
  }

  # Initialize return structure
  result <- list(
    embeddings = NULL,
    success = FALSE
  )

  tryCatch({
    # Import transformers directly for raw embeddings
    transformers <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    np <- reticulate::import("numpy")

    # Determine device
    if (device == "auto") {
      if (torch$cuda$is_available()) {
        device <- "cuda"
        if (!silently) cat("\n  Using GPU (CUDA)...")
      } else if (torch$backends$mps$is_available()) {
        device <- "mps"
        if (!silently) cat("\n  Using GPU (Apple Silicon)...")
      } else {
        device <- "cpu"
        if (!silently) cat("\n  Using CPU...")
      }
    }

    # Load tokenizer and model for raw embeddings
    if (!silently) cat("\n  Loading model and tokenizer...")

    tokenizer <- transformers$AutoTokenizer$from_pretrained(embedding.model)
    model <- transformers$AutoModel$from_pretrained(embedding.model)

    # Move model to device
    model <- model$to(device)
    model$eval()  # Set to evaluation mode

    # Extract statements
    statements <- items$statement
    item_ids <- items$ID
    n_items <- length(statements)

    if (!silently) cat(paste0("\n  Processing ", n_items, " items"))

    # Generate embeddings in batches
    all_embeddings <- list()
    n_batches <- ceiling(n_items / batch.size)

    for (i in seq_len(n_batches)) {
      start_idx <- (i - 1) * batch.size + 1
      end_idx <- min(i * batch.size, n_items)
      batch_statements <- statements[start_idx:end_idx]

      # Tokenize batch
      inputs <- tokenizer(
        batch_statements,
        padding = TRUE,
        truncation = TRUE,
        max_length = as.integer(max.length),
        return_tensors = "pt"
      )

      # Move inputs to device
      if (device != "cpu") {
        input_ids <- inputs$input_ids$to(device)
        attention_mask <- inputs$attention_mask$to(device)
      } else {
        input_ids <- inputs$input_ids
        attention_mask <- inputs$attention_mask
      }

      # Get model outputs - call directly with positional arguments
      with(torch$no_grad(), {
        # Call model directly with input_ids and attention_mask
        outputs <- model(input_ids = input_ids, attention_mask = attention_mask)
      })

      # Extract hidden states (last layer)
      hidden_states <- outputs$last_hidden_state

      # Apply pooling strategy
      if (pooling.strategy == "mean") {
        # Mean pooling - average all token embeddings
        # Expand mask to match hidden states shape
        mask_expanded <- attention_mask$unsqueeze(-1L)$expand_as(hidden_states)

        # Sum embeddings where mask is 1
        sum_embeddings <- torch$sum(hidden_states * mask_expanded, dim = 1L)

        # Count of non-masked tokens
        sum_mask <- torch$clamp(mask_expanded$sum(dim = 1L), min = 1e-9)

        # Average
        embeddings <- sum_embeddings / sum_mask

      } else if (pooling.strategy == "cls") {
        # Use CLS token embedding (first token)
        # Python uses 0-based indexing, : means all in that dimension
        embeddings <- hidden_states[reticulate::py_slice(NULL, NULL), 0L, reticulate::py_slice(NULL, NULL)]

      } else if (pooling.strategy == "max") {
        # Max pooling
        # Need to mask out padding tokens for max pooling
        mask_expanded <- attention_mask$unsqueeze(-1L)$expand_as(hidden_states)

        # Set padding tokens to very negative value before max
        hidden_states_masked <- hidden_states$clone()
        hidden_states_masked[mask_expanded == 0] <- -1e9
        max_result <- torch$max(hidden_states_masked, dim = 1L)
        embeddings <- max_result$values
      }

      # Convert to numpy then R
      embeddings_np <- embeddings$cpu()$numpy()
      batch_matrix <- reticulate::py_to_r(embeddings_np)

      # Store embeddings
      for (j in seq_len(nrow(batch_matrix))) {
        all_embeddings[[start_idx + j - 1]] <- batch_matrix[j, ]
      }

      if (!silently && i %% 5 == 0) {
        cat(".")
      }
    }

    # Combine into matrix (columns = items, rows = dimensions)
    embedding_matrix <- do.call(cbind, all_embeddings)
    colnames(embedding_matrix) <- item_ids

    # Update result
    result$embeddings <- embedding_matrix
    result$success <- TRUE

    if(!silently){
      cat(" Done.\n")
      cat(paste0("  Generated ", ncol(embedding_matrix), " raw embeddings of dimension ",
                 nrow(embedding_matrix), "\n\n"))
    }

  }, error = function(e) {
    cat("\nError occurred during local embedding:\n")
    cat("Error message:", conditionMessage(e), "\n")

    # Provide helpful error messages
    if (grepl("No module named 'transformers'", conditionMessage(e))) {
      cat("\nTransformers not installed. Please run:\n")
      cat("  install_local_dependencies()\n")
    } else if (grepl("CUDA|cuda", conditionMessage(e))) {
      cat("\nGPU error detected. Try setting device='cpu' to use CPU instead.\n")
    } else if (grepl("model", conditionMessage(e), ignore.case = TRUE)) {
      cat("\nModel loading error. Check that the model name is correct.\n")
      cat("Example: 'bert-base-uncased' or 'roberta-base'\n")
    }

    result$success <- FALSE
  })

  return(result)
}




#' Ensure Python Environment for Local Models
#'
#' Sets up Python environment with packages needed for local raw embeddings.
#'
#' @param force Logical. Force reinstallation of packages even if they exist.
#' @param silently Logical. If TRUE, suppress all messages except errors.
#'
ensure_aigenie_python_local <- function(force = FALSE, silently = FALSE) {

  # Check if already initialized (unless forcing)
  if (!force && isTRUE(getOption("aigenie.python_local_initialized", FALSE))) {
    return(invisible(TRUE))
  }

  if (!silently) {
    message("AI-GENIE LOCAL: Checking Python environment for local models...")
  }

  # Configure Python
  reticulate::py_config()

  # Check and install required packages for raw embeddings
  required_packages <- list(
    transformers = "transformers",
    torch = "torch",
    numpy = "numpy"
  )

  for (pkg_import in names(required_packages)) {
    pkg_install <- required_packages[[pkg_import]]

    if (!reticulate::py_module_available(pkg_import) || force) {
      if (!silently) {
        message(paste0("Installing ", pkg_install, "..."))
      }
      reticulate::py_install(pkg_install, pip = TRUE)
    }
  }

  # Initialize Python
  reticulate::py_available(initialize = TRUE)

  # Verify installations
  tryCatch({
    transformers <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    numpy <- reticulate::import("numpy")

    if (!silently) {
      message("AI-GENIE LOCAL: Python environment ready for raw embeddings!")

      # Check for GPU availability
      if (torch$cuda$is_available()) {
        message(paste0("  GPU detected: ", torch$cuda$get_device_name(0L)))
      } else if (torch$backends$mps$is_available()) {
        message("  Apple Silicon GPU detected")
      } else {
        message("  No GPU detected - will use CPU")
      }
    }

  }, error = function(e) {
    stop("Failed to set up Python packages for local models. Error: ",
         conditionMessage(e))
  })

  # Mark as initialized
  options(aigenie.python_local_initialized = TRUE)

  invisible(TRUE)
}

