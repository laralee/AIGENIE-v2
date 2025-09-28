#' Generate Embeddings Using HuggingFace Models (Enhanced)
#'
#' @description
#' Generates embeddings for text items using HuggingFace models with automatic
#' fallback to sentence-transformers library for models not supported by the
#' Inference API. Handles gated models, authentication, and various model types.
#'
#' @param embedding.model Character string specifying the HuggingFace model.
#'   Supported models include:
#'   \itemize{
#'     \item BAAI/bge series (API-compatible)
#'     \item sentence-transformers models (via library)
#'     \item Google EmbeddingGemma models (requires authentication)
#'     \item GTE series models (via library)
#'   }
#' @param hf.token Optional character string. HuggingFace API token for
#'   authentication and increased rate limits.
#' @param items Data frame containing the items to embed. Must have:
#'   \itemize{
#'     \item \code{statement}: Character vector of text to embed
#'     \item \code{ID}: Unique identifiers for each statement
#'   }
#' @param silently Logical. If FALSE, displays progress messages.
#'
#' @return A list with two elements:
#'   \itemize{
#'     \item \code{embeddings}: Numeric matrix where columns are items
#'           and rows are embedding dimensions. NULL if embedding fails.
#'     \item \code{success}: Logical indicating whether embedding was successful.
#'   }
#'
#' @details
#' The function automatically determines the best approach for each model:
#' \enumerate{
#'   \item Tries the Inference API for known compatible models
#'   \item Falls back to sentence-transformers library for unsupported models
#'   \item Handles authentication for gated models
#'   \item Uses memory-efficient loading for large models
#' }
#'
embed_items_huggingface_enhanced <- function(embedding.model, hf.token = NULL, items, silently = FALSE) {

  ensure_aigenie_python()

  if(!silently){
    cat("\nGenerating embeddings with", embedding.model, "...")
  }

  # Models confirmed to work via API
  api_compatible_models <- c(
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5"
  )

  # Models that need sentence-transformers
  needs_sentence_transformers <- c(
    "google/embeddinggemma-300m",
    "google/embeddinggemma-256m",
    "google/embeddinggemma-128m",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "thenlper/gte-small",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2"
  )

  # Detect model characteristics
  is_large_model <- grepl("7B|8B|13B|70B", embedding.model, ignore.case = TRUE)
  is_sentence_transformer <- grepl("sentence-transformers/", embedding.model)

  # Route to appropriate method
  use_st_directly <- embedding.model %in% needs_sentence_transformers ||
    is_sentence_transformer ||
    is_large_model

  if (use_st_directly) {
    if (!silently) {
      cat("\nUsing sentence-transformers library...")
    }
    return(embed_items_via_sentence_transformers(embedding.model, items, hf.token, silently))
  }

  # Try API first for other models
  api_result <- tryCatch({
    embed_items_huggingface(embedding.model, hf.token, items, TRUE)
  }, error = function(e) {
    list(success = FALSE)
  })

  if (api_result$success) {
    if (!silently) {
      cat(" Done.\n\n")
    }
    return(api_result)
  }

  # Fallback to sentence-transformers
  if (!silently) {
    cat("\nAPI approach failed. Trying sentence-transformers library...")
  }

  st_result <- embed_items_via_sentence_transformers(embedding.model, items, hf.token, silently)

  if (!st_result$success && !silently) {
    cat("\nBoth approaches failed. Model may not be compatible.\n")
  }

  return(st_result)
}







#' Generate Embeddings Using Sentence-Transformers Library
#'
#' @description
#' Backend function that uses the sentence-transformers Python library to
#' generate embeddings. Handles authentication for gated models and supports
#' various encoder architectures.
#'
#' @param embedding.model Character string specifying the model name.
#' @param items Data frame with \code{statement} and \code{ID} columns.
#' @param hf.token Optional HuggingFace token for gated models.
#' @param silently Logical. Suppress progress messages if TRUE.
#'
#' @return List with \code{embeddings} matrix and \code{success} flag.
#'
embed_items_via_sentence_transformers <- function(embedding.model, items, hf.token = NULL, silently = FALSE) {

  # Set tokenizers parallelism to false to prevent warnings
  Sys.setenv(TOKENIZERS_PARALLELISM = "false")

  result <- list(
    embeddings = NULL,
    success = FALSE
  )

  # Clear potentially problematic environment variables
  clear_pytorch_environment()

  # Try loading model, with automatic CPU fallback on MPS errors
  attempt_count <- 0
  max_attempts <- 2

  while (!result$success && attempt_count < max_attempts) {
    attempt_count <- attempt_count + 1

    tryCatch({
      # Ensure sentence-transformers is installed
      if (!reticulate::py_module_available("sentence_transformers")) {
        if (!silently) {
          cat("\nInstalling sentence-transformers library...")
        }

      # Suppress the ephemeral environment warning, its ok it will still install
      suppressWarnings({
          reticulate::py_install("sentence-transformers", pip = TRUE)
      })
      }

      # Authenticate if token provided
      if (!is.null(hf.token) && attempt_count == 1) {
        if (!silently) {
          cat("\nAuthenticating with HuggingFace...")
        }
        huggingface_hub <- reticulate::import("huggingface_hub")
        huggingface_hub$login(token = hf.token, add_to_git_credential = FALSE)
      }

      if (!silently) {
        if (attempt_count == 1) {
          cat("\nLoading model...")
        } else {
          cat("\nRetrying with CPU...")
        }
      }

      # Device selection based on attempt
      force_cpu <- (attempt_count > 1)

      py_code <- sprintf('
from sentence_transformers import SentenceTransformer
import os
import torch

# Clear any problematic environment variables
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
    del os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]

# Set token if provided
%s

# Determine device
force_cpu = %s
if force_cpu:
    device = "cpu"
else:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        # Use CPU for known problematic models on MPS
        if any(x in "%s".lower() for x in ["embeddinggemma", "qwen", "8b", "7b"]):
            device = "cpu"
        else:
            device = "mps"
    else:
        device = "cpu"

# Load model
try:
    model = SentenceTransformer("%s", device=device, token=True)
except TypeError:
    # Fallback for older versions
    try:
        model = SentenceTransformer("%s", device=device, use_auth_token=True)
    except:
        model = SentenceTransformer("%s", use_auth_token=True)
',
                         ifelse(!is.null(hf.token),
                                sprintf('os.environ["HF_TOKEN"] = "%s"', hf.token), ""),
                         ifelse(force_cpu, "True", "False"),
                         embedding.model,
                         embedding.model,
                         embedding.model,
                         embedding.model
      )

      reticulate::py_run_string(py_code)
      py_main <- reticulate::import_main()
      model <- py_main$model

      # Generate embeddings
      statements <- items$statement
      item_ids <- items$ID

      if (!silently) {
        cat("\nGenerating embeddings for", length(statements), "items...")
      }

      # Check for specialized encoding methods
      builtins <- reticulate::import_builtins()
      if (builtins$hasattr(model, "encode_document")) {
        embeddings_array <- model$encode_document(statements)
      } else {
        embeddings_array <- model$encode(statements)
      }

      # Convert to R matrix
      embeddings_matrix <- t(reticulate::py_to_r(embeddings_array))
      colnames(embeddings_matrix) <- item_ids

      result$embeddings <- embeddings_matrix
      result$success <- TRUE

      if(!silently){
        cat(" Done.\n")
        if (attempt_count > 1) {
          cat("(Used CPU due to memory constraints)\n")
        }
        cat("Generated embeddings with dimensions:", dim(embeddings_matrix), "\n\n")
      }

    }, error = function(e) {
      error_msg <- conditionMessage(e)

      # Check if this is an MPS memory error
      if (grepl("low watermark ratio|MPS backend", error_msg) && attempt_count == 1) {
        if (!silently) {
          cat("\nMPS memory issue detected. Retrying with CPU...\n")
        }
        # Clear environment again for safety
        clear_pytorch_environment()
        # Continue to next attempt
      } else {
        # Other error or second attempt failed
        if (!silently) {
          cat("\nError:", error_msg, "\n")

          if (grepl("gated repo|401 Client Error", error_msg)) {
            cat("\nThis model requires authentication. Please:\n")
            cat("1. Provide a valid HuggingFace token\n")
            cat("2. Accept the model's license at https://huggingface.co/", embedding.model, "\n")
          } else if (grepl("out of memory|OOM", error_msg)) {
            cat("\nThis model is too large for your system.\n")
            cat("Consider using a smaller alternative.\n")
          }
        }
        result$success <- FALSE
        break  # Exit the while loop
      }
    })
  }

  return(result)
}

# Helper function to clear PyTorch environment variables
clear_pytorch_environment <- function() {
  # Clear R environment variables
  Sys.unsetenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
  Sys.setenv(TOKENIZERS_PARALLELISM = "false")  # Add this to prevent tokenizer warnings

  # Clear Python environment variables if Python is initialized
  if (reticulate::py_available()) {
    tryCatch({
      reticulate::py_run_string('
import os
# Clear MPS memory settings
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
    del os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]
# Set tokenizers parallelism to false to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
')
    }, error = function(e) {
      # Silently continue if this fails
    })
  }
}

