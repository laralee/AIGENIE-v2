# embeddings.R
# Unified Embeddings Module for AI-GENIE
# ============================================================================
# This file provides a unified interface for generating embeddings across
# multiple providers: OpenAI, Jina AI, HuggingFace (API and local), and
# local models.

# ============================================================================
# Provider Detection for Embedding Models
# ============================================================================

#' Validate and Detect Embedding Model Provider
#' 
#' @description
#' Determines which provider to use for embeddings based on the model name.
#' 
#' @param embedding.model Character string specifying the embedding model
#' 
#' @return Character string: "openai", "jina", "huggingface", or "local"
#' @keywords internal
embedding.model_validate <- function(embedding.model) {
  
  # OpenAI embedding models
  openai_models <- c(
    "text-embedding-3-small",
    "text-embedding-3-large", 
    "text-embedding-ada-002"
  )
  
  # Jina AI embedding models
  jina_models <- c(
    "jina-embeddings-v4",
    "jina-embeddings-v3",
    "jina-clip-v2",
    "jina-code-embeddings-1.5b",
    "jina-code-embeddings-0.5b",
    "jina-embeddings-v2-base-en",
    "jina-embeddings-v2-base-zh",
    "jina-embeddings-v2-base-de",
    "jina-embeddings-v2-base-es",
    "jina-embeddings-v2-base-code",
    "jina-embeddings-v2-small-en"
  )
  
  # Known HuggingFace API-compatible models
  hf_api_models <- c(
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-small",
    "thenlper/gte-base",
    "thenlper/gte-large"
  )
  
  # Models that need sentence-transformers library
  sentence_transformer_models <- c(
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "google/embeddinggemma-300m",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2"
  )
  
  # Local raw embedding models (BERT family)
  local_models <- c(
    "bert-base-uncased",
    "bert-base-cased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "distilbert-base-uncased",
    "microsoft/deberta-v3-base",
    "albert-base-v2"
  )
  
  model_lower <- tolower(trimws(embedding.model))
  
  # Check OpenAI
  if (embedding.model %in% openai_models) {
    return("openai")
  }
  
  # Check Jina (exact match or prefix pattern)
  if (embedding.model %in% jina_models || grepl("^jina-", model_lower)) {
    return("jina")
  }
  
  # Check HuggingFace (both API and sentence-transformers)
  if (embedding.model %in% hf_api_models ||
      embedding.model %in% sentence_transformer_models ||
      grepl("^BAAI/|^thenlper/|^sentence-transformers/|^intfloat/|^google/", embedding.model)) {
    return("huggingface")
  }
  
  # Check local models
  if (embedding.model %in% local_models ||
      grepl("^bert|^roberta|^distilbert|^albert|^deberta", model_lower)) {
    return("local")
  }
  
  # If contains "/" assume HuggingFace
  if (grepl("/", embedding.model)) {
    warning("Unknown embedding model '", embedding.model, "'. ",
            "Assuming HuggingFace model.", call. = FALSE, immediate. = TRUE)
    return("huggingface")
  }
  
  # Default to local for unknown models
  warning("Unknown embedding model '", embedding.model, "'. ",
          "Will attempt to load as a local transformer model.", 
          call. = FALSE, immediate. = TRUE)
  return("local")
}

# ============================================================================
# Unified Embedding Interface
# ============================================================================

#' Generate Embeddings Using Any Supported Provider
#' 
#' @description
#' Unified interface for generating embeddings that automatically routes to
#' the appropriate provider based on the model name.
#' 
#' @param embedding.model Character string specifying the embedding model
#' @param items Data frame with 'statement' and 'ID' columns
#' @param openai.API Optional OpenAI API key
#' @param hf.token Optional HuggingFace token
#' @param jina.API Optional Jina AI API key
#' @param silently Logical. Suppress progress messages?
#' @param ... Additional arguments passed to provider-specific functions
#' 
#' @return A list with 'embeddings' matrix and 'success' flag
#' @keywords internal
generate_embeddings <- function(embedding.model,
                                 items,
                                 openai.API = NULL,
                                 hf.token = NULL,
                                 jina.API = NULL,
                                 silently = FALSE,
                                 ...) {
  
  provider <- embedding.model_validate(embedding.model)
  
  if (provider == "openai") {
    if (is.null(openai.API)) {
      stop("OpenAI embedding model requires openai.API key.", call. = FALSE)
    }
    return(embed_items(embedding.model, openai.API, items, silently))
    
  } else if (provider == "jina") {
    if (is.null(jina.API)) {
      jina.API <- Sys.getenv("JINA_API_KEY", unset = "")
      if (nchar(jina.API) == 0) {
        stop("Jina embedding model requires jina.API key or ",
             "JINA_API_KEY environment variable.", call. = FALSE)
      }
    }
    args <- list(...)
    return(embed_items_jina(
      embedding.model = embedding.model,
      jina_api_key = jina.API,
      items = items,
      task = args$task %||% "text-matching",
      dimensions = args$dimensions,
      silently = silently
    ))
    
  } else if (provider == "huggingface") {
    return(embed_items_huggingface(embedding.model, hf.token, items, silently))
    
  } else if (provider == "local") {
    # Extract additional parameters for local embedding
    args <- list(...)
    return(embed_items_local(
      embedding.model = embedding.model,
      items = items,
      pooling.strategy = args$pooling.strategy %||% "mean",
      device = args$device %||% "auto",
      batch.size = args$batch.size %||% 32L,
      max.length = args$max.length %||% 512L,
      silently = silently
    ))
  }
  
  stop("Unknown embedding provider: ", provider, call. = FALSE)
}

# ============================================================================
# OpenAI Embeddings
# ============================================================================

#' Embed Items Using OpenAI's Embedding API
#' 
#' @description
#' Generates embeddings using OpenAI's embedding models.
#' 
#' @param embedding.model OpenAI embedding model name
#' @param openai.API OpenAI API key
#' @param items Data frame with 'statement' and 'ID' columns
#' @param silently Logical. Suppress progress messages?
#' 
#' @return A list with 'embeddings' matrix and 'success' flag
#' @keywords internal
embed_items <- function(embedding.model, openai.API, items, silently) {
  
  ensure_aigenie_python()
  
  if (!silently) {
    cat("\nGenerating embeddings with OpenAI (", embedding.model, ")...")
  }
  
  result <- list(embeddings = NULL, success = FALSE)
  
  tryCatch({
    openai <- reticulate::import("openai")
    openai$api_key <- openai.API
    
    statements <- items$statement
    item_ids <- items$ID
    all_embeddings <- list()
    
    # Process in batches for efficiency
    batch_size <- 100
    n_batches <- ceiling(length(statements) / batch_size)
    
    for (b in seq_len(n_batches)) {
      start_idx <- (b - 1) * batch_size + 1
      end_idx <- min(b * batch_size, length(statements))
      batch_statements <- statements[start_idx:end_idx]
      
      # Create embeddings for batch
      response <- openai$Embedding$create(
        input = as.list(batch_statements),
        model = embedding.model
      )
      
      # Extract embeddings
      for (i in seq_along(response$data)) {
        all_embeddings[[start_idx + i - 1]] <- unlist(response$data[[i]]$embedding)
      }
      
      if (!silently && n_batches > 1) {
        cat(".")
      }
    }
    
    # Combine into matrix (rows = dimensions, columns = items)
    embedding_matrix <- do.call(cbind, all_embeddings)
    colnames(embedding_matrix) <- item_ids
    
    result$embeddings <- embedding_matrix
    result$success <- TRUE
    
    if (!silently) {
      cat(" Done.\n")
      cat("Generated", ncol(embedding_matrix), "embeddings with",
          nrow(embedding_matrix), "dimensions.\n\n")
    }
    
  }, error = function(e) {
    cat("\nError during OpenAI embedding:", conditionMessage(e), "\n")
    result$success <- FALSE
  })
  
  return(result)
}

# ============================================================================
# Jina AI Embeddings
# ============================================================================

#' Embed Items Using Jina AI Embedding API
#' 
#' @description
#' Generates embeddings using Jina AI's embedding models via their REST API
#' at \code{https://api.jina.ai/v1/embeddings}. Uses Bearer token auth and
#' supports Jina-specific features: task adapters, Matryoshka dimension
#' truncation, and late chunking.
#' 
#' The Jina API follows an OpenAI-compatible request/response schema, with
#' additional parameters for task type and output dimensions.
#' 
#' @param embedding.model Jina embedding model name (e.g., "jina-embeddings-v3")
#' @param jina_api_key Jina AI API key
#' @param items Data frame with 'statement' and 'ID' columns
#' @param task Character. Task adapter for optimized embeddings. One of:
#'   \describe{
#'     \item{"text-matching"}{Sentence similarity (default for AIGENIE)}
#'     \item{"retrieval.query"}{Encode queries for retrieval}
#'     \item{"retrieval.passage"}{Encode passages for indexing}
#'     \item{"classification"}{Text classification}
#'     \item{"separation"}{Clustering or reranking}
#'   }
#' @param dimensions Optional integer. Output embedding dimensions for
#'   Matryoshka-capable models (v3: 256-1024, v4: 128-2048). NULL uses the
#'   model default (v3: 1024, v4: 2048).
#' @param silently Logical. Suppress progress messages?
#' 
#' @return A list with 'embeddings' matrix and 'success' flag
#' @keywords internal
embed_items_jina <- function(embedding.model = "jina-embeddings-v3",
                              jina_api_key,
                              items,
                              task = "text-matching",
                              dimensions = NULL,
                              silently = FALSE) {
  
  ensure_aigenie_python()
  
  if (!silently) {
    cat("\nGenerating embeddings with Jina AI (", embedding.model, ")...")
    if (!is.null(dimensions)) {
      cat(" [", dimensions, "d]")
    }
    cat(" task:", task)
  }
  
  json_mod <- reticulate::import("json")
  requests <- reticulate::import("requests")
  
  result <- list(embeddings = NULL, success = FALSE)
  
  tryCatch({
    
    statements <- items$statement
    item_ids <- items$ID
    all_embeddings <- list()
    
    # Jina has no hard batch limit but we batch for memory efficiency
    # and to stay within rate limits (100 RPM free, 500 RPM paid)
    batch_size <- 100
    n_batches <- ceiling(length(statements) / batch_size)
    
    headers <- list(
      "Authorization" = paste("Bearer", jina_api_key),
      "Content-Type"  = "application/json"
    )
    
    for (b in seq_len(n_batches)) {
      start_idx <- (b - 1) * batch_size + 1
      end_idx <- min(b * batch_size, length(statements))
      batch_statements <- as.list(statements[start_idx:end_idx])
      
      # Build request body
      body <- list(
        model = embedding.model,
        input = batch_statements,
        task  = task
      )
      
      # Add optional Matryoshka dimensions
      if (!is.null(dimensions)) {
        body$dimensions <- as.integer(dimensions)
      }
      
      # Serialize via Python json for proper list handling
      body_json <- json_mod$dumps(body)
      
      # Retry logic for rate limits and transient errors
      max_retries <- 5
      for (attempt in seq_len(max_retries)) {
        response <- requests$post(
          "https://api.jina.ai/v1/embeddings",
          headers = headers,
          data = body_json
        )
        
        if (response$status_code == 200L) {
          resp_data <- response$json()
          
          # Response format: {data: [{embedding: [...], index: N}, ...]}
          for (i in seq_along(resp_data$data)) {
            emb <- unlist(resp_data$data[[i]]$embedding)
            all_embeddings[[start_idx + i - 1]] <- emb
          }
          break
          
        } else if (response$status_code == 429L) {
          # Rate limited — back off
          if (!silently) cat(" [rate limited, waiting]")
          Sys.sleep(30 * attempt)
        } else if (response$status_code == 503L) {
          # Service temporarily unavailable
          Sys.sleep(10 * attempt)
        } else {
          if (attempt == max_retries) {
            stop("Jina AI API error (", response$status_code, "): ",
                 response$text, call. = FALSE)
          }
          Sys.sleep(5 * attempt)
        }
      }
      
      if (!silently && n_batches > 1) {
        cat(".")
      }
    }
    
    # Combine into matrix (rows = dimensions, columns = items)
    embedding_matrix <- do.call(cbind, all_embeddings)
    colnames(embedding_matrix) <- item_ids
    
    result$embeddings <- embedding_matrix
    result$success <- TRUE
    
    if (!silently) {
      cat(" Done.\n")
      cat("Generated", ncol(embedding_matrix), "embeddings with",
          nrow(embedding_matrix), "dimensions.\n\n")
    }
    
  }, error = function(e) {
    cat("\nError during Jina AI embedding:", conditionMessage(e), "\n")
    result$success <- FALSE
  })
  
  return(result)
}

# ============================================================================
# HuggingFace Embeddings (Enhanced with fallback)
# ============================================================================

#' Embed Items Using HuggingFace Models
#' 
#' @description
#' Generates embeddings using HuggingFace models. Tries the Inference API first,
#' then falls back to the sentence-transformers library for unsupported models.
#' 
#' @param embedding.model HuggingFace model name
#' @param hf.token Optional HuggingFace API token
#' @param items Data frame with 'statement' and 'ID' columns
#' @param silently Logical. Suppress progress messages?
#' 
#' @return A list with 'embeddings' matrix and 'success' flag
#' @keywords internal
embed_items_huggingface <- function(embedding.model = "BAAI/bge-base-en-v1.5",
                                     hf.token = NULL, 
                                     items, 
                                     silently = FALSE) {
  
  ensure_aigenie_python()
  
  if (!silently) {
    cat("\nGenerating embeddings with HuggingFace (", embedding.model, ")...")
  }
  
  # Models that definitely need sentence-transformers
  needs_sentence_transformers <- c(
    "google/embeddinggemma-300m",
    "google/embeddinggemma-256m",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "thenlper/gte-small",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2"
  )
  
  # Check if model needs sentence-transformers
  is_large_model <- grepl("7B|8B|13B|70B", embedding.model, ignore.case = TRUE)
  use_st <- embedding.model %in% needs_sentence_transformers ||
    grepl("^sentence-transformers/", embedding.model) ||
    is_large_model
  
  if (use_st) {
    if (!silently) {
      cat("\nUsing sentence-transformers library...")
    }
    return(embed_items_via_sentence_transformers(embedding.model, items, hf.token, silently))
  }
  
  # Try API first for other models
  api_result <- tryCatch({
    embed_items_huggingface_api(embedding.model, hf.token, items, silently = TRUE)
  }, error = function(e) {
    list(success = FALSE)
  })
  
  if (api_result$success) {
    if (!silently) {
      cat(" Done.\n")
      cat("Generated", ncol(api_result$embeddings), "embeddings with",
          nrow(api_result$embeddings), "dimensions.\n\n")
    }
    return(api_result)
  }
  
  # Fallback to sentence-transformers
  if (!silently) {
    cat("\nAPI unavailable. Trying sentence-transformers...")
  }
  
  return(embed_items_via_sentence_transformers(embedding.model, items, hf.token, silently))
}

#' Embed Items Using HuggingFace Inference API
#' 
#' @keywords internal
embed_items_huggingface_api <- function(embedding.model, hf.token, items, silently = FALSE) {
  
  requests <- reticulate::import("requests")
  
  result <- list(embeddings = NULL, success = FALSE)
  
  api_url <- paste0("https://api-inference.huggingface.co/pipeline/feature-extraction/", 
                    embedding.model)
  
  headers <- list("Content-Type" = "application/json")
  if (!is.null(hf.token)) {
    headers[["Authorization"]] <- paste("Bearer", hf.token)
  }
  
  statements <- items$statement
  item_ids <- items$ID
  all_embeddings <- list()
  
  # Process in batches
  batch_size <- 50
  n_batches <- ceiling(length(statements) / batch_size)
  
  for (b in seq_len(n_batches)) {
    start_idx <- (b - 1) * batch_size + 1
    end_idx <- min(b * batch_size, length(statements))
    batch_statements <- as.list(statements[start_idx:end_idx])
    
    # Retry logic
    max_retries <- 5
    for (attempt in seq_len(max_retries)) {
      response <- requests$post(
        api_url,
        headers = headers,
        json = list(inputs = batch_statements, options = list(wait_for_model = TRUE))
      )
      
      if (response$status_code == 200L) {
        embeddings_list <- response$json()
        
        for (i in seq_along(embeddings_list)) {
          emb <- embeddings_list[[i]]
          # Handle nested arrays (mean pooling if needed)
          if (is.list(emb) && length(emb) > 1 && is.list(emb[[1]])) {
            emb <- colMeans(do.call(rbind, lapply(emb, unlist)))
          } else {
            emb <- unlist(emb)
          }
          all_embeddings[[start_idx + i - 1]] <- emb
        }
        break
        
      } else if (response$status_code == 503L) {
        # Model loading
        Sys.sleep(10 * attempt)
      } else if (response$status_code == 429L) {
        # Rate limited
        Sys.sleep(30)
      } else {
        if (attempt == max_retries) {
          stop("HuggingFace API error: ", response$status_code)
        }
        Sys.sleep(5 * attempt)
      }
    }
  }
  
  # Combine into matrix
  embedding_matrix <- do.call(cbind, all_embeddings)
  colnames(embedding_matrix) <- item_ids
  
  result$embeddings <- embedding_matrix
  result$success <- TRUE
  
  return(result)
}

#' Embed Items Using Sentence-Transformers Library
#' 
#' @keywords internal
embed_items_via_sentence_transformers <- function(embedding.model, items, 
                                                   hf.token = NULL, silently = FALSE) {
  
  Sys.setenv(TOKENIZERS_PARALLELISM = "false")
  
  result <- list(embeddings = NULL, success = FALSE)
  
  # Clear problematic environment variables
  clear_pytorch_environment()
  
  attempt_count <- 0
  max_attempts <- 2
  
  while (!result$success && attempt_count < max_attempts) {
    attempt_count <- attempt_count + 1
    
    tryCatch({
      # Check if sentence-transformers is installed
      if (!reticulate::py_module_available("sentence_transformers")) {
        if (!silently) {
          cat("\nInstalling sentence-transformers...")
        }
        # UV will handle installation through our environment
        reticulate::py_install("sentence-transformers", pip = TRUE)
      }
      
      # Authenticate if token provided
      if (!is.null(hf.token) && attempt_count == 1) {
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
      
      force_cpu <- (attempt_count > 1)
      
      # Build Python code for loading model
      py_code <- sprintf('
from sentence_transformers import SentenceTransformer
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

force_cpu = %s
if force_cpu:
    device = "cpu"
else:
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if any(x in "%s".lower() for x in ["embeddinggemma", "qwen", "8b", "7b"]):
            device = "cpu"
        else:
            device = "mps"
    else:
        device = "cpu"

try:
    model = SentenceTransformer("%s", device=device)
except Exception as e:
    model = SentenceTransformer("%s")
',
                         ifelse(force_cpu, "True", "False"),
                         embedding.model,
                         embedding.model,
                         embedding.model
      )
      
      reticulate::py_run_string(py_code)
      py_main <- reticulate::import_main()
      model <- py_main$model
      
      statements <- items$statement
      item_ids <- items$ID
      
      if (!silently) {
        cat("\nGenerating embeddings for", length(statements), "items...")
      }
      
      # Generate embeddings
      embeddings_array <- model$encode(statements)
      
      # Convert to R matrix (transpose: rows = dimensions, cols = items)
      embeddings_matrix <- t(reticulate::py_to_r(embeddings_array))
      colnames(embeddings_matrix) <- item_ids
      
      result$embeddings <- embeddings_matrix
      result$success <- TRUE
      
      if (!silently) {
        cat(" Done.\n")
        cat("Generated", ncol(embeddings_matrix), "embeddings with",
            nrow(embeddings_matrix), "dimensions.\n\n")
      }
      
    }, error = function(e) {
      error_msg <- conditionMessage(e)
      
      if (grepl("low watermark ratio|MPS backend", error_msg) && attempt_count == 1) {
        if (!silently) {
          cat("\nMPS memory issue. Retrying with CPU...")
        }
        clear_pytorch_environment()
      } else {
        if (!silently) {
          cat("\nError:", error_msg, "\n")
        }
        result$success <- FALSE
      }
    })
  }
  
  return(result)
}

# ============================================================================
# Local Embeddings (Raw BERT-family models)
# ============================================================================

#' Embed Items Using Local Transformer Models
#' 
#' @description
#' Generates raw embeddings using locally loaded BERT-family models.
#' These are raw encoder outputs, not similarity-optimized embeddings.
#' 
#' @param embedding.model Character string specifying the model
#' @param items Data frame with 'statement' and 'ID' columns
#' @param pooling.strategy Character. One of "mean", "cls", "max"
#' @param device Character. One of "auto", "cpu", "cuda", "mps"
#' @param batch.size Integer. Batch size for processing
#' @param max.length Integer. Maximum sequence length
#' @param silently Logical. Suppress progress messages?
#' 
#' @return A list with 'embeddings' matrix and 'success' flag
#' @keywords internal
embed_items_local <- function(embedding.model,
                               items,
                               pooling.strategy = "mean",
                               device = "auto",
                               batch.size = 32,
                               max.length = 512,
                               silently = FALSE) {
  
  Sys.setenv(TOKENIZERS_PARALLELISM = "false")
  
  ensure_aigenie_python()
  
  if (!silently) {
    cat("\nGenerating raw embeddings locally (", embedding.model, ")...")
  }
  
  result <- list(embeddings = NULL, success = FALSE)
  
  tryCatch({
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
    
    # Load tokenizer and model
    if (!silently) cat("\n  Loading model...")
    
    tokenizer <- transformers$AutoTokenizer$from_pretrained(embedding.model)
    model <- transformers$AutoModel$from_pretrained(embedding.model)
    model <- model$to(device)
    model$eval()
    
    statements <- items$statement
    item_ids <- items$ID
    n_items <- length(statements)
    
    if (!silently) cat(paste0("\n  Processing ", n_items, " items"))
    
    all_embeddings <- list()
    n_batches <- ceiling(n_items / batch.size)
    
    for (i in seq_len(n_batches)) {
      start_idx <- (i - 1) * batch.size + 1
      end_idx <- min(i * batch.size, n_items)
      batch_statements <- statements[start_idx:end_idx]
      
      # Tokenize
      inputs <- tokenizer(
        batch_statements,
        padding = TRUE,
        truncation = TRUE,
        max_length = as.integer(max.length),
        return_tensors = "pt"
      )
      
      # Move to device
      if (device != "cpu") {
        input_ids <- inputs$input_ids$to(device)
        attention_mask <- inputs$attention_mask$to(device)
      } else {
        input_ids <- inputs$input_ids
        attention_mask <- inputs$attention_mask
      }
      
      # Get embeddings
      with(torch$no_grad(), {
        outputs <- model(input_ids = input_ids, attention_mask = attention_mask)
      })
      
      hidden_states <- outputs$last_hidden_state
      
      # Apply pooling
      if (pooling.strategy == "mean") {
        mask_expanded <- attention_mask$unsqueeze(-1L)$expand_as(hidden_states)
        sum_embeddings <- torch$sum(hidden_states * mask_expanded, dim = 1L)
        sum_mask <- torch$clamp(mask_expanded$sum(dim = 1L), min = 1e-9)
        embeddings <- sum_embeddings / sum_mask
      } else if (pooling.strategy == "cls") {
        embeddings <- hidden_states[, 0L, ]
      } else if (pooling.strategy == "max") {
        mask_expanded <- attention_mask$unsqueeze(-1L)$expand_as(hidden_states)
        hidden_states_masked <- hidden_states$clone()
        hidden_states_masked[mask_expanded == 0] <- -1e9
        max_result <- torch$max(hidden_states_masked, dim = 1L)
        embeddings <- max_result$values
      }
      
      # Convert to R
      embeddings_np <- embeddings$cpu()$numpy()
      batch_matrix <- reticulate::py_to_r(embeddings_np)
      
      for (j in seq_len(nrow(batch_matrix))) {
        all_embeddings[[start_idx + j - 1]] <- batch_matrix[j, ]
      }
      
      if (!silently && i %% 5 == 0) cat(".")
    }
    
    # Combine into matrix (rows = dimensions, cols = items)
    embedding_matrix <- do.call(cbind, all_embeddings)
    colnames(embedding_matrix) <- item_ids
    
    result$embeddings <- embedding_matrix
    result$success <- TRUE
    
    if (!silently) {
      cat(" Done.\n")
      cat("  Generated", ncol(embedding_matrix), "embeddings with",
          nrow(embedding_matrix), "dimensions.\n\n")
    }
    
  }, error = function(e) {
    cat("\nError during local embedding:", conditionMessage(e), "\n")
    
    if (grepl("No module named 'transformers'", conditionMessage(e))) {
      cat("Please run: AIGENIE::reinstall_python_env()\n")
    } else if (grepl("CUDA|cuda", conditionMessage(e))) {
      cat("GPU error. Try setting device='cpu'.\n")
    }
    
    result$success <- FALSE
  })
  
  return(result)
}

# ============================================================================
# Helper Functions
# ============================================================================

#' Clear PyTorch Environment Variables
#' 
#' @keywords internal
clear_pytorch_environment <- function() {
  Sys.unsetenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
  Sys.setenv(TOKENIZERS_PARALLELISM = "false")
  
  if (reticulate::py_available()) {
    tryCatch({
      reticulate::py_run_string('
import os
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
    del os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
')
    }, error = function(e) {
      # Silently continue
    })
  }
}

#' Null-coalescing operator
#' @keywords internal
`%||%` <- function(x, y) if (is.null(x)) y else x
