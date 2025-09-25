#' Generate Items Using Local LLM
#'
#' @description
#' Generates scale items using locally installed language models in GGUF format.
#' This provides a privacy-preserving alternative to API-based generation.
#'
#' @param main.prompts Named list of prompts, one per item type
#' @param system.role Character string defining the system role
#' @param model.path Path to local GGUF model file
#' @param temperature Numeric between 0 and 2 for randomness
#' @param top.p Numeric between 0 and 1 for nucleus sampling
#' @param adaptive Logical. If TRUE, includes previous items to avoid repetition
#' @param silently Logical. If FALSE, displays progress
#' @param target.N Named list of target items per type
#' @param n.ctx Integer. Context window size (default 4096)
#' @param n.gpu.layers Integer. Number of layers to offload to GPU (-1 for all)
#' @param max.tokens Integer. Maximum tokens per generation (default 1024)
#'
#' @return A list containing items dataframe and success flag
#'
generate_items_via_local_llm <- function(main.prompts, system.role, model.path,
                                         temperature, top.p, adaptive, silently,
                                         target.N, n.ctx = 4096, n.gpu.layers = -1,
                                         max.tokens = 1024) {

  # Ensure Python environment with llama-cpp
  ensure_llama_cpp_python(silently = silently)

  # Initialize results dataframe (same structure as original)
  all_items_df <- data.frame(
    type = character(),
    attribute = character(),
    statement = character(),
    stringsAsFactors = FALSE
  )

  # Load the local model
  tryCatch({
    llama_cpp <- reticulate::import("llama_cpp")

    if (!silently) {
      cat("Loading local language model...\n")
    }

    # Suppress llama.cpp's verbose output
    # Redirect stderr to null during model operations
    suppress_output <- silently

    # Initialize Llama model with verbose=FALSE
    llm <- llama_cpp$Llama(
      model_path = model.path,
      n_ctx = as.integer(n.ctx),
      n_gpu_layers = as.integer(n.gpu.layers),
      seed = 123L,
      verbose = FALSE,  # This should suppress most output
      logits_all = FALSE,
      suppress_eos_token = FALSE
    )

    # Additional suppression using llama_cpp settings if available
    if (!is.null(llm$verbose)) {
      llm$verbose <- FALSE
    }

    if (!silently) {
      cat("Model loaded successfully.\n\n")
    }

  }, error = function(e) {
    stop(paste("Failed to load local model:", conditionMessage(e)))
  })

  # Iterate through each item type (following original structure)
  for (item_type in names(main.prompts)) {

    if (!silently) {
      cat(paste("Generating items for", item_type, "...\n"))
    }

    # Track items for this type
    type_items_df <- data.frame(
      type = character(),
      attribute = character(),
      statement = character(),
      stringsAsFactors = FALSE
    )

    # Track iterations without new items
    iterations_without_new <- 0
    total_iterations <- 0
    context_limit_reached <- FALSE
    max_previous_items <- Inf

    # Continue until target.N reached or stalled
    while (nrow(type_items_df) < target.N[[item_type]]) {

      total_iterations <- total_iterations + 1

      # Construct prompt with adaptive mode if needed
      current_prompt <- main.prompts[[item_type]]

      if (adaptive && nrow(all_items_df) > 0) {
        # Get previous items to append
        previous_items_to_use <- all_items_df

        # If context limit detected, truncate the number of previous items
        if (context_limit_reached && nrow(previous_items_to_use) > max_previous_items) {
          previous_items_to_use <- tail(previous_items_to_use, max_previous_items)
        }

        examples_string <- construct_item.examples_string(previous_items_to_use, item_type)
        current_prompt <- paste0(
          current_prompt,
          "\n\nDo NOT repeat, rephrase, or reuse the content of ANY items from this list of items you've already generated:\n",
          examples_string
        )
      }

      # Construct full prompt with system role
      full_prompt <- paste0(
        "System: ", system.role, "\n\n",
        "User: ", current_prompt, "\n\n",
        "Assistant:"
      )

      # Check prompt length and adjust if needed
      prompt_tokens <- nchar(full_prompt) / 4  # Rough estimate
      if (prompt_tokens > n.ctx * 0.7) {
        if (adaptive && !context_limit_reached) {
          if (!silently) {
            cat("\nWarning: Approaching context limit. Reducing number of previous items.\n")
          }
          context_limit_reached <- TRUE
          max_previous_items <- floor(nrow(all_items_df) * 0.5)
          next  # Retry with reduced context
        }
      }

      # Generate response
      generation_success <- FALSE
      generation_attempts <- 0

      while (!generation_success && generation_attempts < 5) {
        generation_attempts <- generation_attempts + 1

        try_response <- tryCatch({
          # Suppress verbose output during generation
          if (silently) {
            # Capture and discard output
            capture.output({
              response <- llm(
                prompt = full_prompt,
                max_tokens = as.integer(max.tokens),
                temperature = temperature,
                top_p = top.p,
                echo = FALSE,
                stop = list("User:", "System:"),
                stream = FALSE  # Disable streaming to reduce output
              )
            }, file = nullfile())
          } else {
            # Still suppress the performance metrics
            capture.output({
              response <- llm(
                prompt = full_prompt,
                max_tokens = as.integer(max.tokens),
                temperature = temperature,
                top_p = top.p,
                echo = FALSE,
                stop = list("User:", "System:"),
                stream = FALSE
              )
            }, file = nullfile())
          }

          generation_success <- TRUE
          response
        }, error = function(e) {
          if (grepl("context|token", conditionMessage(e), ignore.case = TRUE)) {
            if (adaptive && !context_limit_reached) {
              context_limit_reached <- TRUE
              max_previous_items <- floor(nrow(all_items_df) * 0.5)
            }
          }
          Sys.sleep(2)  # Brief pause before retry
          e
        })
      }

      # Check if generation failed
      if (!generation_success) {
        error_msg <- paste("Local generation failed after 5 attempts for", item_type,
                           "Error:", conditionMessage(try_response))
        cat(paste0("\n", error_msg, "\n"))

        if (nrow(all_items_df) > 0) {
          warning("Returning partial results generated before error.")
          return(list(items = all_items_df, successful = FALSE))
        } else {
          stop(error_msg)
        }
      }

      # Extract raw text from response
      raw_text <- response[["choices"]][[1]][["text"]]

      # Use the same cleaning function as original
      cleaned_df <- cleaning_function(raw_text, item_type)

      if (nrow(cleaned_df) > 0) {
        # Keep only unique statements
        new_items <- cleaned_df[!cleaned_df$statement %in% c(type_items_df$statement, all_items_df$statement), ]

        if (nrow(new_items) > 0) {
          type_items_df <- rbind(type_items_df, new_items)
          all_items_df <- rbind(all_items_df, new_items)
          iterations_without_new <- 0

          if (!silently) {
            cat(sprintf("\rItems generated for %s: %d", item_type, nrow(type_items_df)))
            flush.console()
          }
        } else {
          iterations_without_new <- iterations_without_new + 1
        }
      } else {
        iterations_without_new <- iterations_without_new + 1
      }

      # Check for stalling
      if (iterations_without_new >= 10) {
        if (!silently) {
          warning(paste0("\nWarning: Unable to generate new unique items for ", item_type,
                         " after 10 iterations. Moving to next item type.\n",
                         "Generated ", nrow(type_items_df), " out of ",
                         target.N[[item_type]], " requested items."))
        }
        break
      }
    }

    if (!silently) {
      cat("\n")  # New line after progress updates
    }
  }

  # Final message
  if (!silently) {
    cat(paste0("All items generated. Final sample size: ", nrow(all_items_df), "\n"))
  }

  return(list(items = all_items_df, successful = TRUE))
}



#' Ensure llama-cpp-python is Installed
#'
#' @param silently Logical. Suppress messages if TRUE
#'
#' Ensure llama-cpp-python is Installed
#'
#' @param silently Logical. Suppress messages if TRUE
#' @param force_reload Logical. Force Python environment reload after install
#'
#' Ensure llama-cpp-python is Installed
#'
#' @param silently Logical. Suppress messages if TRUE
#' @param force_reinstall Logical. Force reinstallation even if module exists
#'
ensure_llama_cpp_python <- function(silently = FALSE, force_reinstall = FALSE) {

  # Check if already available (unless forcing reinstall)
  if (!force_reinstall) {
    tryCatch({
      llama_cpp <- reticulate::import("llama_cpp")
      if (!silently) message("llama-cpp-python is already available.")
      return(invisible(TRUE))
    }, error = function(e) {
      # Module not available, proceed with installation
    })
  }

  if (!silently) {
    message("Setting up llama-cpp-python for local LLM support...")
  }

  # Clear any cached Python configuration to avoid conflicts
  reticulate::py_discover_config(required_module = NULL, use_environment = NULL)

  # Install through reticulate (this method has proven most reliable)
  tryCatch({
    if (!silently) message("Installing llama-cpp-python through reticulate...")

    # Determine if we need special compilation flags for Apple Silicon
    sys_info <- Sys.info()
    if (sys_info["sysname"] == "Darwin" && grepl("arm64|aarch64", sys_info["machine"])) {
      # Apple Silicon - use Metal acceleration
      Sys.setenv(CMAKE_ARGS = "-DLLAMA_METAL=on")
    }

    # Install with force-reinstall to ensure clean installation
    reticulate::py_install(
      "llama-cpp-python",
      pip = TRUE,
      pip_options = if(force_reinstall) "--force-reinstall --no-cache-dir" else ""
    )

    # Clear the CMAKE_ARGS after installation
    Sys.unsetenv("CMAKE_ARGS")

    if (!silently) {
      message("Installation complete. Testing import...")
    }

    # Try to import immediately
    llama_cpp <- reticulate::import("llama_cpp", delay_load = FALSE)
    if (!silently) message("✓ llama-cpp-python successfully installed and loaded!")
    return(invisible(TRUE))

  }, error = function(e) {
    # Installation succeeded but import failed - need R restart
    if (grepl("ModuleNotFoundError", conditionMessage(e))) {
      if (!silently) {
        message("\n================================================")
        message("llama-cpp-python installed but requires R restart")
        message("================================================")
        message("Please:")
        message("1. Restart R (Session -> Restart R in RStudio)")
        message("2. Run your code again")
        message("================================================\n")
      }
      return(invisible(FALSE))
    } else {
      # Actual installation error
      stop("Failed to install llama-cpp-python: ", conditionMessage(e))
    }
  })
}

#' Download or Verify Local LLM Model
#'
#' @description
#' Downloads a GGUF model from HuggingFace or verifies an existing local model.
#'
#' @param model_id HuggingFace model ID or local path
#' @param cache_dir Directory to store downloaded models (default: ~/.cache/aigenie/models/)
#' @param quantization Quantization level (e.g., "Q4_K_M", "Q5_K_M", "Q8_0")
#' @param silently Logical. Suppress progress messages
#'
#' @return Path to the local GGUF file
#'
#' @export
get_local_llm <- function(model_id,
                          cache_dir = "~/.cache/aigenie/models/",
                          quantization = "Q4_K_M",
                          silently = FALSE) {

  # Expand cache directory path
  cache_dir <- path.expand(cache_dir)

  # If already a local path, verify it exists
  if (file.exists(model_id)) {
    if (!silently) cat("Using existing model file:", model_id, "\n")
    return(model_id)
  }

  # Create cache directory if needed
  if (!dir.exists(cache_dir)) {
    dir.create(cache_dir, recursive = TRUE)
  }

  # Construct local filename
  model_name <- gsub("/", "_", model_id)
  local_file <- file.path(cache_dir, paste0(model_name, "_", quantization, ".gguf"))

  # Check if already downloaded
  if (file.exists(local_file)) {
    if (!silently) cat("Model already cached:", local_file, "\n")
    return(local_file)
  }

  # Download from HuggingFace
  if (!silently) cat("Downloading model from HuggingFace...\n")

  # Common GGUF model URLs
  base_urls <- list(
    "meta-llama/Llama-3.2-3B" = "https://huggingface.co/QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF/resolve/main/Meta-Llama-3.2-3B-Instruct.{Q}.gguf",
    "microsoft/Phi-3.5-mini" = "https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-{Q}.gguf"
  )

  # Get URL template
  url_template <- base_urls[[model_id]]
  if (is.null(url_template)) {
    stop("Model not found in registry. Please provide a direct path to a GGUF file.")
  }

  # Replace quantization placeholder
  download_url <- gsub("\\{Q\\}", quantization, url_template)

  # Download file
  tryCatch({
    download.file(download_url, local_file, mode = "wb")
    if (!silently) cat("Model downloaded successfully!\n")
  }, error = function(e) {
    stop("Failed to download model: ", conditionMessage(e))
  })

  return(local_file)
}


#' Check Local LLM Setup
#'
#' @description
#' Verifies that all components for local LLM generation are properly installed
#' and configured. Run this before attempting generation to ensure setup is complete.
#'
#' @param model.path Optional. Path to GGUF model file to verify it exists
#' @param silently Logical. Suppress messages if TRUE
#'
#' @return Logical. TRUE if setup is complete, FALSE otherwise
#'
#' @export
check_local_llm_setup <- function(model.path = NULL, silently = FALSE) {

  setup_ok <- TRUE

  if (!silently) {
    cat("Checking local LLM setup...\n")
    cat("==========================\n")
  }

  # Check Python
  tryCatch({
    py_config <- reticulate::py_config()
    if (!silently) {
      cat("✓ Python found:", py_config$python, "\n")
    }
  }, error = function(e) {
    cat("✗ Python not configured\n")
    setup_ok <- FALSE
  })

  # Check llama-cpp-python
  tryCatch({
    llama_cpp <- reticulate::import("llama_cpp")
    if (!silently) {
      cat("✓ llama-cpp-python installed\n")
    }
  }, error = function(e) {
    cat("✗ llama-cpp-python not found. Run: ensure_llama_cpp_python()\n")
    setup_ok <- FALSE
  })

  # Check model file if provided
  if (!is.null(model.path)) {
    if (file.exists(model.path)) {
      file_size <- file.info(model.path)$size / 1024^3  # Size in GB
      if (!silently) {
        cat(sprintf("✓ Model file found: %.2f GB\n", file_size))
      }
    } else {
      cat("✗ Model file not found:", model.path, "\n")
      setup_ok <- FALSE
    }
  }

  if (!silently) {
    cat("==========================\n")
    if (setup_ok) {
      cat("Setup complete! Ready for local generation.\n")
    } else {
      cat("Setup incomplete. Please address the issues above.\n")
    }
  }

  return(invisible(setup_ok))
}
