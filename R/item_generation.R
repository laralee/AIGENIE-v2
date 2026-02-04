# item_generation.R
# Item Generation Module for AI-GENIE
# ============================================================================
# This file handles item generation using LLMs across all supported providers:
# OpenAI, Groq, and local GGUF models.

# ============================================================================
# Main Item Generation Function
# ============================================================================

#' Generate Items via LLM
#' 
#' @description
#' Generates scale items using the specified LLM provider. Supports OpenAI,
#' Groq, and local GGUF models.
#' 
#' @param main.prompts Named list of prompts for each item type
#' @param system.role Character string defining the system role
#' @param model Character string specifying the model
#' @param top.p Numeric. Nucleus sampling parameter
#' @param temperature Numeric. Sampling temperature
#' @param adaptive Logical. Use adaptive generation with previous items?
#' @param silently Logical. Suppress progress messages?
#' @param groq.API Optional Groq API key
#' @param openai.API Optional OpenAI API key
#' @param target.N Named list of target item counts per type
#' 
#' @return A list with 'items' data frame and 'successful' flag
#' @keywords internal
generate_items_via_llm <- function(main.prompts, system.role, model, top.p, temperature,
                                    adaptive, silently, groq.API, openai.API,
                                    anthropic.API = NULL, target.N) {
  
  ensure_aigenie_python()
  
  # Detect provider
  provider_info <- detect_llm_provider(model, groq.API, openai.API,
                                        anthropic.API = anthropic.API)
  provider <- provider_info$provider
  model <- provider_info$model
  
  if (!silently) {
    cat("\nGenerating items using", provider, "model:", model, "\n")
  }
  
  # Initialize results
  all_items_df <- data.frame(
    type = character(),
    attribute = character(),
    statement = character(),
    stringsAsFactors = FALSE
  )
  
  successful <- TRUE
  
  # Process each item type
  for (item_type in names(main.prompts)) {
    
    if (!silently) {
      cat("\n--- Generating items for:", item_type, "---\n")
    }
    
    type_items_df <- data.frame(
      type = character(),
      attribute = character(),
      statement = character(),
      stringsAsFactors = FALSE
    )
    
    iterations_without_new <- 0
    total_iterations <- 0
    max_iterations <- 50
    
    # Generate until we reach target
    while (nrow(type_items_df) < target.N[[item_type]] && total_iterations < max_iterations) {
      
      total_iterations <- total_iterations + 1
      
      # Build prompt with adaptive mode
      current_prompt <- main.prompts[[item_type]]
      
      if (adaptive && nrow(all_items_df) > 0) {
        examples_string <- construct_item.examples_string(all_items_df, item_type)
        if (!is.null(examples_string)) {
          current_prompt <- paste0(
            current_prompt,
            "\n\nDo NOT repeat, rephrase, or reuse ANY items from this list:\n",
            examples_string
          )
        }
      }
      
      # Generate text using unified interface
      raw_text <- tryCatch({
        generate_text_llm(
          prompt = current_prompt,
          system.role = system.role,
          model = model,
          temperature = temperature,
          top.p = top.p,
          max_tokens = 2048L,
          openai.API = openai.API,
          groq.API = groq.API,
          anthropic.API = anthropic.API
        )
      }, error = function(e) {
        if (!silently) {
          cat("Generation error:", conditionMessage(e), "\n")
        }
        NULL
      })
      
      if (is.null(raw_text)) {
        iterations_without_new <- iterations_without_new + 1
        Sys.sleep(2)
        next
      }
      
      # Parse response
      cleaned_df <- cleaning_function(raw_text, item_type)
      
      if (nrow(cleaned_df) > 0) {
        # Remove duplicates
        new_items <- cleaned_df[!cleaned_df$statement %in% c(type_items_df$statement, 
                                                              all_items_df$statement), ]
        
        if (nrow(new_items) > 0) {
          type_items_df <- rbind(type_items_df, new_items)
          all_items_df <- rbind(all_items_df, new_items)
          iterations_without_new <- 0
          
          if (!silently) {
            cat("\rItems for", item_type, ":", nrow(type_items_df), "/", 
                target.N[[item_type]], "   ")
            flush.console()
          }
        } else {
          iterations_without_new <- iterations_without_new + 1
        }
      } else {
        iterations_without_new <- iterations_without_new + 1
        # Debug: show first 200 chars of raw_text when parsing fails completely
        if (!silently && iterations_without_new <= 3) {
          preview <- substr(raw_text, 1, min(200, nchar(raw_text)))
          cat("\n[Parse failed for", item_type, "- preview:", preview, "...]\n")
        }
      }
      
      # Check for stalling
      if (iterations_without_new >= 10) {
        if (!silently) {
          warning("\nUnable to generate new items for ", item_type, 
                  " after 10 iterations. Generated ", nrow(type_items_df),
                  " of ", target.N[[item_type]], " items.", immediate. = TRUE)
        }
        break
      }
      
      # Small delay to avoid rate limits
      Sys.sleep(0.5)
    }
    
    if (!silently) {
      cat("\n")
    }
  }
  
  if (!silently) {
    cat("\n=== Generation complete ===\n")
    cat("Total items generated:", nrow(all_items_df), "\n\n")
  }
  
  return(list(items = all_items_df, successful = successful))
}

# ============================================================================
# Local LLM Item Generation
# ============================================================================

#' Generate Items Using Local LLM (GGUF)
#' 
#' @description
#' Generates items using a locally installed GGUF model via llama-cpp-python.
#' 
#' @param main.prompts Named list of prompts
#' @param system.role Character string with system role
#' @param model.path Path to local GGUF model file
#' @param temperature Numeric. Sampling temperature
#' @param top.p Numeric. Nucleus sampling parameter
#' @param adaptive Logical. Use adaptive generation?
#' @param silently Logical. Suppress messages?
#' @param target.N Named list of target counts
#' @param n.ctx Integer. Context window size
#' @param n.gpu.layers Integer. GPU layers (-1 for all)
#' @param max.tokens Integer. Max tokens per generation
#' 
#' @return A list with 'items' data frame and 'successful' flag
#' @keywords internal
generate_items_via_local_llm <- function(main.prompts, system.role, model.path,
                                          temperature, top.p, adaptive, silently,
                                          target.N, n.ctx = 4096, n.gpu.layers = -1,
                                          max.tokens = 1024) {
  
  # Ensure llama-cpp is available
  ensure_llama_cpp_python(silently = silently)
  
  # Initialize results
  all_items_df <- data.frame(
    type = character(),
    attribute = character(),
    statement = character(),
    stringsAsFactors = FALSE
  )
  
  # Load the model
  tryCatch({
    llama_cpp <- reticulate::import("llama_cpp")
    
    if (!silently) {
      cat("Loading local model...\n")
    }
    
    llm <- llama_cpp$Llama(
      model_path = model.path,
      n_ctx = as.integer(n.ctx),
      n_gpu_layers = as.integer(n.gpu.layers),
      seed = 123L,
      verbose = FALSE
    )
    
    if (!silently) {
      cat("Model loaded successfully.\n\n")
    }
    
  }, error = function(e) {
    stop("Failed to load local model: ", conditionMessage(e), call. = FALSE)
  })
  
  # Process each item type
  for (item_type in names(main.prompts)) {
    
    if (!silently) {
      cat("Generating items for", item_type, "...\n")
    }
    
    type_items_df <- data.frame(
      type = character(),
      attribute = character(),
      statement = character(),
      stringsAsFactors = FALSE
    )
    
    iterations_without_new <- 0
    context_limit_reached <- FALSE
    max_previous_items <- Inf
    
    while (nrow(type_items_df) < target.N[[item_type]]) {
      
      # Build prompt
      current_prompt <- main.prompts[[item_type]]
      
      if (adaptive && nrow(all_items_df) > 0) {
        previous_items <- all_items_df
        
        if (context_limit_reached && nrow(previous_items) > max_previous_items) {
          previous_items <- tail(previous_items, max_previous_items)
        }
        
        examples_string <- construct_item.examples_string(previous_items, item_type)
        if (!is.null(examples_string)) {
          current_prompt <- paste0(
            current_prompt,
            "\n\nDo NOT repeat any of these items:\n",
            examples_string
          )
        }
      }
      
      # Format for local model
      full_prompt <- paste0(
        "System: ", system.role, "\n\n",
        "User: ", current_prompt, "\n\n",
        "Assistant:"
      )
      
      # Check context limit
      prompt_tokens <- nchar(full_prompt) / 4
      if (prompt_tokens > n.ctx * 0.7) {
        context_limit_reached <- TRUE
        max_previous_items <- floor(nrow(all_items_df) * 0.5)
        next
      }
      
      # Generate
      raw_text <- tryCatch({
        response <- llm(
          prompt = full_prompt,
          max_tokens = as.integer(max.tokens),
          temperature = temperature,
          top_p = top.p,
          echo = FALSE,
          stop = list("User:", "System:")
        )
        response[["choices"]][[1]][["text"]]
      }, error = function(e) {
        if (!silently) cat("Generation error:", conditionMessage(e), "\n")
        NULL
      })
      
      if (is.null(raw_text)) {
        iterations_without_new <- iterations_without_new + 1
        if (iterations_without_new >= 10) break
        next
      }
      
      # Parse
      cleaned_df <- cleaning_function(raw_text, item_type)
      
      if (nrow(cleaned_df) > 0) {
        new_items <- cleaned_df[!cleaned_df$statement %in% c(type_items_df$statement,
                                                              all_items_df$statement), ]
        
        if (nrow(new_items) > 0) {
          type_items_df <- rbind(type_items_df, new_items)
          all_items_df <- rbind(all_items_df, new_items)
          iterations_without_new <- 0
          
          if (!silently) {
            cat("\rItems:", nrow(type_items_df), "/", target.N[[item_type]], "   ")
            flush.console()
          }
        } else {
          iterations_without_new <- iterations_without_new + 1
        }
      } else {
        iterations_without_new <- iterations_without_new + 1
      }
      
      if (iterations_without_new >= 10) {
        if (!silently) {
          warning("Unable to generate new items for ", item_type, 
                  " after 10 iterations.")
        }
        break
      }
    }
    
    if (!silently) cat("\n")
  }
  
  if (!silently) {
    cat("Total items generated:", nrow(all_items_df), "\n")
  }
  
  return(list(items = all_items_df, successful = TRUE))
}

#' Ensure llama-cpp-python is Installed
#' 
#' @param silently Logical. Suppress messages?
#' @param force_reinstall Logical. Force reinstallation?
#' 
#' @keywords internal
ensure_llama_cpp_python <- function(silently = FALSE, force_reinstall = FALSE) {
  
  if (!force_reinstall) {
    tryCatch({
      llama_cpp <- reticulate::import("llama_cpp")
      if (!silently) message("llama-cpp-python is available.")
      return(invisible(TRUE))
    }, error = function(e) {
      # Not available, proceed with installation
    })
  }
  
  if (!silently) {
    message("Setting up llama-cpp-python for local LLM support...")
  }
  
  # Install through UV
  env_path <- get_aigenie_env_path()
  python_path <- get_python_path(env_path)
  
  # Check for Apple Silicon
  sys_info <- Sys.info()
  if (sys_info["sysname"] == "Darwin" && grepl("arm64|aarch64", sys_info["machine"])) {
    Sys.setenv(CMAKE_ARGS = "-DLLAMA_METAL=on")
  }
  
  result <- system2("uv",
                    args = c("pip", "install",
                             "--python", shQuote(python_path),
                             "llama-cpp-python"),
                    stdout = TRUE, stderr = TRUE)
  
  Sys.unsetenv("CMAKE_ARGS")
  
  exit_status <- attr(result, "status")
  if (!is.null(exit_status) && exit_status != 0) {
    stop("Failed to install llama-cpp-python: ", paste(result, collapse = "\n"),
         call. = FALSE)
  }
  
  tryCatch({
    llama_cpp <- reticulate::import("llama_cpp", delay_load = FALSE)
    if (!silently) message("llama-cpp-python installed successfully!")
    return(invisible(TRUE))
  }, error = function(e) {
    message("\nllama-cpp-python installed but requires R restart.")
    message("Please restart R and try again.")
    return(invisible(FALSE))
  })
}

# ============================================================================
# Response Cleaning and Parsing
# ============================================================================

#' Clean and Parse LLM Response
#' 
#' @description
#' Parses LLM-generated text to extract structured item data.
#' Handles JSON format and falls back to text parsing.
#' 
#' @param raw_text Character string with LLM response
#' @param item_type Character string with the item type
#' 
#' @return Data frame with type, attribute, statement columns
#' @keywords internal
cleaning_function <- function(raw_text, item_type) {
  
  result_df <- data.frame(
    type = character(),
    attribute = character(),
    statement = character(),
    stringsAsFactors = FALSE
  )
  
  if (is.null(raw_text) || nchar(trimws(raw_text)) == 0) {
    return(result_df)
  }
  
  # Try JSON parsing first
  json_result <- tryCatch({
    # Remove markdown code fences if present
    clean_text <- raw_text
    clean_text <- gsub("```json\\s*", "", clean_text)
    clean_text <- gsub("```\\s*", "", clean_text)
    
    # Find JSON array in response ((?s) makes . match newlines)
    json_match <- regmatches(clean_text, regexpr("(?s)\\[.*\\]", clean_text, perl = TRUE))
    
    if (length(json_match) > 0 && nchar(json_match) > 2) {
      parsed <- jsonlite::fromJSON(json_match, flatten = TRUE)
      
      if (is.data.frame(parsed) && "attribute" %in% names(parsed) && "statement" %in% names(parsed)) {
        data.frame(
          type = item_type,
          attribute = trimws(tolower(as.character(parsed$attribute))),
          statement = trimws(as.character(parsed$statement)),
          stringsAsFactors = FALSE
        )
      } else {
        NULL
      }
    } else {
      NULL
    }
  }, error = function(e) NULL)
  
  if (!is.null(json_result) && nrow(json_result) > 0) {
    # Filter out empty statements
    json_result <- json_result[nchar(json_result$statement) > 5, ]
    return(json_result)
  }
  
  # Fallback: text parsing
  lines <- strsplit(raw_text, "\n")[[1]]
  lines <- trimws(lines)
  lines <- lines[nchar(lines) > 0]
  
  for (line in lines) {
    # Try to extract numbered items
    # Pattern: "1. [attribute]: statement" or "1) attribute - statement"
    patterns <- list(
      "^\\d+[.)]\\s*\\[?([^\\]:\\-]+)\\]?[:\\-]\\s*(.+)$",
      "^\\*\\s*\\[?([^\\]:\\-]+)\\]?[:\\-]\\s*(.+)$",
      "^[\\-•]\\s*\\[?([^\\]:\\-]+)\\]?[:\\-]\\s*(.+)$"
    )
    
    for (pattern in patterns) {
      match <- regmatches(line, regexec(pattern, line, perl = TRUE))[[1]]
      if (length(match) == 3) {
        attribute <- trimws(tolower(match[2]))
        statement <- trimws(match[3])
        
        # Clean statement
        statement <- gsub('^["\']|["\']$', '', statement)
        statement <- gsub("\\s+", " ", statement)
        
        if (nchar(statement) > 5) {
          result_df <- rbind(result_df, data.frame(
            type = item_type,
            attribute = attribute,
            statement = statement,
            stringsAsFactors = FALSE
          ))
        }
        break
      }
    }
  }
  
  return(result_df)
}

#' Construct Item Examples String for Prompts
#' 
#' @description
#' Formats previous items as JSON for inclusion in prompts.
#' 
#' @param item.examples Data frame with previous items
#' @param current_type Character string with current item type
#' 
#' @return JSON-formatted string or NULL
#' @keywords internal
construct_item.examples_string <- function(item.examples, current_type) {
  
  # Filter by type if available
  if ("type" %in% names(item.examples)) {
    filtered <- item.examples[tolower(item.examples$type) == tolower(current_type), ]
  } else {
    filtered <- item.examples
  }
  
  if (nrow(filtered) == 0) return(NULL)
  
  # Build simplified data frame
  df <- data.frame(
    attribute = as.character(filtered$attribute),
    statement = as.character(filtered$statement),
    stringsAsFactors = FALSE
  )
  
  # Convert to JSON
  jsonlite::toJSON(df, auto_unbox = TRUE)
}


# ============================================================================
# Local LLM Setup and Model Management
# ============================================================================

#' Check Local LLM Setup
#'
#' @description
#' Verifies that all requirements for local LLM inference are met,
#' including Python environment, llama-cpp-python installation, and
#' model file accessibility.
#'
#' @param model.path Path to the GGUF model file
#' @param silently Logical. Suppress progress messages?
#'
#' @return Logical. TRUE if setup is complete, FALSE otherwise.
#' @export
check_local_llm_setup <- function(model.path, silently = FALSE) {

  all_ok <- TRUE

  # Check 1: Model file exists
  if (!file.exists(model.path)) {
    if (!silently) {
      cat("\n[X] Model file not found:", model.path, "\n")
      cat("    Please download a GGUF model or check the file path.\n")
      cat("    You can use get_local_llm() to download a model.\n")
    }
    all_ok <- FALSE
  } else {
    if (!silently) cat("[OK] Model file found:", basename(model.path), "\n")
  }

  # Check 2: File is a GGUF file
  if (all_ok && !grepl("\\.gguf$", model.path, ignore.case = TRUE)) {
    if (!silently) {
      cat("\n[!] Warning: Model file does not have .gguf extension.\n")
      cat("    local_AIGENIE requires GGUF format models.\n")
    }
  }

  # Check 3: Python environment
  tryCatch({
    ensure_aigenie_python()
    if (!silently) cat("[OK] Python environment ready\n")
  }, error = function(e) {
    if (!silently) {
      cat("\n[X] Python environment not ready:", conditionMessage(e), "\n")
      cat("    Run: AIGENIE::ensure_aigenie_python()\n")
    }
    all_ok <<- FALSE
  })

  # Check 4: llama-cpp-python
  if (all_ok) {
    llama_available <- reticulate::py_module_available("llama_cpp")
    if (!llama_available) {
      if (!silently) {
        cat("\n[X] llama-cpp-python not installed\n")
        cat("    Run: AIGENIE::install_local_llm_support()\n")
      }
      all_ok <- FALSE
    } else {
      if (!silently) cat("[OK] llama-cpp-python installed\n")
    }
  }

  if (all_ok && !silently) {
    cat("\nAll checks passed. Ready for local item generation.\n")
  }

  return(all_ok)
}


#' Download a Local LLM Model
#'
#' @description
#' Downloads a GGUF model from HuggingFace for use with local_AIGENIE.
#' Models are saved to a user-specified directory or the default AIGENIE
#' models directory.
#'
#' @param repo_id HuggingFace repository ID (e.g., "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
#' @param filename Specific GGUF filename to download (e.g., "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
#' @param save_dir Directory to save the model. If NULL, uses the default
#'   AIGENIE models directory.
#' @param hf.token Optional HuggingFace token for gated models
#'
#' @return Character string with the full path to the downloaded model file.
#'
#' @examples
#' \dontrun{
#' # Download a Mistral 7B model
#' model_path <- get_local_llm(
#'   repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
#'   filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
#' )
#'
#' # Use it with local_AIGENIE
#' results <- local_AIGENIE(
#'   item.attributes = my_attributes,
#'   model.path = model_path
#' )
#' }
#'
#' @export
get_local_llm <- function(repo_id,
                           filename,
                           save_dir = NULL,
                           hf.token = NULL) {

  # Set default save directory
  if (is.null(save_dir)) {
    save_dir <- file.path(tools::R_user_dir("AIGENIE", "data"), "models")
  }

  # Create directory if needed
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }

  # Check if file already exists
  dest_path <- file.path(save_dir, filename)
  if (file.exists(dest_path)) {
    cat("Model already downloaded:", dest_path, "\n")
    return(dest_path)
  }

  # Ensure Python environment is ready
  ensure_aigenie_python()

  cat("Downloading model from HuggingFace...\n")
  cat("  Repo:", repo_id, "\n")
  cat("  File:", filename, "\n")
  cat("  Destination:", save_dir, "\n\n")

  tryCatch({
    huggingface_hub <- reticulate::import("huggingface_hub")

    # Authenticate if token provided
    if (!is.null(hf.token)) {
      huggingface_hub$login(token = hf.token, add_to_git_credential = FALSE)
    }

    # Download the file
    downloaded_path <- huggingface_hub$hf_hub_download(
      repo_id = repo_id,
      filename = filename,
      local_dir = save_dir
    )

    # Convert to R string
    final_path <- as.character(downloaded_path)

    cat("\nModel downloaded successfully!\n")
    cat("Path:", final_path, "\n")

    return(final_path)

  }, error = function(e) {
    stop("Failed to download model: ", conditionMessage(e),
         "\n\nPlease check:\n",
         "  1. The repo_id and filename are correct\n",
         "  2. You have internet connectivity\n",
         "  3. For gated models, provide a valid hf.token\n",
         call. = FALSE)
  })
}
