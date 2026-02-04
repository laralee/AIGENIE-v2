# llm_providers.R
# Unified LLM Provider Interface for AI-GENIE
# ============================================================================
# This file provides a unified interface for text generation across multiple
# LLM providers: OpenAI, Groq, and HuggingFace (both API and local models).

# ============================================================================
# Provider Detection and Model Mapping
# ============================================================================

#' Detect LLM Provider from Model Name
#' 
#' @description
#' Determines which API provider to use based on the model name.
#' 
#' @param model Character string specifying the model name
#' @param groq.API Optional Groq API key (if provided, prefers Groq for compatible models)
#' @param openai.API Optional OpenAI API key
#' @param hf.token Optional HuggingFace token
#' 
#' @return A list with provider name and normalized model string
#' @keywords internal
detect_llm_provider <- function(model, groq.API = NULL, openai.API = NULL, 
                                 hf.token = NULL, anthropic.API = NULL) {
  
  model_lower <- tolower(trimws(model))
  
  # OpenAI models
  openai_models <- c(
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-4-32k",
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "o1", "o1-mini", "o1-preview"
  )
  
  openai_patterns <- c("gpt-4", "gpt-3.5", "o1-", "chatgpt", "gpt-5")
  
  # Anthropic Claude models
  anthropic_patterns <- c("claude-")
  
  # Groq-hosted models (including open-source models)
  groq_models <- c(
    # Llama models
    "llama-3.3-70b-versatile", "llama-3.3-70b-specdec", "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant", "llama-3.2-1b-preview", "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview",
    "llama3-70b-8192", "llama3-8b-8192",
    # Mixtral
    "mixtral-8x7b-32768",
    # Gemma
    "gemma-7b-it", "gemma2-9b-it",
    # DeepSeek  
    "deepseek-r1-distill-llama-70b",
    # Qwen
    "qwen-2.5-72b", "qwen-2.5-coder-32b", "qwen-qwq-32b"
  )
  
  # Normalize common aliases
  model_aliases <- list(
    # OpenAI aliases
    "gpt4o" = "gpt-4o",
    "gpt4" = "gpt-4",
    "gpt 4o" = "gpt-4o",
    "gpt 4" = "gpt-4",
    "gpt-4o mini" = "gpt-4o-mini",
    "chatgpt" = "gpt-4o",
    "o1" = "o1",
    "o1 mini" = "o1-mini",
    # Anthropic aliases
    "claude" = "claude-sonnet-4-5-20250929",
    "claude sonnet" = "claude-sonnet-4-5-20250929",
    "claude opus" = "claude-opus-4-20250514",
    "claude haiku" = "claude-haiku-4-5-20251001",
    "claude 4.5" = "claude-sonnet-4-5-20250929",
    "claude 4" = "claude-sonnet-4-20250514",
    "sonnet" = "claude-sonnet-4-5-20250929",
    "opus" = "claude-opus-4-20250514",
    "haiku" = "claude-haiku-4-5-20251001",
    # Groq/Open-source aliases
    "llama3" = "llama3-70b-8192",
    "llama 3" = "llama3-70b-8192",
    "llama3.3" = "llama-3.3-70b-versatile",
    "llama 3.3" = "llama-3.3-70b-versatile",
    "llama3.1" = "llama-3.1-70b-versatile",
    "llama 3.1" = "llama-3.1-70b-versatile",
    "mixtral" = "mixtral-8x7b-32768",
    "deepseek" = "deepseek-r1-distill-llama-70b",
    "deepseek r1" = "deepseek-r1-distill-llama-70b",
    "qwen" = "qwen-2.5-72b",
    "gemma" = "gemma2-9b-it"
  )
  
  # Check for alias
  if (model_lower %in% names(model_aliases)) {
    model <- model_aliases[[model_lower]]
    model_lower <- tolower(model)
  }
  
  # Determine provider
  provider <- NULL
  normalized_model <- model
  
  # Check if it's clearly an OpenAI model
  is_openai <- model_lower %in% tolower(openai_models) ||
    any(sapply(openai_patterns, function(p) grepl(p, model_lower, fixed = TRUE)))
  
  # Check if it's an Anthropic Claude model
  is_anthropic <- any(sapply(anthropic_patterns, function(p) grepl(p, model_lower, fixed = TRUE)))
  
  # Check if it's a Groq-hosted model (exact match in hardcoded list)
  is_groq <- model_lower %in% tolower(groq_models)
  
  # Models with "/" could be HuggingFace OR Groq (Groq now hosts models with
  # HuggingFace-style IDs like "meta-llama/llama-4-maverick-17b-128e-instruct").
  # If groq.API is provided, prefer Groq for slash-style models.
  has_slash <- grepl("/", model)
  is_huggingface <- has_slash && !is_openai && !is_groq && !is_anthropic && is.null(groq.API)
  
  # If the model has a slash and groq.API is provided (but not in hardcoded list),
  # route to Groq — the user explicitly provided a Groq key
  is_groq_slash <- has_slash && !is_openai && !is_anthropic && !is.null(groq.API)
  
  if (is_openai) {
    if (is.null(openai.API)) {
      stop("Model '", model, "' requires an OpenAI API key. Please provide openai.API.",
           call. = FALSE)
    }
    provider <- "openai"
    # Normalize to official model name
    for (official in openai_models) {
      if (grepl(tolower(official), model_lower, fixed = TRUE) || 
          model_lower == tolower(official)) {
        normalized_model <- official
        break
      }
    }
  } else if (is_anthropic) {
    if (is.null(anthropic.API)) {
      stop("Model '", model, "' requires an Anthropic API key. Please provide anthropic.API.",
           call. = FALSE)
    }
    provider <- "anthropic"
    normalized_model <- model
  } else if (is_groq || is_groq_slash) {
    if (is.null(groq.API)) {
      stop("Model '", model, "' is available on Groq. Please provide groq.API.",
           call. = FALSE)
    }
    provider <- "groq"
    # Find exact match in groq_models (for hardcoded models)
    idx <- which(tolower(groq_models) == model_lower)
    if (length(idx) > 0) {
      normalized_model <- groq_models[idx[1]]
    }
    # For slash-style models, pass through as-is (Groq accepts them directly)
  } else if (is_huggingface) {
    provider <- "huggingface"
    normalized_model <- model
  } else {
    # Try to infer based on available API keys
    if (!is.null(groq.API)) {
      provider <- "groq"
      if (model_lower %in% c("default", "auto", "")) {
        normalized_model <- "llama-3.3-70b-versatile"
      }
    } else if (!is.null(openai.API)) {
      provider <- "openai"
      if (model_lower %in% c("default", "auto", "")) {
        normalized_model <- "gpt-4o"
      }
    } else if (!is.null(anthropic.API)) {
      provider <- "anthropic"
      if (model_lower %in% c("default", "auto", "")) {
        normalized_model <- "claude-sonnet-4-5-20250929"
      }
    } else {
      stop("Could not determine provider for model '", model, "'. ",
           "Please provide an API key (openai.API, groq.API, or anthropic.API).",
           call. = FALSE)
    }
  }
  
  # Strip provider prefix from model name if present (APIs expect bare model names)
  normalized_model <- sub("^(OpenAI|Groq|Anthropic|HuggingFace)/", "", normalized_model)
  
  list(
    provider = provider,
    model = normalized_model
  )
}

#' Normalize Model Name (Legacy Compatibility)
#' 
#' @description
#' Validates and normalizes model names. This function maintains backward
#' compatibility with existing code.
#' 
#' @param model Character string specifying the model
#' @param groq.API Optional Groq API key
#' @param openai.API Optional OpenAI API key
#' @param anthropic.API Optional Anthropic API key
#' @param silently Logical. Suppress informational messages?
#' 
#' @return Normalized model name string
#' @keywords internal
normalize_model_name <- function(model, groq.API = NULL, openai.API = NULL,
                                  anthropic.API = NULL, silently = FALSE) {
  
  result <- detect_llm_provider(model, groq.API, openai.API,
                                 anthropic.API = anthropic.API)
  
  if (!silently) {
    message("Using ", result$provider, " model: ", result$model)
  }
  
  return(result$model)
}

# ============================================================================
# Unified Text Generation Interface
# ============================================================================

#' Generate Text Using Any Supported LLM Provider
#' 
#' @description
#' Unified interface for text generation that automatically routes to the
#' appropriate provider (OpenAI, Groq, Anthropic, or HuggingFace).
#' 
#' @param prompt Character string with the user prompt
#' @param system.role Character string with the system prompt
#' @param model Character string specifying the model
#' @param temperature Numeric. Sampling temperature (0-2)
#' @param top.p Numeric. Nucleus sampling parameter (0-1)
#' @param max_tokens Integer. Maximum tokens to generate
#' @param openai.API Optional OpenAI API key
#' @param groq.API Optional Groq API key
#' @param anthropic.API Optional Anthropic API key
#' @param hf.token Optional HuggingFace token
#' 
#' @return Character string with the generated text
#' @keywords internal
generate_text_llm <- function(prompt, 
                               system.role = NULL,
                               model = "gpt-4o",
                               temperature = 1,
                               top.p = 1,
                               max_tokens = 2048,
                               openai.API = NULL,
                               groq.API = NULL,
                               anthropic.API = NULL,
                               hf.token = NULL) {
  
  # Detect provider
  provider_info <- detect_llm_provider(
    model, groq.API, openai.API, hf.token, anthropic.API
  )
  
  # Route to appropriate provider
  if (provider_info$provider == "openai") {
    return(generate_text_openai(
      prompt = prompt,
      system.role = system.role,
      model = provider_info$model,
      temperature = temperature,
      top.p = top.p,
      max_tokens = max_tokens,
      api_key = openai.API
    ))
  } else if (provider_info$provider == "groq") {
    return(generate_text_groq(
      prompt = prompt,
      system.role = system.role,
      model = provider_info$model,
      temperature = temperature,
      top.p = top.p,
      max_tokens = max_tokens,
      api_key = groq.API
    ))
  } else if (provider_info$provider == "anthropic") {
    return(generate_text_anthropic(
      prompt = prompt,
      system.role = system.role,
      model = provider_info$model,
      temperature = temperature,
      top.p = top.p,
      max_tokens = max_tokens,
      api_key = anthropic.API
    ))
  } else if (provider_info$provider == "huggingface") {
    return(generate_text_huggingface(
      prompt = prompt,
      system.role = system.role,
      model = provider_info$model,
      temperature = temperature,
      top.p = top.p,
      max_tokens = max_tokens,
      hf_token = hf.token
    ))
  } else {
    stop("Unknown provider: ", provider_info$provider, call. = FALSE)
  }
}

# ============================================================================
# OpenAI Provider
# ============================================================================

#' Generate Text Using OpenAI API
#' 
#' @param prompt Character string with the user prompt
#' @param system.role Character string with the system prompt
#' @param model Character string specifying the model
#' @param temperature Numeric. Sampling temperature
#' @param top.p Numeric. Nucleus sampling parameter
#' @param max_tokens Integer. Maximum tokens to generate
#' @param api_key OpenAI API key
#' 
#' @return Character string with the generated text
#' @keywords internal
generate_text_openai <- function(prompt, system.role = NULL, model = "gpt-4o",
                                  temperature = 1, top.p = 1, max_tokens = 2048,
                                  api_key) {
  
  ensure_aigenie_python()
  
  openai <- reticulate::import("openai")
  openai$api_key <- api_key
  
  # Build messages
  messages <- list()
  if (!is.null(system.role) && nchar(system.role) > 0) {
    messages[[length(messages) + 1]] <- list(role = "system", content = system.role)
  }
  messages[[length(messages) + 1]] <- list(role = "user", content = prompt)
  
  # Create completion
  response <- openai$ChatCompletion$create(
    model = model,
    messages = messages,
    temperature = temperature,
    top_p = top.p,
    max_tokens = as.integer(max_tokens)
  )
  
  # Extract text
  return(response$choices[[1]]$message$content)
}

# ============================================================================
# Groq Provider
# ============================================================================

#' Generate Text Using Groq API
#' 
#' @param prompt Character string with the user prompt
#' @param system.role Character string with the system prompt
#' @param model Character string specifying the model
#' @param temperature Numeric. Sampling temperature
#' @param top.p Numeric. Nucleus sampling parameter
#' @param max_tokens Integer. Maximum tokens to generate
#' @param api_key Groq API key
#' 
#' @return Character string with the generated text
#' @keywords internal
generate_text_groq <- function(prompt, system.role = NULL, model = "llama-3.3-70b-versatile",
                                temperature = 1, top.p = 1, max_tokens = 2048,
                                api_key) {
  
  ensure_aigenie_python()
  
  groq <- reticulate::import("groq")
  client <- groq$Groq(api_key = api_key)
  
  # Build messages
  messages <- list()
  if (!is.null(system.role) && nchar(system.role) > 0) {
    messages[[length(messages) + 1]] <- list(role = "system", content = system.role)
  }
  messages[[length(messages) + 1]] <- list(role = "user", content = prompt)
  
  # Create completion
  response <- client$chat$completions$create(
    model = model,
    messages = messages,
    temperature = temperature,
    top_p = top.p,
    max_tokens = as.integer(max_tokens)
  )
  
  # Extract text
  return(response$choices[[1]]$message$content)
}

# ============================================================================
# Anthropic Provider
# ============================================================================

#' Generate Text Using Anthropic Messages API
#' 
#' @description
#' Generates text using Anthropic's Claude models via the /v1/messages endpoint.
#' Uses the requests library directly (no extra SDK dependency).
#' 
#' @param prompt Character string with the user prompt
#' @param system.role Character string with the system prompt
#' @param model Character string specifying the Claude model
#' @param temperature Numeric. Sampling temperature (0-1)
#' @param top.p Numeric. Nucleus sampling parameter (0-1)
#' @param max_tokens Integer. Maximum tokens to generate
#' @param api_key Anthropic API key
#' 
#' @return Character string with the generated text
#' @keywords internal
generate_text_anthropic <- function(prompt, system.role = NULL,
                                     model = "claude-sonnet-4-5-20250929",
                                     temperature = 1, top.p = 1,
                                     max_tokens = 2048, api_key) {
  
  ensure_aigenie_python()
  
  json_mod <- reticulate::import("json")
  requests <- reticulate::import("requests")
  
  # Build headers (Anthropic uses x-api-key, not Bearer token)
  headers <- list(
    "x-api-key"        = api_key,
    "anthropic-version" = "2023-06-01",
    "content-type"      = "application/json"
  )
  
  # Build messages (Anthropic: system is top-level, not in messages array)
  messages <- list(
    list(role = "user", content = prompt)
  )
  
  # Build request body
  body <- list(
    model      = model,
    max_tokens = as.integer(max_tokens),
    messages   = messages
  )
  
  # Add system prompt if provided (top-level parameter in Anthropic API)
  if (!is.null(system.role) && nchar(system.role) > 0) {
    body$system <- system.role
  }
  
  # Add sampling parameters
  if (temperature != 1) {
    body$temperature <- temperature
  }
  if (top.p != 1) {
    body$top_p <- top.p
  }
  
  # Serialize to JSON via Python for proper list-of-lists handling
  body_json <- json_mod$dumps(body)
  
  # Make request
  response <- requests$post(
    "https://api.anthropic.com/v1/messages",
    headers = headers,
    data = body_json
  )
  
  if (response$status_code != 200L) {
    stop("Anthropic API error (", response$status_code, "): ",
         response$text, call. = FALSE)
  }
  
  result <- response$json()
  
  # Extract text from content blocks
  # Anthropic returns: {content: [{type: "text", text: "..."}]}
  content <- result$content
  
  if (is.null(content) || length(content) == 0) {
    stop("Anthropic API returned empty content.", call. = FALSE)
  }
  
  # Concatenate all text blocks
  text_parts <- vapply(content, function(block) {
    if (!is.null(block$type) && block$type == "text" && !is.null(block$text)) {
      return(block$text)
    }
    return("")
  }, character(1))
  
  return(paste(text_parts, collapse = "\n"))
}

#' Generate Text Using HuggingFace Inference API
#' 
#' @param prompt Character string with the user prompt
#' @param system.role Character string with the system prompt
#' @param model Character string specifying the HuggingFace model ID
#' @param temperature Numeric. Sampling temperature
#' @param top.p Numeric. Nucleus sampling parameter
#' @param max_tokens Integer. Maximum tokens to generate
#' @param hf_token Optional HuggingFace token
#' 
#' @return Character string with the generated text
#' @keywords internal
generate_text_huggingface <- function(prompt, system.role = NULL, model,
                                       temperature = 1, top.p = 1, max_tokens = 2048,
                                       hf_token = NULL) {
  
  ensure_aigenie_python()
  
  requests <- reticulate::import("requests")
  
  # Build the full prompt
  full_prompt <- ""
  if (!is.null(system.role) && nchar(system.role) > 0) {
    full_prompt <- paste0("System: ", system.role, "\n\nUser: ", prompt, "\n\nAssistant:")
  } else {
    full_prompt <- prompt
  }
  
  # Build API URL
  api_url <- paste0("https://api-inference.huggingface.co/models/", model)
  
  # Build headers
  headers <- list("Content-Type" = "application/json")
  if (!is.null(hf_token)) {
    headers[["Authorization"]] <- paste("Bearer", hf_token)
  }
  
  # Build payload
  payload <- list(
    inputs = full_prompt,
    parameters = list(
      temperature = temperature,
      top_p = top.p,
      max_new_tokens = as.integer(max_tokens),
      return_full_text = FALSE
    )
  )
  
  # Make request with retry logic
  max_retries <- 3
  for (attempt in seq_len(max_retries)) {
    response <- requests$post(
      api_url,
      headers = headers,
      json = payload
    )
    
    if (response$status_code == 200L) {
      result <- response$json()
      if (is.list(result) && length(result) > 0) {
        if (!is.null(result[[1]]$generated_text)) {
          return(result[[1]]$generated_text)
        }
      }
      return(as.character(result))
    } else if (response$status_code == 503L) {
      # Model loading, wait and retry
      Sys.sleep(10 * attempt)
    } else {
      if (attempt == max_retries) {
        stop("HuggingFace API error (", response$status_code, "): ",
             response$text, call. = FALSE)
      }
      Sys.sleep(2 * attempt)
    }
  }
  
  stop("Failed to get response from HuggingFace after ", max_retries, " attempts",
       call. = FALSE)
}

# ============================================================================
# Available Models Information
# ============================================================================

#' List Available Models
#' 
#' @description
#' Queries the OpenAI, Groq, Anthropic, and/or Jina AI APIs to retrieve 
#' currently available models. Requires API keys for live provider queries.
#' Jina AI models are returned from a curated static list (no list endpoint).
#' 
#' @param provider Optional. Filter by provider: "openai", "groq", "anthropic",
#'   "jina", or NULL for all.
#' @param openai.API Optional OpenAI API key. If NULL, checks OPENAI_API_KEY env var.
#' @param groq.API Optional Groq API key. If NULL, checks GROQ_API_KEY env var.
#' @param anthropic.API Optional Anthropic API key. If NULL, checks ANTHROPIC_API_KEY env var.
#' @param type Filter by model type: "chat", "embedding", or NULL for all.
#'   Default is NULL (show everything).
#' 
#' @return A data frame with columns: provider, model, type, display_name, created
#' @export
list_available_models <- function(provider = NULL,
                                   openai.API = NULL,
                                   groq.API = NULL,
                                   anthropic.API = NULL,
                                   type = NULL) {
  
  # Resolve API keys from environment variables
  if (is.null(openai.API)) {
    openai.API <- Sys.getenv("OPENAI_API_KEY", unset = NA)
    if (is.na(openai.API)) openai.API <- NULL
  }
  if (is.null(groq.API)) {
    groq.API <- Sys.getenv("GROQ_API_KEY", unset = NA)
    if (is.na(groq.API)) groq.API <- NULL
  }
  if (is.null(anthropic.API)) {
    anthropic.API <- Sys.getenv("ANTHROPIC_API_KEY", unset = NA)
    if (is.na(anthropic.API)) anthropic.API <- NULL
  }
  
  # Validate type parameter
  if (!is.null(type)) {
    type <- tolower(type)
    if (!type %in% c("chat", "embedding")) {
      stop("'type' must be 'chat', 'embedding', or NULL.", call. = FALSE)
    }
  }
  
  # Initialize empty result
  empty_df <- data.frame(
    provider = character(),
    model = character(),
    type = character(),
    display_name = character(),
    created = character(),
    stringsAsFactors = FALSE
  )
  all_models <- empty_df
  
  # Normalize provider for matching
  prov <- if (!is.null(provider)) tolower(provider) else NULL
  
  # --- OpenAI ---
  if (is.null(prov) || prov == "openai") {
    if (!is.null(openai.API)) {
      df <- fetch_openai_models(openai.API)
      if (!is.null(df) && nrow(df) > 0) all_models <- rbind(all_models, df)
    } else if (is.null(prov)) {
      message("No OpenAI API key found. Skipping OpenAI models.")
    } else {
      stop("OpenAI API key required. Provide openai.API or set OPENAI_API_KEY.", call. = FALSE)
    }
  }
  
  # --- Groq ---
  if (is.null(prov) || prov == "groq") {
    if (!is.null(groq.API)) {
      df <- fetch_groq_models(groq.API)
      if (!is.null(df) && nrow(df) > 0) all_models <- rbind(all_models, df)
    } else if (is.null(prov)) {
      message("No Groq API key found. Skipping Groq models.")
    } else {
      stop("Groq API key required. Provide groq.API or set GROQ_API_KEY.", call. = FALSE)
    }
  }
  
  # --- Anthropic ---
  if (is.null(prov) || prov == "anthropic") {
    if (!is.null(anthropic.API)) {
      df <- fetch_anthropic_models(anthropic.API)
      if (!is.null(df) && nrow(df) > 0) all_models <- rbind(all_models, df)
    } else if (is.null(prov)) {
      message("No Anthropic API key found. Skipping Anthropic models.")
    } else {
      stop("Anthropic API key required. Provide anthropic.API or set ANTHROPIC_API_KEY.", call. = FALSE)
    }
  }
  
  # --- Jina AI (static list, no API key needed to list) ---
  if (is.null(prov) || prov == "jina") {
    df <- get_jina_models()
    if (!is.null(df) && nrow(df) > 0) all_models <- rbind(all_models, df)
  }
  
  if (nrow(all_models) == 0) {
    message("No models retrieved. Provide API keys or check your connection.")
    return(empty_df)
  }
  
  # Apply type filter
  if (!is.null(type)) {
    all_models <- all_models[all_models$type == type, ]
  }
  
  # Sort by provider then model name
  all_models <- all_models[order(all_models$provider, all_models$type, all_models$model), ]
  rownames(all_models) <- NULL
  
  return(all_models)
}


# ============================================================================
# Provider-Specific Fetch Functions
# ============================================================================

#' Fetch Available Models from OpenAI API
#' 
#' @param api_key OpenAI API key
#' @return Data frame of models or NULL on failure
#' @keywords internal
fetch_openai_models <- function(api_key) {
  
  ensure_aigenie_python()
  requests <- reticulate::import("requests")
  
  tryCatch({
    response <- requests$get(
      "https://api.openai.com/v1/models",
      headers = list(
        "Authorization" = paste("Bearer", api_key),
        "Content-Type" = "application/json"
      )
    )
    
    if (response$status_code != 200L) {
      warning("OpenAI API returned status ", response$status_code, call. = FALSE)
      return(NULL)
    }
    
    result <- response$json()
    model_data <- result$data
    if (length(model_data) == 0) return(NULL)
    
    # Embedding model patterns
    embedding_patterns <- c("^text-embedding", "^embedding")
    embedding_regex <- paste(embedding_patterns, collapse = "|")
    
    # Non-model patterns to exclude entirely
    exclude_patterns <- c(
      "^tts-", "^whisper", "^dall-e", "^davinci", "^babbage",
      "^curie", "^ada", "moderation", "^ft:", "search",
      "instruct(?!.*gpt)", "code-", "text-davinci",
      "text-curie", "text-babbage", "text-ada"
    )
    exclude_regex <- paste(exclude_patterns, collapse = "|")
    
    df <- do.call(rbind, lapply(model_data, function(m) {
      mid <- as.character(if (!is.null(m$id)) m$id else "")
      data.frame(
        provider = "openai",
        model = mid,
        type = ifelse(grepl(embedding_regex, mid, ignore.case = TRUE), "embedding", "chat"),
        display_name = mid,
        created = format(
          as.POSIXct(as.numeric(if (!is.null(m$created)) m$created else 0),
                     origin = "1970-01-01"),
          "%Y-%m-%d"
        ),
        stringsAsFactors = FALSE
      )
    }))
    
    # Remove non-model entries (tts, whisper, dall-e, legacy, etc.)
    df <- df[!grepl(exclude_regex, df$model, ignore.case = TRUE, perl = TRUE), ]
    
    return(df)
    
  }, error = function(e) {
    warning("Failed to fetch OpenAI models: ", conditionMessage(e), call. = FALSE)
    return(NULL)
  })
}


#' Fetch Available Models from Groq API
#' 
#' @param api_key Groq API key
#' @return Data frame of models or NULL on failure
#' @keywords internal
fetch_groq_models <- function(api_key) {
  
  ensure_aigenie_python()
  requests <- reticulate::import("requests")
  
  tryCatch({
    response <- requests$get(
      "https://api.groq.com/openai/v1/models",
      headers = list(
        "Authorization" = paste("Bearer", api_key),
        "Content-Type" = "application/json"
      )
    )
    
    if (response$status_code != 200L) {
      warning("Groq API returned status ", response$status_code, call. = FALSE)
      return(NULL)
    }
    
    result <- response$json()
    model_data <- result$data
    if (length(model_data) == 0) return(NULL)
    
    # Patterns for non-chat models
    exclude_patterns <- c("^whisper", "^text-embedding", "^embedding", "^tts-")
    exclude_regex <- paste(exclude_patterns, collapse = "|")
    
    df <- do.call(rbind, lapply(model_data, function(m) {
      mid <- as.character(if (!is.null(m$id)) m$id else "")
      is_active <- as.logical(if (!is.null(m$active)) m$active else TRUE)
      data.frame(
        provider = "groq",
        model = mid,
        type = "chat",
        display_name = mid,
        created = format(
          as.POSIXct(as.numeric(if (!is.null(m$created)) m$created else 0),
                     origin = "1970-01-01"),
          "%Y-%m-%d"
        ),
        .active = is_active,
        stringsAsFactors = FALSE
      )
    }))
    
    # Filter out non-chat, inactive, and drop temp column
    df <- df[!grepl(exclude_regex, df$model, ignore.case = TRUE), ]
    df <- df[df$.active == TRUE, ]
    df$.active <- NULL
    
    return(df)
    
  }, error = function(e) {
    warning("Failed to fetch Groq models: ", conditionMessage(e), call. = FALSE)
    return(NULL)
  })
}


#' Fetch Available Models from Anthropic API
#' 
#' @description
#' Queries the Anthropic /v1/models endpoint with pagination support.
#' 
#' @param api_key Anthropic API key
#' @return Data frame of models or NULL on failure
#' @keywords internal
fetch_anthropic_models <- function(api_key) {
  
  ensure_aigenie_python()
  requests <- reticulate::import("requests")
  
  tryCatch({
    all_models <- list()
    has_more <- TRUE
    after_id <- NULL
    
    while (has_more) {
      # Build URL with pagination
      url <- "https://api.anthropic.com/v1/models?limit=100"
      if (!is.null(after_id)) {
        url <- paste0(url, "&after_id=", after_id)
      }
      
      response <- requests$get(
        url,
        headers = list(
          "x-api-key" = api_key,
          "anthropic-version" = "2023-06-01",
          "Content-Type" = "application/json"
        )
      )
      
      if (response$status_code != 200L) {
        warning("Anthropic API returned status ", response$status_code, call. = FALSE)
        return(NULL)
      }
      
      result <- response$json()
      page_data <- result$data
      
      if (length(page_data) > 0) {
        all_models <- c(all_models, page_data)
      }
      
      has_more <- isTRUE(result$has_more)
      if (has_more && !is.null(result$last_id)) {
        after_id <- result$last_id
      } else {
        has_more <- FALSE
      }
    }
    
    if (length(all_models) == 0) return(NULL)
    
    df <- do.call(rbind, lapply(all_models, function(m) {
      mid <- as.character(if (!is.null(m$id)) m$id else "")
      dname <- as.character(if (!is.null(m$display_name)) m$display_name else mid)
      created <- if (!is.null(m$created_at)) {
        tryCatch(
          format(as.POSIXct(m$created_at, format = "%Y-%m-%dT%H:%M:%SZ"), "%Y-%m-%d"),
          error = function(e) ""
        )
      } else ""
      
      data.frame(
        provider = "anthropic",
        model = mid,
        type = "chat",
        display_name = dname,
        created = created,
        stringsAsFactors = FALSE
      )
    }))
    
    return(df)
    
  }, error = function(e) {
    warning("Failed to fetch Anthropic models: ", conditionMessage(e), call. = FALSE)
    return(NULL)
  })
}


#' Get Jina AI Embedding Models
#' 
#' @description
#' Returns a curated list of Jina AI embedding models. Jina does not provide
#' a model listing API endpoint, so this list is maintained manually.
#' 
#' @return Data frame of Jina AI embedding models
#' @keywords internal
get_jina_models <- function() {
  
  data.frame(
    provider = "jina",
    model = c(
      # Current flagship models
      "jina-embeddings-v4",
      "jina-embeddings-v3",
      "jina-clip-v2",
      # Code embedding models
      "jina-code-embeddings-1.5b",
      "jina-code-embeddings-0.5b",
      # V2 family (still supported)
      "jina-embeddings-v2-base-en",
      "jina-embeddings-v2-base-zh",
      "jina-embeddings-v2-base-de",
      "jina-embeddings-v2-base-es",
      "jina-embeddings-v2-base-code",
      "jina-embeddings-v2-small-en"
    ),
    type = "embedding",
    display_name = c(
      "Jina Embeddings v4 (3.8B, multimodal, 2048d)",
      "Jina Embeddings v3 (570M, multilingual, 1024d)",
      "Jina CLIP v2 (885M, text-image, 1024d)",
      "Jina Code Embeddings 1.5B",
      "Jina Code Embeddings 0.5B",
      "Jina Embeddings v2 Base EN (137M, 768d)",
      "Jina Embeddings v2 Base ZH (137M, 768d)",
      "Jina Embeddings v2 Base DE (137M, 768d)",
      "Jina Embeddings v2 Base ES (137M, 768d)",
      "Jina Embeddings v2 Base Code (137M, 768d)",
      "Jina Embeddings v2 Small EN (33M, 512d)"
    ),
    created = c(
      "2025-06-24", "2024-09-18", "2024-12-12",
      "2025-07-15", "2025-07-15",
      "2023-10-30", "2023-10-30", "2023-10-30",
      "2023-10-30", "2023-10-30", "2023-10-30"
    ),
    stringsAsFactors = FALSE
  )
}
