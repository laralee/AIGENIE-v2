# Validate Booleans ----

#' Validate Boolean Arguments
#'
#' Validates that all arguments passed to the function are scalar boolean values (`TRUE` or `FALSE`).
#' If any argument is not a boolean, an error is thrown that identifies the offending variable
#' by name and instructs the user to set it to either `TRUE` or `FALSE`.
#'
#' @param ... One or more variables to check. We are expecting each to be a logical scalar (`TRUE` or `FALSE`). In `AIGENIE`, these variables would be `items.only`, `adaptive`, `plot`, `keep.org`, `silently`, and `embeddings.only`.
#'
validate_booleans <- function(...) {
  args <- list(...)
  calls <- as.list(match.call(expand.dots = FALSE)$...)

  for (i in seq_along(args)) {
    val <- args[[i]]
    name <- deparse(calls[[i]])

    if (!is.logical(val) || length(val) != 1 || is.na(val)) {
      # Clean, user-facing error message
      message <- paste0(
        "AI-GENIE expects '", name,
        "' to be a boolean. Set '", name,
        "' to either TRUE or FALSE."
      )

      stop(simpleError(message))
    }
  }
}






# Validate strings ----
#' Validate That Inputs Are Strings
#'
#' Ensures that each argument is a single, non-NA string. Throws an error
#' if any argument is not a character scalar.
#'
#' @param ... One or more variables to validate.
#'
validate_strings <- function(...) {
  args <- list(...)
  calls <- as.list(match.call(expand.dots = FALSE)$...)

  for (i in seq_along(args)) {
    val <- args[[i]]
    name <- deparse(calls[[i]])

    if (!is.null(val)) {
      if (!is.character(val) || length(val) != 1 || is.na(val) || val == "") {
        stop(
          paste0(
            "AI-GENIE expects ", name,
            " to be a non-empty string. Set ", name,
            " to a valid string value."
          ),
          call. = FALSE
        )
      }
    }
  }
}


# Validate `items.attributes` ----

#' Validate `items.attributes`
#'
#' Validates that `items.attributes` is a **named list** whose names are
#' truly unique after trimming whitespace and ignoring case, and that each
#' element is itself a list **containing only strings**, with **at least two**
#' truly unique strings (same trimming + case-insensitive rule).
#'
#' @param items.attributes A named list. Each element must be a list containing
#'   only character scalars (strings). Each of those inner lists must contain
#'   at least two truly unique strings after trimming and case-folding.
#' @return A cleaned version of `items.attributes` with normalized names and
#'   values. Errors are thrown if validation fails.
#'
items.attributes_validate <- function(items.attributes) {
  norm_str <- function(x) trimws(tolower(x))

  # ---- Top-level: must be a named list ----
  if (!is.list(items.attributes)) {
    stop(
      "AI-GENIE expects items.attributes to be a named list.",
      call. = FALSE
    )
  }

  nm <- names(items.attributes)
  if (is.null(nm) || any(is.na(nm)) || any(nm == "")) {
    stop(
      "AI-GENIE expects items.attributes to be a named list with non-empty names.",
      call. = FALSE
    )
  }

  # ---- Normalize top-level names ----
  nm_norm <- norm_str(nm)

  if (any(duplicated(nm_norm))) {
    dups <- unique(nm_norm[duplicated(nm_norm)])
    msg_lines <- lapply(dups, function(key) {
      originals <- nm[nm_norm == key]
      paste0("• Names ", paste(sprintf("`%s`", originals), collapse = ", "),
             " normalize to `", key, "`")
    })
    stop(
      paste0(
        "AI-GENIE expects items.attributes to have unique names after trimming and case-folding.\n",
        "The following names collide:\n",
        paste(unlist(msg_lines), collapse = "\n")
      ),
      call. = FALSE
    )
  }

  names(items.attributes) <- nm_norm

  # ---- Validate + Clean each element ----
  cleaned <- list()

  for (i in seq_along(items.attributes)) {
    top_name <- names(items.attributes)[i]
    value <- items.attributes[[i]]

    # Normalize into a character vector
    vals_char <- NULL

    if (is.list(value)) {
      vals_char <- vapply(value, function(x) {
        if (!is.character(x) || length(x) != 1L || is.na(x)) {
          stop(
            paste0(
              "AI-GENIE expects items.attributes$", top_name,
              " to contain only strings."
            ),
            call. = FALSE
          )
        }
        x
      }, character(1))
    } else if (is.atomic(value)) {
      if (!is.character(value)) {
        stop(
          paste0(
            "AI-GENIE expects items.attributes$", top_name,
            " to be either a list of strings or a character vector."
          ),
          call. = FALSE
        )
      }
      if (any(is.na(value))) {
        stop(
          paste0(
            "AI-GENIE expects items.attributes$", top_name,
            " to contain no NA strings."
          ),
          call. = FALSE
        )
      }
      vals_char <- as.character(value)
    } else {
      stop(
        paste0(
          "AI-GENIE expects items.attributes$", top_name,
          " to be either a list of strings or a character vector."
        ),
        call. = FALSE
      )
    }

    # ---- Clean values ----
    vals_clean <- norm_str(vals_char)

    if (any(vals_clean == "")) {
      stop(
        paste0(
          "AI-GENIE expects items.attributes$", top_name,
          " to contain no empty strings (after trimming)."
        ),
        call. = FALSE
      )
    }

    vals_unique <- unique(vals_clean)

    if (length(vals_unique) < 2L) {
      stop(
        paste0(
          "AI-GENIE expects items.attributes$", top_name,
          " to contain at least two truly unique strings (after trim + case-fold)."
        ),
        call. = FALSE
      )
    }

    # Save cleaned version
    cleaned[[top_name]] <- vals_unique
  }

  return(cleaned)
}











# Validate `item.examples` ----
#' Validate and Clean `item.examples` Against Cleaned `items.attributes`
#'
#' Ensures `item.examples` is a data frame with required string columns and that the values
#' in `type` and `attribute` align with the cleaned structure of `items.attributes`.
#' Returns a cleaned version of the data frame with normalized values:
#'   - `type` and `attribute` are trimmed and lowercased
#'   - `statement` is trimmed (case preserved)
#'
#' @param item.examples A data frame with columns `type`, `attribute`, `statement`.
#'   All values must be non-empty strings.
#' @param items.attributes A cleaned list from `validate_items.attributes()`.
#'   All names and values must be normalized (lowercased and trimmed).
#'
#' @return A cleaned version of `item.examples` with normalized values.
#'
item.examples_validate <- function(item.examples, items.attributes) {
  norm_str <- function(x) trimws(tolower(x))
  trim_str <- function(x) trimws(x)

  # ---- Check structure ----
  if (!is.data.frame(item.examples)) {
    stop(
      "AI-GENIE expects item.examples to be a data frame with columns `type`, `attribute`, `statement`.",
      call. = FALSE
    )
  }

  required_cols <- c("type", "attribute", "statement")
  missing_cols <- setdiff(required_cols, names(item.examples))
  if (length(missing_cols) > 0) {
    stop(
      paste0(
        "AI-GENIE expects item.examples to contain columns: `type`, `attribute`, `statement`. ",
        "Missing: ", paste(missing_cols, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---- Check column types and contents ----
  for (col in required_cols) {
    v <- item.examples[[col]]
    if (!is.character(v)) {
      stop(
        paste0("AI-GENIE expects item.examples$", col, " to contain strings."),
        call. = FALSE
      )
    }
    if (any(is.na(v))) {
      stop(
        paste0("AI-GENIE expects item.examples$", col, " to contain no NA values."),
        call. = FALSE
      )
    }
    if (any(trimws(v) == "")) {
      stop(
        paste0("AI-GENIE expects item.examples$", col, " to contain no empty strings."),
        call. = FALSE
      )
    }
  }

  # ---- Normalize all columns ----
  cleaned <- data.frame(
    type = norm_str(item.examples$type),
    attribute = norm_str(item.examples$attribute),
    statement = trim_str(item.examples$statement),
    stringsAsFactors = FALSE
  )

  # ---- Validate type exists ----
  known_types <- names(items.attributes)
  unknown_types <- setdiff(unique(cleaned$type), known_types)
  if (length(unknown_types) > 0) {
    offending <- unique(item.examples$type[cleaned$type %in% unknown_types])
    stop(
      paste0(
        "AI-GENIE expects each value in item.examples$type to match a name in items.attributes ",
        "(after trimming and case-folding). Invalid types: ",
        paste(sprintf("`%s`", offending), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---- Validate attribute exists per row ----
  bad_rows <- integer(0)
  for (i in seq_len(nrow(cleaned))) {
    t <- cleaned$type[i]
    a <- cleaned$attribute[i]
    allowed <- items.attributes[[t]]

    if (!(a %in% allowed)) {
      bad_rows <- c(bad_rows, i)
    }
  }

  if (length(bad_rows) > 0) {
    max_show <- min(length(bad_rows), 10)
    preview <- paste0(
      "• Row ", bad_rows[1:max_show], ": type = `",
      item.examples$type[bad_rows[1:max_show]], "`, attribute = `",
      item.examples$attribute[bad_rows[1:max_show]], "`"
    )
    extra <- if (length(bad_rows) > max_show) {
      paste0("\n… and ", length(bad_rows) - max_show, " more row(s).")
    } else ""

    stop(
      paste0(
        "AI-GENIE expects each item.examples$attribute to belong to its corresponding item.examples$type, ",
        "using cleaned items.attributes as reference.\nInvalid rows:\n",
        paste(preview, collapse = "\n"),
        extra
      ),
      call. = FALSE
    )
  }

  return(cleaned)
}





# Validate `item.type.definitions` ----
#' Validate and Clean `item.type.definitions`
#'
#' Validates that `item.type.definitions` is a named list where:
#'   - Names are unique (after trim + case-fold)
#'   - Names exist in `items.attributes`
#'   - Values are non-empty strings
#'
#' Returns a cleaned version with:
#'   - Normalized names (trimmed and lowercased)
#'   - Trimmed values (case preserved)
#'
#' @param item.type.definitions A named list of strings, where each name must
#'   correspond to a name in `items.attributes` and each value must be a
#'   non-empty string.
#' @param items.attributes A cleaned list from `validate_items.attributes()`.
#'
#' @return A cleaned version of `item.type.definitions`.
item.type.definitions_validate <- function(item.type.definitions, items.attributes) {
  norm_str <- function(x) trimws(tolower(x))
  trim_str <- function(x) trimws(x)

  # ---- Check type ----
  if (!is.list(item.type.definitions)) {
    stop(
      "AI-GENIE expects item.type.definitions to be a named list.",
      call. = FALSE
    )
  }

  names_raw <- names(item.type.definitions)
  if (is.null(names_raw) || any(is.na(names_raw)) || any(names_raw == "")) {
    stop(
      "AI-GENIE expects item.type.definitions to have non-empty names.",
      call. = FALSE
    )
  }

  names_norm <- norm_str(names_raw)

  # ---- Check name uniqueness ----
  if (any(duplicated(names_norm))) {
    dups <- unique(names_norm[duplicated(names_norm)])
    msg_lines <- lapply(dups, function(key) {
      originals <- names_raw[names_norm == key]
      paste0("• Names ", paste(sprintf("`%s`", originals), collapse = ", "),
             " normalize to `", key, "`")
    })
    stop(
      paste0(
        "AI-GENIE expects item.type.definitions to have unique names after trimming and case-folding.\n",
        "The following names collide:\n",
        paste(unlist(msg_lines), collapse = "\n")
      ),
      call. = FALSE
    )
  }

  # ---- Check names match items.attributes ----
  attr_names <- names(items.attributes)
  unknown_names <- setdiff(names_norm, attr_names)
  if (length(unknown_names) > 0) {
    offending <- names_raw[names_norm %in% unknown_names]
    stop(
      paste0(
        "AI-GENIE expects every name in item.type.definitions to match a name in items.attributes ",
        "(after trimming and case-folding). Invalid name(s): ",
        paste(sprintf("`%s`", offending), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---- Validate values ----
  cleaned <- list()
  for (i in seq_along(item.type.definitions)) {
    value <- item.type.definitions[[i]]
    name_norm <- names_norm[i]

    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(
        paste0(
          "AI-GENIE expects item.type.definitions$", names_raw[i],
          " to be a non-empty string."
        ),
        call. = FALSE
      )
    }

    val_trim <- trim_str(value)
    if (val_trim == "") {
      stop(
        paste0(
          "AI-GENIE expects item.type.definitions$", names_raw[i],
          " to be a non-empty string (after trimming)."
        ),
        call. = FALSE
      )
    }

    cleaned[[name_norm]] <- val_trim
  }

  return(cleaned)
}

# Validate embedding model and model ----

#' Resolve and Normalize Model Name
#'
#' Accepts a free-form model name and returns a standardized string.
#' Known aliases are resolved to canonical model names.
#' If the model is not recognized, a warning is issued and the cleaned
#' original input is returned.
#'
#' @param model A single string, the user-supplied model name.
#' @param silently A flag to determine if warnings should be printed to the screen.
#'
#' @return A standardized model name.
resolve_model_name <- function(model, silently) {
  if (is.null(model) || !is.character(model) || length(model) != 1 || is.na(model)) {
    stop("AI-GENIE expects `model` to be a non-empty string.", call. = FALSE)
  }

  model_trimmed <- trimws(model)
  model_clean <- tolower(gsub("[^a-z0-9]", "", model_trimmed))  # Remove spaces, hyphens, dots, etc.

  # Canonical models
  canonical_models <- c(
    "gpt-5",
    "gpt-4o",
    "gpt-4.1",
    "gpt-3.5-turbo",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "deepseek-r1-distill-llama-70b",
    "meta-llama/Llama-3.3-70B-Instruct",
    "gemma2-9b-it"
  )

  # Alias mapping
  alias_map <- list(
    gpt5       = "gpt-5",
    gpt4o      = "gpt-4o",
    gpt41      = "gpt-4.1",
    gpt35      = "gpt-3.5-turbo",
    gpt35turbo = "gpt-3.5-turbo",
    gpt3dot5   = "gpt-3.5-turbo",
    oss20b     = "openai/gpt-oss-20b",
    oss120b    = "openai/gpt-oss-120b",
    deepseek   = "deepseek-r1-distill-llama-70b",
    llama      = "meta-llama/Llama-3.3-70B-Instruct",
    gemma      = "gemma2-9b-it"
  )

  # 1. Exact match to canonical model → return directly
  if (tolower(model_trimmed) %in% tolower(canonical_models)) {
    return(model_trimmed)
  }

  # 2. Starts with canonical model → assume versioned variant
  for (canon in canonical_models) {
    if (startsWith(tolower(model_trimmed), tolower(canon))) {
      return(model_trimmed)
    }
  }

  # 3. Starts with alias → return mapped canonical model
  for (alias in names(alias_map)) {
    if (startsWith(model_clean, alias)) {
      return(alias_map[[alias]])
    }
  }

  # 4. Unchanged input — warn and return as-is
  if(!silently){
    warning(paste0("AI-GENIE does not recognize the model `", model_trimmed, "`. Proceeding as-is."))
  }
    return(model_trimmed)
}


#' Validate OpenAI Embedding Model
#'
#' Validates that the embedding model is one of the supported OpenAI models.
#' Allowed values are:
#'   - "text-embedding-3-small"
#'   - "text-embedding-3-large"
#'   - "text-embedding-ada-002"
#'
#'
#' @param embedding.model A string or `NULL`.
#'
embedding.model_validate <- function(embedding.model) {
  allowed <- c(
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
  )

  if (!is.character(embedding.model) || length(embedding.model) != 1 || is.na(embedding.model)) {
    stop(
      "AI-GENIE expects embedding.model to be a string. Set it to a valid OpenAI embedding model.",
      call. = FALSE
    )
  }

  if (!(embedding.model %in% allowed)) {
    stop(
      paste0(
        "AI-GENIE expects embedding.model to be one of the following: ",
        paste(sprintf("`%s`", allowed), collapse = ", "),
        ". Received: `", embedding.model, "`."
      ),
      call. = FALSE
    )
  }

}





# Validate EGA parameters ----
#' Validate EGA Parameters
#'
#' Validates and normalizes the EGA algorithm, unidimensionality method, and model parameters.
#' Trims whitespace and performs case-insensitive matching. Returns canonical-cased values.
#'
#' @param EGA.algorithm A string: one of "leiden", "louvain", "walktrap"
#' @param EGA.uni.method A string: one of "expand", "LE", "louvain"
#' @param EGA_model A string or NULL: one of "glasso", "TMFG"
#'
#' @return A named list with cleaned and correctly-cased values.
validate_ega_params <- function(EGA.algorithm, EGA.uni.method, EGA_model) {
  norm_str <- function(x) tolower(trimws(x))

  # Canonical sets
  ALGORITHMS <- c("leiden", "louvain", "walktrap")
  UNI_METHODS <- c("expand", "LE", "louvain")
  MODELS <- c("glasso", "TMFG")

  # Build matchers: lowercase names → canonical casing
  algorithm_map <- setNames(ALGORITHMS, tolower(ALGORITHMS))
  uni_method_map <- setNames(UNI_METHODS, tolower(UNI_METHODS))
  model_map <- setNames(MODELS, tolower(MODELS))

  # --- Validate algorithm ---
  if (!is.character(EGA.algorithm) || length(EGA.algorithm) != 1 || is.na(EGA.algorithm)) {
    stop("AI-GENIE expects EGA.algorithm to be a non-empty string.", call. = FALSE)
  }

  algo_key <- norm_str(EGA.algorithm)
  if (!algo_key %in% names(algorithm_map)) {
    stop(
      paste0(
        "AI-GENIE expects EGA.algorithm to be one of: ",
        paste(sprintf("`%s`", ALGORITHMS), collapse = ", "),
        ". Received: `", EGA.algorithm, "`."
      ),
      call. = FALSE
    )
  }

  # --- Validate uni_method ---
  if (!is.character(EGA.uni.method) || length(EGA.uni.method) != 1 || is.na(EGA.uni.method)) {
    stop("AI-GENIE expects EGA.uni.method to be a non-empty string.", call. = FALSE)
  }

  uni_key <- norm_str(EGA.uni.method)
  if (!uni_key %in% names(uni_method_map)) {
    stop(
      paste0(
        "AI-GENIE expects EGA.uni.method to be one of: ",
        paste(sprintf("`%s`", UNI_METHODS), collapse = ", "),
        ". Received: `", EGA.uni.method, "`."
      ),
      call. = FALSE
    )
  }

  # --- Validate model ---
  model_cleaned <- NULL
  if (is.null(EGA_model)) {
    model_cleaned <- NULL
  } else if (!is.character(EGA_model) || length(EGA_model) != 1 || is.na(EGA_model)) {
    stop("AI-GENIE expects EGA_model to be a string or NULL.", call. = FALSE)
  } else {
    model_key <- norm_str(EGA_model)
    if (!model_key %in% names(model_map)) {
      stop(
        paste0(
          "AI-GENIE expects EGA_model to be one of: ",
          paste(sprintf("`%s`", MODELS), collapse = ", "),
          ", or NULL. Received: `", EGA_model, "`."
        ),
        call. = FALSE
      )
    }
    model_cleaned <- model_map[[model_key]]
  }

  return(list(
    EGA.algorithm = algorithm_map[[algo_key]],
    EGA.uni.method = uni_method_map[[uni_key]],
    EGA_model = model_cleaned
  ))
}


# Validate target N ----
#' Validate and Expand `target.N` for Each Item Attribute
#'
#' Ensures that `target.N` is either:
#'   - NULL → defaults to 60 per attribute
#'   - A single integer → repeated for each attribute
#'   - A list/vector of integers → must match number of attributes
#'
#' @param target.N An integer, list/vector of integers, or NULL.
#' @param items.attributes A cleaned list returned from `validate_items.attributes()`.
#' @param items.only A flag used to determine if only items need to be generated
#' @param embeddings.only A flag used to determine if only embeddings need to be generated
#' @param silently A flag used to determine if warnings should be printed
#'
#' @return A list of integers, one per attribute (named).
target.N_validate <- function(target.N, items.attributes, items.only, embeddings.only, silently) {
  n_attr <- length(items.attributes)
  attr_names <- names(items.attributes)
  norm_str <- function(x) trimws(tolower(x))

  # --- Case: NULL → default to 60s ---
  if (is.null(target.N)) {
    default_list <- setNames(as.list(rep(60L, n_attr)), attr_names)
    target.N <- default_list
  }

  # --- Helper: scalar integer check ---
  is_scalar_int <- function(x) {
    is.numeric(x) && length(x) == 1 && !is.na(x) && x == as.integer(x)
  }

  # --- Case: single integer → expand it ---
  if (is_scalar_int(target.N)) {
    target.N <- setNames(as.list(rep(as.integer(target.N), n_attr)), attr_names)
  }

  # --- Case: named list/vector ---
  if (is.atomic(target.N) || is.list(target.N)) {
    vec <- unlist(target.N, use.names = TRUE)

    if (is.null(names(vec)) || any(names(vec) == "") || any(is.na(names(vec)))) {
      stop("AI-GENIE expects target.N to be a named list or vector if not scalar or NULL.", call. = FALSE)
    }

    vec_names_norm <- norm_str(names(vec))
    attr_names_norm <- norm_str(attr_names)

    if (!setequal(vec_names_norm, attr_names_norm)) {
      missing <- setdiff(attr_names_norm, vec_names_norm)
      extra   <- setdiff(vec_names_norm, attr_names_norm)

      msg <- "AI-GENIE expects target.N to provide one integer for *every* item_attribute."
      if (length(missing) > 0) {
        msg <- paste0(msg, "\nMissing types: ", paste(sprintf("`%s`", missing), collapse = ", "))
      }
      if (length(extra) > 0) {
        msg <- paste0(msg, "\nUnexpected types: ", paste(sprintf("`%s`", extra), collapse = ", "))
      }

      stop(msg, call. = FALSE)
    }

    if (!is.numeric(vec) || any(is.na(vec)) || any(vec != as.integer(vec))) {
      stop("AI-GENIE expects all values in target.N to be non-NA integers.", call. = FALSE)
    }

    vec_int <- as.integer(vec)
    normalized <- setNames(vec_int, vec_names_norm)
    ordered <- normalized[norm_str(attr_names)]
    target.N <- setNames(as.list(ordered), attr_names)
  } else {
    stop("AI-GENIE expects target.N to be NULL, a single integer, or a named list/vector of integers.", call. = FALSE)
  }

  # --- Reduction sanity check (target.N / n_levels >= 15) ---
  low_ratio <- c()
  for (name in attr_names) {
    n <- target.N[[name]]
    k <- length(items.attributes[[name]])
    if ((n / k) < 15) {
      low_ratio <- c(low_ratio, name)
    }
  }

  if (length(low_ratio) > 0) {

    # Return right away if only embeddings or items are desired (no reduction)
    if (embeddings.only || items.only){
      return(target.N)
    }

    if(!silently){
      warning(
        paste0(
          "AI-GENIE recommends at least 15 examples per attribute value ",
          "for meaningful dimension reduction. Consider increasing target.N for: ",
          paste0("`", low_ratio, "`", collapse = ", ")
        )
      )
    }

  }

  return(target.N)
}


# Validate LLM parameters ----
#' Validate `temperature` for Text Generation
#'
#' Ensures `temperature` is a numeric value between 0 and 2
#'
#' @param temperature A numeric value
temperature_validate <- function(temperature) {
  if (!is.numeric(temperature) || length(temperature) != 1 || is.na(temperature)) {
    stop("AI-GENIE expects temperature to be a numeric value or NULL.", call. = FALSE)
  }

  if (temperature < 0 || temperature > 2) {
    stop("AI-GENIE expects temperature to be between 0 and 2.", call. = FALSE)
  }
}

#' Validate `top.p` for Text Generation
#'
#' Ensures `top.p` is a numeric value between 0 and 1, or NULL.
#'
#' @param top.p A numeric value
#'
top.p_validate <- function(top.p) {

  if (!is.numeric(top.p) || length(top.p) != 1 || is.na(top.p)) {
    stop("AI-GENIE expects top.p to be a numeric value or NULL.", call. = FALSE)
  }

  if (top.p < 0 || top.p > 1) {
    stop("AI-GENIE expects top.p to be between 0 and 1.", call. = FALSE)
  }

}

# Validate additional prompt components ----
#' Validate and Clean `response.options`
#'
#' Validates that `response.options` is an atomic vector of non-empty strings,
#' with no missing or invalid values. Whitespace is trimmed from each string.
#'
#' @param response.options An atomic character vector of response labels.
#'
response.options_validate <- function(response.options) {
  # --- If not specified, ignore ---
  if(!is.null(response.options)){


    # --- Type check ---
    if (!is.atomic(response.options)) {
      stop("AI-GENIE expects response.options to be an atomic vector of strings.", call. = FALSE)
    }

    # --- Empty check ---
    if (length(response.options) == 0) {
      stop("AI-GENIE expects response.options to contain at least one string.", call. = FALSE)
    }

    # --- NA or non-character check ---
    if (!is.character(response.options)) {
      stop("AI-GENIE expects all values in response.options to be strings.", call. = FALSE)
    }

    if (any(is.na(response.options))) {
      stop("AI-GENIE expects response.options to contain no missing (NA) values.", call. = FALSE)
    }

    # --- Trim + Empty string check ---
    cleaned <- trimws(response.options)

    if (any(cleaned == "")) {
      stop("AI-GENIE expects response.options to contain no empty strings (after trimming).", call. = FALSE)
    }
  }
}

#' Validate and Normalize `prompt.notes`
#'
#' Accepts a string, NULL, or a named list of strings/NULLs. Ensures one entry
#' per attribute in `items.attributes`, returning a fully named and cleaned list.
#'
#' @param prompt.notes A single string, NULL, or named list of strings/NULLs.
#' @param items.attributes A cleaned list from `validate_items.attributes()`.
#'
#' @return A named list of strings, one per attribute, with NULLs replaced by "".
validate_prompt.notes <- function(prompt.notes, items.attributes) {
  attr_names <- names(items.attributes)
  attr_names_norm <- trimws(tolower(attr_names))
  n_attr <- length(attr_names)
  norm_str <- function(x) trimws(tolower(x))

  # --- Case: NULL → return empty string for each attr ---
  if (is.null(prompt.notes)) {
    return(setNames(as.list(rep("", n_attr)), attr_names))
  }

  # --- Case: single string → repeat for each attr ---
  if (is.character(prompt.notes) && length(prompt.notes) == 1 && !is.na(prompt.notes)) {
    return(setNames(as.list(rep(prompt.notes, n_attr)), attr_names))
  }

  # --- Case: named list of strings or NULLs ---
  if (is.list(prompt.notes)) {
    note_names <- names(prompt.notes)

    if (is.null(note_names) || any(note_names == "") || any(is.na(note_names))) {
      stop("AI-GENIE expects prompt.notes to be a named list if not a string or NULL.", call. = FALSE)
    }

    note_names_norm <- norm_str(note_names)

    # Check for full alignment with items.attributes
    if (!setequal(note_names_norm, attr_names_norm)) {
      missing <- setdiff(attr_names_norm, note_names_norm)
      extra   <- setdiff(note_names_norm, attr_names_norm)

      msg <- "AI-GENIE expects prompt.notes to include exactly one entry per item type."
      if (length(missing) > 0) {
        msg <- paste0(msg, "\nMissing types: ", paste(sprintf("`%s`", missing), collapse = ", "))
      }
      if (length(extra) > 0) {
        msg <- paste0(msg, "\nUnexpected types: ", paste(sprintf("`%s`", extra), collapse = ", "))
      }

      stop(msg, call. = FALSE)
    }

    # Normalize values and re-order
    out <- list()
    for (attr in attr_names) {
      key <- which(note_names_norm == norm_str(attr))
      val <- prompt.notes[[key]]

      if (!is.null(val)) {
        if (!is.character(val) || length(val) != 1 || is.na(val)) {
          stop(
            paste0(
              "AI-GENIE expects prompt.notes$", note_names[[key]],
              " to be a string or NULL."
            ),
            call. = FALSE
          )
        }
      }

      out[[attr]] <- if (is.null(val)) "" else val
    }

    return(out)
  }

  # --- Invalid type ---
  stop("AI-GENIE expects prompt.notes to be NULL, a string, or a named list of strings/NULLs.", call. = FALSE)
}

# Validate 'Custom Mode' Settings ----
#' Validate and Normalize `main.prompts`
#'
#' Validates that `main.prompts` is a named list of non-empty strings,
#' one for each attribute in `items.attributes`, matched by normalized name.
#'
#' @param main.prompts A named list of prompt strings, one per attribute.
#' @param items.attributes A cleaned list from `validate_items.attributes()`.
#' @param silently A flag determining wheter a warning message should be printed
#'
#' @return A cleaned and ordered named list of trimmed prompt strings.
#'    Also returns the appropriate 'custom' flag (TRUE if custom ok, FALSE if not)
main.prompts_validate <- function(main.prompts, items.attributes, silently) {
  attr_names <- names(items.attributes)
  attr_names_norm <- trimws(tolower(attr_names))
  norm_str <- function(x) trimws(tolower(x))

  custom <- TRUE # assumes that the prompts are adequate for custom use

  # --- Type and name check ---
  if (!is.list(main.prompts) || is.null(names(main.prompts))) {
    stop("AI-GENIE expects `main.prompts` to be a named list.", call. = FALSE)
  }

  prompt_names_raw <- names(main.prompts)
  prompt_names_norm <- norm_str(prompt_names_raw)

  # --- Name alignment check ---
  if (!setequal(prompt_names_norm, attr_names_norm)) {
    missing <- setdiff(attr_names_norm, prompt_names_norm)
    extra   <- setdiff(prompt_names_norm, attr_names_norm)

    msg <- "AI-GENIE expects `main.prompts` to include exactly one non-empty prompt per `items.attributes` key."
    if (length(missing) > 0) {
      msg <- paste0(msg, "\nMissing prompts for attributes: ", paste(sprintf("`%s`", missing), collapse = ", "))
    }
    if (length(extra) > 0) {
      msg <- paste0(msg, "\nUnexpected prompts found for: ", paste(sprintf("`%s`", extra), collapse = ", "))
    }

    stop(msg, call. = FALSE)
  }

  # --- Value validation and normalization ---
  out <- list()
  for (attr in attr_names) {
    key <- which(prompt_names_norm == norm_str(attr))
    val <- main.prompts[[key]]
    raw_key <- prompt_names_raw[key]

    if (!is.character(val) || length(val) != 1 || is.na(val)) {
      stop(
        paste0("AI-GENIE expects `main.prompts$", raw_key, "` to be a non-empty string."),
        call. = FALSE
      )
    }

    val_trimmed <- trimws(val)

    if (val_trimmed == "") {
      stop(
        paste0("AI-GENIE expects `main.prompts$", raw_key, "` to be a non-empty string (after trimming)."),
        call. = FALSE
      )
    }

    # --- Attribute presence check within the prompt text ---
    declared_attributes <- items.attributes[[attr]]
    declared_attributes_norm <- norm_str(declared_attributes)
    prompt_norm <- norm_str(val_trimmed)

    missing_attrs <- declared_attributes[!vapply(declared_attributes_norm, function(a) grepl(a, prompt_norm, fixed = TRUE), logical(1))]

    if (length(missing_attrs) > 0) {
      if(!silently){
        message(
          paste0(
            "NOTE: The custom prompt validation failed.\nThe prompt for `", attr, "` must explicitly mention ALL associated attributes.\n",
            "The following attribute(s) are missing from the prompt:\n",
            paste0("- ", missing_attrs, collapse = "\n"),
            "\n\nAIGENIE will proceed by generating prompts for you. If you'd like to use your custom prompt, \nplease revise your prompt(s) so that it explicitly instructs the LLM to generate \nitems based on these attributes."
          )
        )
      }
      custom <- FALSE # update flag
    }

    out[[attr]] <- val_trimmed
  }

  return(list(out = out,
              custom = custom))
}






