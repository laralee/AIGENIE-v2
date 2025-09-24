
#### Building Prompts for AIGENIE #### ----

## Construct system.role ----
#' Create a System Role Prompt for an LLM Item Writer
#'
#' Constructs a system-level prompt to guide an LLM in behaving like an expert scale developer.
#' The prompt communicates role identity, domain expertise, scale context, audience constraints,
#' and response option considerations.
#'
#' @param domain (Optional) A string indicating the scale's conceptual or applied domain
#'   (e.g., "clinical psychology", "behavioral economics").
#' @param scale.title (Optional) A string providing the title of the scale (e.g., "Emotion Regulation Index").
#' @param audience (Optional) A string specifying the target respondent group (e.g., "adolescents", "working adults").
#' @param response.options (Optional) A character vector of response choices that the LLM should consider when phrasing items
#'   (e.g., \code{c("Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree")}).
#' @param system.role (Optional) A custom system prompt provided directly by the user. If supplied, it will be used as-is.
#'
#' @return A single character string representing the full system prompt to be passed to an LLM interface (e.g., OpenAI Chat API).
#' If \code{system.role} is not provided, the function dynamically constructs one based on the other parameters.
#'
create_system.role <- function(domain, scale.title, audience,
                               response.options, system.role){

  if(is.null(system.role)){
    ### Build System role prompt if one was not already specified by the user.
    system.role <- paste0(
      "You are an expert measurement methodologist; more specifically, you are an accomplished,",
      " well-trained, and knowledgeable scale-developer",
      ifelse(!is.null(domain), paste0(" specializing in ", domain, "."),"."),
      " Your task is to create novel, high-quality, and robust items for a new inventory",
      ifelse(!is.null(scale.title), paste0(" called '", scale.title, ".'"), "."),
      ifelse(!is.null(audience), paste0(" Ensure the items are appropriate for an audience of ", audience, "."), "")
      )
  }

  # add the response options here, if provided
  if(!is.null(response.options)){

    n <- length(response.options)
    numbered <- paste0("(", seq_len(n), ") ", response.options)

    if (n == 1) {
      numbered <- numbered[1]
    } else if (n == 2) {
      numbered <- paste(numbered, collapse = " and ")
    } else {
      numbered <- paste(
        paste(numbered[1:(n - 1)], collapse = ", "),
        "and", numbered[n]
      )
    }

    system.role <- paste0(system.role,
                          "\n",
                          "For your reference, the response options for the items ",
                          "you are authoring will be as follows: ", numbered,
                          ". I will add the response options myself AFTER you create the items; do NOT",
                          " include them in the items you write.",
                          " However, you should still",
                          " ensure the items are appropriately phrased given these options.")
  }

  return(system.role)
}

### Create/Modify Main Prompts ----

#' Create Initial Main Prompts for Item Generation
#'
#' Constructs structured prompts for an LLM to generate scale items based on a list of item attributes,
#' optional item type definitions, audience and domain context, and example items. Each resulting prompt
#' includes strict formatting requirements, attribute listings, and item-generation instructions.
#'
#' @param item.attributes A named list where each element is a character vector of attribute names for an item type.
#' @param item.type.definitions (Optional) A named list of textual definitions for each item type, used to provide conceptual clarity in the prompt.
#' @param domain (Optional) A string specifying the domain (e.g., "psychological", "clinical") the items belong to.
#' @param scale.title (Optional) The title of the scale (e.g., "Emotion Regulation Inventory").
#' @param prompt.notes (Optional) A named list of additional instructions or warnings to include per item type.
#' @param audience (Optional) A string describing the target audience or population (e.g., "adolescents", "working adults").
#' @param item.examples (Optional) A data frame of existing high-quality example items. Used to guide item phrasing and structure. Must be compatible with the helper `construct_item.examples_string()`.
#'
#' @return A named list of character strings. Each entry corresponds to one item type and contains a complete prompt
#' to guide an LLM in generating two distinct items per attribute, formatted as a JSON array.
#'
create_main.prompts <- function(item.attributes, item.type.definitions,
                                domain, scale.title, prompt.notes,
                                audience, item.examples){
  item_types <- names(item.attributes)


  main.prompts <- list()

  # Create user prompts

  for (i in seq_along(item_types)) {
    current_type <- item_types[[i]]
    attributes <- item.attributes[[current_type]]

    # Build attributes string
    n <- length(attributes)
    numbered <- paste0("(", seq_len(n), ") ", attributes)

    if (n == 1) {
      numbered <- numbered[1]
    } else if (n == 2) {
      numbered <- paste(numbered, collapse = " and ")
    } else {
      numbered <- paste(
        paste(numbered[1:(n - 1)], collapse = ", "),
        "and", numbered[n]
      )
    }


    # Retrieve definition if provided
    definition <- ""
    if (!is.null(item.type.definitions) && !is.null(item.type.definitions[[current_type]])) {
      definition <- item.type.definitions[[current_type]]
      definition <- paste0("The precise definition of '", current_type, "' in this context is as follows: ", definition, "\n")
    }

    if(is.data.frame(item.examples)){
      examples_str <- construct_item.examples_string(item.examples, current_type)
    } else {
      examples_str <- NULL
    }

    # Construct the main prompt for all prompts
    main.prompts[[current_type]] <- paste0(
      "Generate a grand total of ", length(attributes) * 2, " novel, UNIQUE, reliable, and valid ",
      ifelse(!is.null(domain), paste0(domain, " "), ""),
      "items",ifelse(is.null(scale.title), " for a scale. ", paste0(" for a scale called '", scale.title, ".' ")),
      ifelse(is.null(audience), "", paste0("This inventory will be administered to an audience of ", audience, ". ")),
      "Write items related to the attributes of the item type '", current_type, ".' ", definition,
      "Here are the attributes of the item type '", current_type, "': ", numbered,
      ". Generate EXACTLY TWO items PER attribute. Use the ",length(attributes)," attributes EXACTLY as provided; do NOT add your own or leave any out." ,
      "\nEACH item should be ROBUST, NOVEL, and UNIQUE. These items must be top-quality.\n",
      "Return output STRICTLY as a JSON array of objects, each with keys `attribute` and `statement`, e.g.:\n",
      "[{\"attribute\":\"", item.attributes[[current_type]][1], "\",\"statement\":\"Your item here.\"}, …]\n",
      "This JSON formatting is EXTREMELY important. ONLY output the items in this formatting; DO NOT include any other text in your response.",
      ifelse(is.null(examples_str), "", paste0("\n\nTo better help you understand how EXACTLY to phrase/constuct your items,",
               " here are some EXAMPLE high-quality items that already exist on the scale that you MUST",
            " emulate in terms of QUALITY and item STRUCTURE (how the item is framed/setup/constructed).",
            " Remember that we want each item to be NOVEL and DISTINCT on this scale to best capture as much variance as possible,",
            " so do NOT recycle any of these examples' core content. Here are the examples:\n", examples_str
        )),
      ifelse(prompt.notes[[current_type]] == "", "", paste0(
        "\n\nFinally, I have an EXTREMELY critical note that you MUST keep in mind:\n",
        prompt.notes[[current_type]])

      )
    )
  }

  return(main.prompts)
}


#' Construct Formatted String of Example Items
#'
#' Given a validated item examples data frame, this function constructs
#' a string of well-formatted example items grouped by `attribute`, for a given `type`.
#'
#' @param item.examples A validated data frame of item examples.
#' @param current_type A character scalar indicating the type of items to include in the string.
#'
#' @return A character string with grouped and formatted item examples.
#' Construct JSON String of Item Examples (Filtered by Type)
#'
#' Converts a filtered set of item.examples into a JSON array of
#' attribute/statement objects, ready for prompt usage or LLM API calls.
#'
#' @param item.examples A validated data frame with `type`, `attribute`, `statement`.
#' @param current_type A string specifying which type to filter for.
#'
#' @return A single JSON-formatted string (or NULL if no matches).
construct_item.examples_string <- function(item.examples, current_type) {
  # Filter by exact match (already normalized during validation)
  filtered <- item.examples[item.examples$type == current_type, , drop = FALSE]

  if (nrow(filtered) == 0) return(NULL)

  # Prepare clean subset
  df <- data.frame(
    attribute = as.character(filtered$attribute),
    statement = as.character(filtered$statement),
    stringsAsFactors = FALSE
  )

  # Convert to JSON string
  json_string <- jsonlite::toJSON(df, auto_unbox = TRUE)

  return(as.character(json_string))
}


#' Modify Main Prompts with Contextual Enhancements
#'
#' This function appends structured context and formatting instructions to a list of main prompts
#' used for item generation. It ensures that each prompt includes relevant domain information,
#' audience guidance, scale definitions, JSON formatting rules, and optionally example items or
#' critical author notes — but only if these elements are not already present in the prompt
#' (checked case-insensitively and with whitespace trimmed).
#'
#' @param main.prompts A named list of character strings, where each element is a prompt associated with an item type.
#' @param item.attributes A named list where each element is a character vector of attribute names for an item type.
#' @param item.type.definitions (Optional) A named list of definitions corresponding to each item type. Used to append conceptual clarity.
#' @param domain (Optional) A string describing the content domain (e.g., "psychological", "clinical"). Included in the prompt if not already present.
#' @param scale.title (Optional) The name of the scale (e.g., "Social Anxiety Scale") for which items are being generated.
#' @param prompt.notes (Optional) A named list of author-supplied notes for each item type that should be emphasized in the prompt.
#' @param audience (Optional) A string describing the target population (e.g., "adults", "high school students").
#' @param item.examples (Optional) A data frame of example items. Must contain a column matching each item type to extract examples.
#'
#' @return A modified list of character strings, with each prompt updated to include relevant metadata, instructions, and formatting examples as needed.
#'
modify_main.prompts <- function(main.prompts, item.attributes,
                                item.type.definitions,
                                domain, scale.title, prompt.notes,
                                audience, item.examples) {

  item_types <- names(item.attributes)

  # Helper to do case-insensitive, trimmed presence check
  already_present <- function(haystack, needle) {
    haystack_clean <- tolower(trimws(haystack))
    needle_clean <- tolower(trimws(needle))
    grepl(needle_clean, haystack_clean, fixed = TRUE)
  }

  for (i in seq_along(item_types)) {
    current_type <- item_types[[i]]
    prompt <- main.prompts[[current_type]]

    # Start building the additional statement
    additions <- character()

    # DOMAIN
    if (!is.null(domain)) {
      domain_text <- paste0(" Keep in mind you are generating ", domain, " items.")
      if (!already_present(prompt, domain_text)) {
        additions <- c(additions, domain_text)
      }
    }

    # SCALE TITLE
    if (!is.null(scale.title)) {
      scale_text <- paste0(" These items are for a scale called '", scale.title, ".'")
      if (!already_present(prompt, scale_text)) {
        additions <- c(additions, scale_text)
      }
    }

    # AUDIENCE
    if (!is.null(audience)) {
      audience_text <- paste0(" This inventory will be administered to an audience of ", audience, ".")
      if (!already_present(prompt, audience_text)) {
        additions <- c(additions, audience_text)
      }
    }

    # DEFINITION
    definition <- ""
    if (!is.null(item.type.definitions) && !is.null(item.type.definitions[[current_type]])) {
      def_text <- paste0(" The precise definition of '", current_type,
                         "' in this context is as follows: ", item.type.definitions[[current_type]], "\n")
      if (!already_present(prompt, def_text)) {
        definition <- def_text
      }
    }

    # EXAMPLES
    examples_str <- NULL
    if (is.data.frame(item.examples)) {
      examples_str <- construct_item.examples_string(item.examples, current_type)
    }

    examples_section <- ""
    if (!is.null(examples_str)) {
      examples_intro <- "To better help you understand how EXACTLY to phrase/constuct your items"
      if (!already_present(prompt, examples_intro)) {
        examples_section <- paste0(
          "\n\n", examples_intro,
          ", here are some EXAMPLE high-quality items that already exist on the scale that you MUST",
          " emulate in terms of QUALITY and item STRUCTURE (how the item is framed/setup/constructed).",
          " Remember that we want each item to be NOVEL and DISTINCT on this scale to best capture as much variance as possible,",
          " so do NOT recycle any of these examples' core content. Here are the examples:\n", examples_str
        )
      }
    }

    # PROMPT NOTES
    notes_section <- ""
    if (!is.null(prompt.notes[[current_type]]) && nzchar(prompt.notes[[current_type]])) {
      notes_intro <- "Finally, I have an EXTREMELY critical note that you MUST keep in mind:"
      if (!already_present(prompt, notes_intro)) {
        notes_section <- paste0("\n\n", notes_intro, "\n", prompt.notes[[current_type]])
      }
    }

    # JSON format instruction (always added unless already present)
    json_format_str <- paste0(
      "\n\nReturn output STRICTLY as a JSON array of objects, each with keys `attribute` and `statement`, e.g.:\n",
      "[{\"attribute\":\"", item.attributes[[current_type]][1], "\",\"statement\":\"Your item here.\"}, …]\n",
      "This JSON formatting is EXTREMELY important. ONLY output the items in this formatting; DO NOT include any other text in your response."
    )
    if (already_present(prompt, "`attribute`") || already_present(prompt, "Return output STRICTLY")) {
      json_format_str <- ""
    }

    # Final additions block
    additional_statement <- paste0(
      if (length(additions)) paste0(paste(additions, collapse = " "), "\n") else "",
      definition,
      json_format_str,
      examples_section,
      notes_section
    )

    # Append to prompt
    main.prompts[[current_type]] <- paste0(prompt, additional_statement)
  }

  return(main.prompts)
}








#### Generating Items #### ----

generate_items_via_llm <- function(main.prompts, system.role, model, top.p, temperature,
                                   adaptive, silently, groq.API, openai.API, target.N) {

  # Ensure Python environment is ready
  ensure_aigenie_python()

  # Initialize results dataframe
  all_items_df <- data.frame(type = character(),
                             attribute = character(),
                             statement = character(),
                             stringsAsFactors = FALSE)

  # Determine which API to use based on model string
  use_groq <- FALSE

  # Check for model type - special case for "oss" overrides "gpt"
  if (grepl("oss", model, ignore.case = TRUE)) {
    use_groq <- TRUE
  } else if (grepl("gpt|o1|o3|o4", model, ignore.case = TRUE)) {
    use_groq <- FALSE
  } else {
    use_groq <- TRUE
  }

  # Check for missing Groq API key if needed
  if (use_groq && is.null(groq.API)) {
    if (!silently) {
      cat("Groq API key missing. Switching to GPT-4o for generation.\n")
    }
    use_groq <- FALSE
    model <- "gpt-4o"
  }

  # Set up API client and generate function
  tryCatch({
    if (use_groq) {
      groq <- reticulate::import("groq")
      groq_client <- groq$Groq(api_key = groq.API)
      generate_FUN <- groq_client$chat$completions$create
    } else {
      openai <- reticulate::import("openai")
      openai$api_key <- openai.API
      generate_FUN <- openai$ChatCompletion$create
    }
  }, error = function(e) {
    stop(paste("Failed to initialize API client:", conditionMessage(e)))
  })

  # Helper function to make API call
  call_generate_FUN <- function(messages_list) {
    call_params <- list(
      model = model,
      messages = messages_list,
      temperature = temperature,
      top_p = top.p
    )
    do.call(generate_FUN, call_params)
  }

  # Iterate through each item type
  for (item_type in names(main.prompts)) {

    if (!silently) {
      cat(paste("Generating items for", item_type, "...\n"))
    }

    # Track items for this type
    type_items_df <- data.frame(type = character(),
                                attribute = character(),
                                statement = character(),
                                stringsAsFactors = FALSE)

    # Track iterations without new items
    iterations_without_new <- 0
    total_iterations <- 0
    rate_limit_truncate <- FALSE
    max_previous_items <- Inf

    # Continue until target.N reached or stalled
    while (nrow(type_items_df) < target.N[[item_type]]) {

      total_iterations <- total_iterations + 1

      # Construct prompt with adaptive mode if needed
      current_prompt <- main.prompts[[item_type]]

      if (adaptive && nrow(all_items_df) > 0) {
        # Get previous items to append
        previous_items_to_use <- all_items_df

        # If rate limiting detected, truncate the number of previous items
        if (rate_limit_truncate && nrow(previous_items_to_use) > max_previous_items) {
          previous_items_to_use <- tail(previous_items_to_use, max_previous_items)
        }

        examples_string <- construct_item.examples_string(previous_items_to_use, item_type)
        current_prompt <- paste0(current_prompt,
                                 "\n\nDo NOT repeat, rephrase, or reuse the content of ANY items from this list of items you've already generated:\n",
                                 examples_string)
      }

      # Prepare messages list
      messages_list <- list(
        list("role" = "system", "content" = system.role),
        list("role" = "user", "content" = current_prompt)
      )

      # Try API call with retry logic
      api_success <- FALSE
      api_attempts <- 0

      while (!api_success && api_attempts < 5) {
        api_attempts <- api_attempts + 1

        try_resp <- tryCatch({
          response <- call_generate_FUN(messages_list)
          api_success <- TRUE
          response
        }, error = function(e) {
          # Check for rate limiting error
          if (grepl("token limit|context length|context window|token window",
                    conditionMessage(e), ignore.case = TRUE)) {
            if (adaptive && !rate_limit_truncate) {
              if (!silently) {
                cat("\nWarning: Rate limiting detected. Reducing number of previous items in prompt.\n")
              }
              rate_limit_truncate <- TRUE
              max_previous_items <- floor(nrow(all_items_df) * 0.5)  # Reduce by half
            }
          }
          e
        })

        if (!api_success) {
          Sys.sleep(2)  # Brief pause before retry
        }
      }

      # Check if API calls failed
      if (!api_success) {
        error_msg <- paste("API call failed after 5 attempts for", item_type,
                           "Error:", conditionMessage(try_resp))
        cat(paste0("\n", error_msg, "\n"))

        if (nrow(all_items_df) > 0) {
          warning("Returning partial results generated before error.")
          return(all_items_df)
        } else {
          stop(error_msg)
        }
      }

      # Extract raw text from response
      raw_text <- if (use_groq) {
        response$choices[[1]]$message$content
      } else {
        response$choices[[1]]$message$content
      }

      # Clean the output
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



cleaning_function <- function(raw_text, item_type) {

  # Return empty dataframe for null/empty responses
  if (is.null(raw_text) || nchar(trimws(raw_text)) == 0) {
    return(data.frame(type = character(),
                      attribute = character(),
                      statement = character(),
                      stringsAsFactors = FALSE))
  }

  # Initialize results dataframe
  result_df <- data.frame(type = character(),
                          attribute = character(),
                          statement = character(),
                          stringsAsFactors = FALSE)

  # Find all JSON arrays in the text
  # Pattern to match JSON arrays (accounting for nested objects)
  json_pattern <- "\\[\\s*\\{[^\\[\\]]*\\}\\s*(,\\s*\\{[^\\[\\]]*\\}\\s*)*\\]"

  # Extract all potential JSON arrays
  json_matches <- regmatches(raw_text, gregexpr(json_pattern, raw_text, perl = TRUE))[[1]]

  # If no JSON arrays found, try to find JSON that might be malformed
  if (length(json_matches) == 0) {
    # Try a more lenient search - anything between [ and ]
    lenient_pattern <- "\\[.*?\\]"
    json_matches <- regmatches(raw_text, gregexpr(lenient_pattern, raw_text, perl = TRUE))[[1]]
  }

  # Process each potential JSON array
  for (json_str in json_matches) {

    # Clean up common JSON formatting issues
    cleaned_json <- json_str

    # Remove trailing commas before closing brackets/braces
    cleaned_json <- gsub(",\\s*\\}", "}", cleaned_json, perl = TRUE)
    cleaned_json <- gsub(",\\s*\\]", "]", cleaned_json, perl = TRUE)

    # Remove any non-printable characters
    cleaned_json <- gsub("[^[:print:]]", "", cleaned_json)

    # Fix single quotes to double quotes (common LLM error)
    # But be careful not to replace single quotes within the actual text
    # This is a simple approach - may need refinement based on actual outputs
    cleaned_json <- gsub("(\\{|,|:)\\s*'([^']*?)'\\s*(:|,|\\})", '\\1"\\2"\\3', cleaned_json, perl = TRUE)

    # Try to parse the JSON
    parsed_json <- NULL
    try({
      parsed_json <- jsonlite::fromJSON(cleaned_json, simplifyDataFrame = TRUE)
    }, silent = TRUE)

    # If parsing failed, try one more aggressive cleaning
    if (is.null(parsed_json)) {
      try({
        # Remove any text before the first { and after the last }
        cleaned_json <- sub("^[^\\[]*", "", cleaned_json)
        cleaned_json <- sub("[^\\]]*$", "", cleaned_json)
        parsed_json <- jsonlite::fromJSON(cleaned_json, simplifyDataFrame = TRUE)
      }, silent = TRUE)
    }

    # If we successfully parsed JSON, extract the data
    if (!is.null(parsed_json)) {

      # Handle different JSON structures
      if (is.data.frame(parsed_json)) {
        # It's already a dataframe
        temp_df <- parsed_json
      } else if (is.list(parsed_json)) {
        # Convert list to dataframe
        try({
          temp_df <- do.call(rbind.data.frame, lapply(parsed_json, as.data.frame, stringsAsFactors = FALSE))
        }, silent = TRUE)
      } else {
        next  # Skip if we can't process this structure
      }

      # Check if required columns exist
      if (exists("temp_df") && all(c("attribute", "statement") %in% names(temp_df))) {
        # Add the type column
        temp_df$type <- item_type

        # Select only the columns we need in the right order
        temp_df <- temp_df[, c("type", "attribute", "statement")]

        # Ensure all columns are character type
        temp_df$type <- as.character(temp_df$type)
        temp_df$attribute <- as.character(temp_df$attribute)
        temp_df$statement <- as.character(temp_df$statement)

        # Remove any rows with NA or empty statements
        temp_df <- temp_df[!is.na(temp_df$statement) & nchar(trimws(temp_df$statement)) > 0, ]

        # Append to results
        if (nrow(temp_df) > 0) {
          result_df <- rbind(result_df, temp_df)
        }
      }
    }
  }

  # Remove any duplicate statements within this batch
  if (nrow(result_df) > 0) {
    result_df <- result_df[!duplicated(result_df$statement), ]
  }

  return(result_df)
}

#### Embedding Items ----
#' Embed Items Using OpenAI's Embedding API
#'
#' This function takes a data frame of items and creates embeddings using OpenAI's API.
#' It returns a matrix where each column represents an item and each row represents
#' an embedding dimension.
#'
#' @param embedding.model A string indicating which OpenAI embedding model to use
#' @param openai.API A string containing the user's OpenAI API key
#' @param items A data frame with 'ID' and 'statement' columns containing items to embed
#' @param silently A flag that describes whether to issue progress statements
#'
#' @return A list containing:
#'   \item{embeddings}{A matrix where columns are items (named by ID) and rows are embedding dimensions}
#'   \item{success}{A logical indicating whether the embedding process was successful}
#'
embed_items <- function(embedding.model, openai.API, items, silently) {

  # Ensure Python environment is ready
  ensure_aigenie_python()

  if(!silently){
    cat("\n")
    cat("Generating embeddings...")

  }

  # Initialize return structure
  result <- list(
    embeddings = NULL,
    success = FALSE
  )

  # Initialize OpenAI API
  tryCatch({
    openai <- reticulate::import("openai")
    openai$api_key <- openai.API

    # Extract statements to embed
    statements <- items$statement
    item_ids <- items$ID

    # Initialize list to store embeddings
    all_embeddings <- list()

    # Process each item statement
    for (i in seq_along(statements)) {

      # Create embedding for current statement
      response <- openai$Embedding$create(
        input = statements[i],
        model = embedding.model
      )

      # Extract embedding vector
      embedding_vector <- unlist(response$data[[1]]$embedding)
      all_embeddings[[i]] <- embedding_vector
    }

    # Combine all embeddings into a matrix
    # Each column is an item, each row is an embedding dimension
    embedding_matrix <- do.call(cbind, all_embeddings)

    # Set column names to item IDs
    colnames(embedding_matrix) <- item_ids

    # Update result
    result$embeddings <- embedding_matrix
    result$success <- TRUE

    if(!silently){
      cat(" Done.\n\n")

    }

  }, error = function(e) {
    # Handle API errors gracefully
    cat("Error occurred during embedding process:\n")
    cat("Error message:", conditionMessage(e), "\n")
    cat("Returning partial results with success = FALSE\n")

    result$success <- FALSE
    # result$embeddings remains NULL
  })

  return(result)
}
