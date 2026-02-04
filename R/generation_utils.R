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
#' @keywords internal
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

### Create/Modify Main Prompts

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
#' @keywords internal
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
      examples_str <- construct_item.examples_string_for_prompt(item.examples, current_type)
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


#' Construct Formatted String of Example Items for Prompts
#'
#' Given a validated item examples data frame, this function constructs
#' a JSON string of example items for a given type, to be used in prompt building.
#'
#' @param item.examples A validated data frame with `type`, `attribute`, `statement`.
#' @param current_type A string specifying which type to filter for.
#'
#' @return A single JSON-formatted string (or NULL if no matches).
#'
#' @keywords internal
construct_item.examples_string_for_prompt <- function(item.examples, current_type) {
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
#' @keywords internal
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
      examples_str <- construct_item.examples_string_for_prompt(item.examples, current_type)
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
