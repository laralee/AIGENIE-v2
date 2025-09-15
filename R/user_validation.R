#### Validate User Input ####

# AIGENIE ----
#' Validate All User Inputs for AI-GENIE
#'
#' This function performs comprehensive validation and normalization of all user-supplied
#' inputs to the AI-GENIE package. It checks logical flags, strings, model names, item
#' attribute structures, and ensures consistency across all interdependent components.
#'
#' If any input is invalid or misaligned with the package’s expected structure,
#' informative errors or warnings are raised. Cleaned and normalized objects are returned
#' for use downstream.
#'
#' @param item_attributes A named list of attributes and item types. Must be validated via
#'   \code{item_attributes_validate()}.
#' @param openai_API A string. OpenAI API key.
#' @param main_prompts A named list of custom prompts that the user specifies (if desired)
#' @param groq_API A string or NULL. Groq API key.
#' @param model A string. The user-specified language model. Will be resolved to a
#'   canonical model name using \code{resolve_model_name()}.
#' @param temperature A numeric value between 0 and 2.
#' @param top_p A numeric value between 0 and 1.
#' @param embedding_model A string or NULL. Must be one of the accepted OpenAI embedding models.
#' @param target_N Either a scalar integer, NULL, or a named list/vector of integers
#'   corresponding to each attribute. Used for synthetic item generation.
#' @param domain A string describing the domain of the assessment.
#' @param scale_title A string naming the scale.
#' @param item_examples A data frame containing `type`, `attribute`, and `statement` columns.
#'   All values must be strings. Optional.
#' @param audience A string or NULL. The intended audience of the assessment.
#' @param item_type_definitions A named list mapping item types to their descriptions.
#'   Optional.
#' @param response_options An atomic vector of strings listing the response options users will have.
#'   Optional.
#' @param prompt_notes A named list or string that gives the LLM additional instructions to be appended to the prompt.
#'   Optional.
#' @param system_role A string or NULL. Used to customize the system prompt.
#' @param EGA_model A string or NULL. One of `"BGGM"`, `"glasso"`, or `"TMFG"`.
#' @param EGA_algorithm A string. One of `"leiden"`, `"louvain"`, or `"walktrap"`.
#' @param EGA_uni_method A string. One of `"expand"`, `"LE"`, or `"louvain"`.
#' @param keep_org A boolean. If TRUE, preserve original inputs in the output.
#' @param items_only A boolean. Whether to generate only items.
#' @param embeddings_only A boolean. Whether to run in embedding-only mode.
#' @param adaptive A boolean. Whether adaptive design logic should be applied.
#' @param plot A boolean. Whether to display plots for visual diagnostics.
#' @param silently A boolean. If TRUE, suppresses warning messages.
#'
#' @return A named list containing:
#' \describe{
#'   \item{target_N}{A named list of integers, aligned with `item_attributes`}
#'   \item{EGA_model}{Canonical model string or NULL}
#'   \item{EGA_uni_method}{Canonical unidimensionality method}
#'   \item{EGA_algorithm}{Canonical community detection algorithm}
#'   \item{model}{Resolved model string for text generation}
#'   \item{item_type_definitions}{Cleaned item type definitions (if provided)}
#'   \item{item_examples}{Cleaned item examples (if provided)}
#'   \item{item_attributes}{Cleaned and normalized item attributes}
#'   \item{prompt_notes}{Cleaned and normalized prompt notes (if provided)}
#'   \item{main_prompts}{Cleaned and normalized main prompts (if provided)}
#'   \item{custom}{A flag signaling whether we are in custom mode or not}
#' }
#'
validate_user_input_AIGENIE <- function(item_attributes, openai_API, main_prompts,
                                        groq_API, model, temperature,
                                        top_p, embedding_model, target_N,
                                        domain, scale_title, item_examples,
                                        audience, item_type_definitions, response_options,
                                        prompt_notes,
                                        system_role, EGA_model, EGA_algorithm,
                                        EGA_uni_method, keep_org, items_only,
                                        embeddings_only, adaptive, plot, silently) {

  # Ensure all "TRUEs and FALSEs" are specified accordingly
  validate_booleans(items_only, adaptive, plot, keep_org, silently, embeddings_only)

  # Ensure all string objects are actually strings (or set to NULL)
  validate_strings(openai_API, groq_API, audience, scale_title,
                   system_role, domain, model, EGA_model, EGA_algorithm,
                   embedding_model, EGA_uni_method)

  # Validate the `item_attributes` object
  item_attributes <- item_attributes_validate(item_attributes)

  # Validate the `item_examples` object based on `item_attributes`
  if(!is.null(item_examples)){ # only run the check if user provided
    item_examples <- item_examples_validate(item_examples, item_attributes)
  }

  # Validate the `item_type_definitions` object based on `item_attributes`
  if(!is.null(item_type_definitions)){ # only run the check if user provided
    item_type_definitions <- item_type_definitions_validate(item_type_definitions, item_attributes)
  }

  # Validate the `model` string and replace it with a valid model string if necessary
  model <- resolve_model_name(model, silently)

  # Validate the embedding model
  embedding_model_validate(embedding_model)

  # Validate the parameters to be passed to EGA
  EGA_params <- validate_ega_params(EGA_algorithm, EGA_uni_method, EGA_model)
  EGA_algorithm <- EGA_params$EGA_algorithm
  EGA_uni_method <- EGA_params$EGA_uni_method
  EGA_model <- EGA_params$EGA_model


  # Validate target N
  target_N <- target_N_validate(target_N, item_attributes, items_only, embeddings_only, silently)


  # Validate LLM parameters
  top_p_validate(top_p)
  temperature_validate(temperature)

  # Validate prompt components
  response_options_validate(response_options)
  prompt_notes <- validate_prompt_notes(prompt_notes, item_attributes)

  # Check to see if the user is in custom mode
  if(!is.null(main_prompts)){
    main_prompts_validation <- main_prompts_validate(main_prompts, item_attributes, silently)
    custom <- main_prompts_validation$custom
    main_prompts <- main_prompts_validation$out

  } else {
    custom <- FALSE
  }

  # Return
  return(list(
    target_N = target_N,
    EGA_model = EGA_model,
    EGA_uni_method = EGA_uni_method,
    EGA_algorithm = EGA_algorithm,
    model = model,
    item_type_definitions = item_type_definitions,
    item_examples = item_examples,
    item_attributes = item_attributes,
    prompt_notes = prompt_notes,
    main_prompts = main_prompts,
    custom = custom
  ))

}
