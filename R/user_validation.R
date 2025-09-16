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
#' @param item.attributes A named list of attributes and item types. Must be validated via
#'   \code{item.attributes_validate()}.
#' @param openai.API A string. OpenAI API key.
#' @param main.prompts A named list of custom prompts that the user specifies (if desired)
#' @param groq.API A string or NULL. Groq API key.
#' @param model A string. The user-specified language model. Will be resolved to a
#'   canonical model name using \code{resolve_model_name()}.
#' @param temperature A numeric value between 0 and 2.
#' @param top.p A numeric value between 0 and 1.
#' @param embedding.model A string or NULL. Must be one of the accepted OpenAI embedding models.
#' @param target.N Either a scalar integer, NULL, or a named list/vector of integers
#'   corresponding to each attribute. Used for synthetic item generation.
#' @param domain A string describing the domain of the assessment.
#' @param scale.title A string naming the scale.
#' @param item.examples A data frame containing `type`, `attribute`, and `statement` columns.
#'   All values must be strings. Optional.
#' @param audience A string or NULL. The intended audience of the assessment.
#' @param item.type.definitions A named list mapping item types to their descriptions.
#'   Optional.
#' @param response.options An atomic vector of strings listing the response options users will have.
#'   Optional.
#' @param prompt.notes A named list or string that gives the LLM additional instructions to be appended to the prompt.
#'   Optional.
#' @param system.role A string or NULL. Used to customize the system prompt.
#' @param EGA.model A string or NULL. One of `"BGGM"`, `"glasso"`, or `"TMFG"`.
#' @param EGA.algorithm A string. One of `"leiden"`, `"louvain"`, or `"walktrap"`.
#' @param EGA.uni.method A string. One of `"expand"`, `"LE"`, or `"louvain"`.
#' @param keep.org A boolean. If TRUE, preserve original inputs in the output.
#' @param items.only A boolean. Whether to generate only items.
#' @param embeddings.only A boolean. Whether to run in embedding-only mode.
#' @param adaptive A boolean. Whether adaptive design logic should be applied.
#' @param plot A boolean. Whether to display plots for visual diagnostics.
#' @param silently A boolean. If TRUE, suppresses warning messages.
#'
#' @return A named list containing:
#' \describe{
#'   \item{target.N}{A named list of integers, aligned with `item.attributes`}
#'   \item{EGA.model}{Canonical model string or NULL}
#'   \item{EGA.uni.method}{Canonical unidimensionality method}
#'   \item{EGA.algorithm}{Canonical community detection algorithm}
#'   \item{model}{Resolved model string for text generation}
#'   \item{item.type.definitions}{Cleaned item type definitions (if provided)}
#'   \item{item.examples}{Cleaned item examples (if provided)}
#'   \item{item.attributes}{Cleaned and normalized item attributes}
#'   \item{prompt.notes}{Cleaned and normalized prompt notes (if provided)}
#'   \item{main.prompts}{Cleaned and normalized main prompts (if provided)}
#'   \item{custom}{A flag signaling whether we are in custom mode or not}
#' }
#'
validate_user_input_AIGENIE <- function(item.attributes, openai.API, main.prompts,
                                        groq.API, model, temperature,
                                        top.p, embedding.model, target.N,
                                        domain, scale.title, item.examples,
                                        audience, item.type.definitions, response.options,
                                        prompt.notes,
                                        system.role, EGA.model, EGA.algorithm,
                                        EGA.uni.method, keep.org, items.only,
                                        embeddings.only, adaptive, plot, silently) {

  # Ensure all "TRUEs and FALSEs" are specified accordingly
  validate_booleans(items.only, adaptive, plot, keep.org, silently, embeddings.only)

  # Ensure all string objects are actually strings (or set to NULL)
  validate_strings(openai.API, groq.API, audience, scale.title,
                   system.role, domain, model, EGA.model, EGA.algorithm,
                   embedding.model, EGA.uni.method)

  # Validate the `item.attributes` object
  item.attributes <- items.attributes_validate(item.attributes)

  # Validate the `item.examples` object based on `item.attributes`
  if(!is.null(item.examples)){ # only run the check if user provided
    item.examples <- item.examples_validate(item.examples, item.attributes)
  }

  # Validate the `item.type.definitions` object based on `item.attributes`
  if(!is.null(item.type.definitions)){ # only run the check if user provided
    item.type.definitions <- item.type.definitions_validate(item.type.definitions, item.attributes)
  }

  # Validate the `model` string and replace it with a valid model string if necessary
  model <- resolve_model_name(model, silently)

  # Validate the embedding model
  embedding.model_validate(embedding.model)

  # Validate the parameters to be passed to EGA
  EGA_params <- validate_ega_params(EGA.algorithm, EGA.uni.method, EGA.model)
  EGA.algorithm <- EGA_params$EGA.algorithm
  EGA.uni.method <- EGA_params$EGA.uni.method
  EGA.model <- EGA_params$EGA.model


  # Validate target N
  target.N <- target.N_validate(target.N, item.attributes, items.only, embeddings.only, silently)


  # Validate LLM parameters
  top.p_validate(top.p)
  temperature_validate(temperature)

  # Validate prompt components
  response.options_validate(response.options)
  prompt.notes <- validate_prompt.notes(prompt.notes, item.attributes)

  # Check to see if the user is in custom mode
  if(!is.null(main.prompts)){
    main.prompts_validation <- main.prompts_validate(main.prompts, item.attributes, silently)
    custom <- main.prompts_validation$custom
    main.prompts <- main.prompts_validation$out

  } else {
    custom <- FALSE
  }

  # Return
  return(list(
    target.N = target.N,
    EGA.model = EGA.model,
    EGA.uni.method = EGA.uni.method,
    EGA.algorithm = EGA.algorithm,
    model = model,
    item.type.definitions = item.type.definitions,
    item.examples = item.examples,
    item.attributes = item.attributes,
    prompt.notes = prompt.notes,
    main.prompts = main.prompts,
    custom = custom
  ))

}

