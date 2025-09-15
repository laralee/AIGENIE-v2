#' @export
AIGENIE_v2 <- function(item_attributes, openai_API, # required parameters

                       # optional parameters --

                       # if using AIGENIE in custom mode, this should be set:
                       main_prompts = NULL,

                       # LLM parameters
                       groq_API = NULL, model = "gpt4o", temperature = 1,
                       top_p = 1, embedding_model = "text-embedding-3-small",
                       target_N = NULL,

                       # Prompt parameters
                       domain = NULL, scale_title = NULL, item_examples = NULL,
                       audience = NULL, item_type_definitions = NULL,
                       response_options = NULL, prompt_notes = NULL, system_role = NULL,

                       # EGA parameters
                       EGA_model = NULL, EGA_algorithm = "walktrap", EGA_uni_method = "louvain",

                       # Flags
                       keep_org = FALSE, items_only = FALSE, embeddings_only = FALSE,
                       adaptive = TRUE, plot = TRUE, silently = FALSE
                       ){


  # Validate all params and reassign params
  validation <- validate_user_input_AIGENIE(item_attributes, openai_API, main_prompts,
                                            groq_API, model, temperature,
                                            top_p, embedding_model, target_N,
                                            domain, scale_title, item_examples,
                                            audience, item_type_definitions,
                                            response_options, prompt_notes,
                                            system_role, EGA_model, EGA_algorithm,
                                            EGA_uni_method, keep_org, items_only,
                                            embeddings_only, adaptive, plot, silently)


  target_N <- validation$target_N
  EGA_model <- validation$EGA_model
  EGA_uni_method <- validation$EGA_uni_method
  EGA_algorithm <- validation$EGA_algorithm
  model <- validation$model
  item_type_definitions <- validation$item_type_definitions
  item_examples <- validation$item_examples
  item_attributes <- validation$item_attributes
  prompt_notes <- validation$prompt_notes
  main_prompts <- validation$main_prompts
  custom <- validation$custom

  # Begin constructing the prompts
  # first, the system role if one was not provided
  system_role <- create_system_role(domain, scale_title, audience,
                                    response_options, system_role)


  # Create/Modify the prompts
  if(!custom){
    main_prompts <- create_main_prompts(item_attributes, item_type_definitions,
                                      domain, scale_title, prompt_notes,
                                      audience, item_examples)
  } else {
    main_prompts <- modify_main_prompts(main_prompts, item_attributes,
                                        item_type_definitions,
                                        domain, scale_title, prompt_notes,
                                        audience, item_examples)

  }


  # Generate the items for reduction analysis
  items_gen <- generate_items_via_llm(main_prompts, system_role, model, top_p, temperature,
                                  adaptive, silently, groq_API, openai_API, target_N)
  items <- items_gen$items
  success <- items_gen$successful

  if(is.data.frame(items)){
    items$ID <- 1:nrow(items) # create an ID variable
  }

  # return items if requested OR if the run was not a success
  if(items_only || !success){

    if(!success && !silently){
      message("Item generation failed before completion. Returning a data frame of items generated thus far.")
    }

    return(items)
  }


  # Now, generate item embeddings
  attempt_to_embed <- embed_items(embedding_model, openai_API, items, silently)
  success <- attempt_to_embed$success
  embeddings <- attempt_to_embed$embeddings

  # Return partial results if failure or just the embeddings if requested
  if(!success || embeddings_only){
    if(!success && !silently){
      message("Embedding step has failed. Returning a data frame of items generated instead.")
    }

    if(!success){
      return(items)
    }

     return(list(embeddings = embeddings, items = items))
  }

}


