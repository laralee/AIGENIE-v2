#' @export
AIGENIE_v2 <- function(item.attributes, openai.API, # required parameters

                       # optional parameters --

                       # if using AIGENIE in custom mode, this should be set:
                       main.prompts = NULL,

                       # LLM parameters
                       groq.API = NULL, model = "gpt4o", temperature = 1,
                       top.p = 1, embedding.model = "text-embedding-3-small",
                       target.N = NULL,

                       # Prompt parameters
                       domain = NULL, scale.title = NULL, item.examples = NULL,
                       audience = NULL, item.type.definitions = NULL,
                       response.options = NULL, prompt.notes = NULL, system.role = NULL,

                       # EGA parameters
                       EGA.model = NULL, EGA.algorithm = "walktrap", EGA.uni.method = "louvain",

                       # Flags
                       keep.org = FALSE, items.only = FALSE, embeddings.only = FALSE,
                       adaptive = TRUE, plot = TRUE, silently = FALSE
                       ){


  # Validate all params and reassign params
  validation <- validate_user_input_AIGENIE(item.attributes, openai.API, main.prompts,
                                            groq.API, model, temperature,
                                            top.p, embedding.model, target.N,
                                            domain, scale.title, item.examples,
                                            audience, item.type.definitions,
                                            response.options, prompt.notes,
                                            system.role, EGA.model, EGA.algorithm,
                                            EGA.uni.method, keep.org, items.only,
                                            embeddings.only, adaptive, plot,
                                            silently)


  target.N <- validation$target.N
  EGA.model <- validation$EGA.model
  EGA.uni.method <- validation$EGA.uni.method
  EGA.algorithm <- validation$EGA.algorithm
  model <- validation$model
  item.type.definitions <- validation$item.type.definitions
  item.examples <- validation$item.examples
  item.attributes <- validation$item.attributes
  prompt.notes <- validation$prompt.notes
  main.prompts <- validation$main.prompts
  custom <- validation$custom

  # Begin constructing the prompts
  # first, the system role if one was not provided
  system.role <- create_system.role(domain, scale.title, audience,
                                    response.options, system.role)


  # Create/Modify the prompts
  if(!custom){
    main.prompts <- create_main.prompts(item.attributes, item.type.definitions,
                                      domain, scale.title, prompt.notes,
                                      audience, item.examples)
  } else {
    main.prompts <- modify_main.prompts(main.prompts, item.attributes,
                                        item.type.definitions,
                                        domain, scale.title, prompt.notes,
                                        audience, item.examples)

  }


  # Generate the items for reduction analysis
  items_gen <- generate_items_via_llm(main.prompts, system.role, model, top.p, temperature,
                                  adaptive, silently, groq.API, openai.API, target.N)
  items <- items_gen$items
  success <- items_gen$successful

  if(is.data.frame(items)){
    items$ID <- 1:nrow(items) # create an ID variable
  }

  # return items if requested OR if the run was not a success
  if(items.only || !success){

    if(!success && !silently){
      message("Item generation failed before completion. Returning a data frame of items generated thus far.")
    }

    return(items)
  }


  # Now, generate item embeddings
  attempt_to_embed <- embed_items(embedding.model, openai.API, items, silently)
  success <- attempt_to_embed$success
  embeddings <- attempt_to_embed$embeddings

  # Return partial results if failure or just the embeddings if requested
  if(!success || embeddings.only){
    if(!success && !silently){
      message("Embedding step has failed. Returning a data frame of items generated instead.")
    }

    if(!success){
      return(items)
    }

     return(list(embeddings = embeddings, items = items))
  }


  # Generate item level results
  try_item_level <- run_item_reduction_pipeline(embeddings,
                                         items,
                                         EGA.model,
                                         EGA.algorithm,
                                         EGA.uni.method,
                                         keep.org,
                                         silently)

  if(!try_item_level$success){
    return(try_item_level$item_level)
  }

  item_level <- try_item_level$item_level


  # If successful, generate results for items overall
  try_overall_result <- run_pipeline_for_all(item_level,
                                         items,
                                         embeddings,
                                         EGA.model,
                                         EGA.algorithm,
                                         EGA.uni.method,
                                         keep.org,
                                         silently)

  if(!try_overall_result$success){
    return(item_level)
  }

  overall_result <- try_overall_result$overall_result

  return( list(overall = overall_result,
               item_type_level = item_level))



}


