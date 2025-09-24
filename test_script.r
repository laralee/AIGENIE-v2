# basic functionality of AIGENIE

openai_API <- "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"

item_attributes <- list(

  openness = c("curious", "artistic", "philisophical", "willing to change"),
  agreeableness = c("friendly", "humble", "team player", "outgoing")

)

get_embeddings <- AIGENIE.v2::AIGENIE_v2(item.attributes = item_attributes,
                       openai.API = openai_API,
                       embeddings.only = TRUE)

get_embeddings <- readRDS("embeddings_testing.RDS")

embeds <- get_embeddings$embeddings
items <- get_embeddings$items


openness_items <- items[items$type == "openness",]
openness_embeds <- embeds[,colnames(embeds) %in% openness_items$ID]

test_one_item_type <- AIGENIE.v2:::run_pipeline_for_item_type( openness_embeds,
                                                               openness_items,
                                                               "openness",
                                                               model = NULL,
                                                               algorithm = "walktrap",
                                                               uni.method = "louvain",
                                                               keep.org = TRUE,
                                                               silently = FALSE

)




test_item_reduction <- AIGENIE.v2:::run_item_reduction_pipeline(embeds,
                                        items,
                                        EGA.model = NULL,
                                        EGA.algorithm = "walktrap",
                                        EGA.uni.method = "louvain",
                                        keep.org = TRUE,
                                        silently = FALSE)

overall_result <- AIGENIE.v2:::run_pipeline_for_all(test_item_reduction$item_level,
                                       items,
                                       embeds,
                                       model = NULL,
                                       algorithm = "walktrap",
                                       uni.method = "louvain",
                                       keep.org = TRUE,
                                       silently = FALSE)



test2 <- AIGENIE.v2::AIGENIE_v2(item.attributes = item_attributes,
                                openai.API = openai_API,
                                keep.org = TRUE)









