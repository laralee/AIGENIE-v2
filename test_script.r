# basic functionality of AIGENIE

openai_API <- "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"

item_attributes <- list(

  openness = c("curious", "artistic", "philisophical", "willing to change"),
  agreeableness = c("friendly", "humble", "team player", "outgoing")

)

get_embeddings2 <- AIGENIE.v2::AIGENIE_v2(item.attributes = item_attributes,
                       hf.token = "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",
                       embedding.model = "BAAI/bge-large-en-v1.5",
                       embeddings.only = TRUE,
                       openai.API = openai_API)

get_embeddings <- readRDS("embeddings_testing.RDS")

embeds <- get_embeddings$embeddings
items <- get_embeddings$items


openness_items <- items[items$type == "openness",]
openness_embeds <- embeds[,colnames(embeds) %in% openness_items$ID]

test_one_item_type <- AIGENIE.v2::GENIE(items = items,
                                        embedding.matrix = embeds)


test_item_reduction <- AIGENIE.v2:::run_item_reduction_pipeline(embeds,
                                        items,
                                        EGA.model = NULL,
                                        EGA.algorithm = "walktrap",
                                        EGA.uni.method = "louvain",
                                        keep.org = TRUE,
                                        silently = FALSE,
                                        plot = TRUE)

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




hf_embeddings <- AIGENIE.v2:::embed_items_huggingface(items = openness_items,
  hf.token = "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", silently = FALSE, embedding.model = "BAAI/bge-large-en-v1.5")

'bert-base-uncased'


local_embeds <- AIGENIE.v2:::embed_items_local(embedding.model = 'bert-base-uncased',
                                               items = openness_items)


item.type.definitions <- NULL
domain <- "personality"
scale.title <- "Partial Big Five Assessment"
audience <- "adults in the US"
prompt.notes <- NULL
item.examples <- NULL
response.options <- NULL
system.role <- NULL

system.role <- AIGENIE.v2:::create_system.role(domain, scale.title, audience,
                                               response.options, system.role)

main.prompts <- AIGENIE.v2:::create_main.prompts(item.attributes = item_attributes, item.type.definitions,
                                                 domain, scale.title, prompt.notes,
                                                 audience, item.examples)




model_path <- "/Users/llr7cb/Downloads/phi-3.5-mini-instruct-q4_k_m.gguf"


temperature <- 1
top.p <- 1

adaptive <- FALSE
silently <- FALSE
target.N <- list(openness = 60,
                 agreeableness = 60)

local_items <- AIGENIE.v2::local_AIGENIE(item.attributes = item_attributes,
                                         model.path = model_path,
                                         adaptive = FALSE,
                                         domain = "personality",
                                         scale.title = "Shortened Big Five Assessment")



