#' Run full pipeline for a single item type
#'
#' @param embedding_matrix Numeric matrix (columns = items for one type)
#' @param items Data frame of items for this type (must include ID, statement, attribute)
#' @param type_name Character. Type label used for tracking/logging.
#' @param model NULL, "glasso", or "TMFG"
#' @param algorithm EGA algorithm
#' @param uni.method EGA uni.method
#' @param keep.org Logical. Whether to include original items and embeddings
#' @param verbose Logical
#'
#' @return A named list containing pipeline results for this type
run_pipeline_for_item_type <- function(embedding_matrix,
                                       items,
                                       type_name,
                                       model = NULL,
                                       algorithm = "walktrap",
                                       uni.method = "louvain",
                                       keep.org = FALSE,
                                       verbose = FALSE,
                                       silently) {

  log_msg <- function(...) if (verbose) message("[", type_name, "] ", ...)


  if(keep.org){
    result <- list(
      final_NMI = NULL,
      embeddings = list(),
      UVA = list(),
      bootEGA = list(),
      EGA.model_selected = NULL,
      final_items = NULL,
      initial_items = items
    )} else {
      result <- list(
        final_NMI = NULL,
        embeddings = list(),
        UVA = list(),
        bootEGA = list(),
        EGA.model_selected = NULL,
        final_items = NULL
      )
  }

  if(!silently){
    cat("\n\n")
    cat(paste("Starting item pool reduction for", type_name  ,"items.\n"))
    cat("-------------------\n")
  }

  # 1. Convert attribute to numeric factor for true communities
  true_communities <- as.factor(as.integer(factor(items$attribute)))
  names(true_communities) <- items$ID

  # 2. Redundancy reduction (UVA)

  uva_res <- reduce_redundancy_uva(embedding_matrix, items, silently = !verbose)

  if (!uva_res$success) {
    warning("[", type_name, "] UVA failed — returning partial result.")
    return(result)
  }

  if(!silently){
    cat("Unique Variable Analysis complete.\n")
  }


  result$UVA$n_removed <- uva_res$items_removed
  result$UVA$n_sweeps <- uva_res$iterations
  result$UVA$redundant_pairs <- uva_res$redundant_pairs

  reduced_matrix <- uva_res$embedding_matrix
  reduced_items <- items[items$ID %in% colnames(reduced_matrix), , drop = FALSE]

  if (keep.org) {
    result$embeddings$full_org <- embedding_matrix
    result$embeddings$sparse_org <- sparsify_embeddings(embedding_matrix)
  }

  # 3. Optimal embedding/model selection
  log_msg("Selecting optimal model/embedding...")
  select_res <- select_optimal_embedding(
    embedding_matrix = reduced_matrix,
    true_communities = true_communities,
    model = model,
    algorithm = algorithm,
    uni.method = uni.method,
    verbose = verbose
  )

  if (!isTRUE(select_res$success)) {
    warning("[", type_name, "] Model selection failed — returning partial result.")
    return(result)
  }

  if(!silently){
    if(is.null(model)){
      cat("Optimal EGA model and embedding type found.\n")
    } else {
      cat("Optimal embedding type found.\n")
    }

  }


  selected_embedding <- select_res$best_embedding_matrix
  result$embeddings$selected <- select_res$embedding_type
  result$EGA.model_selected <- select_res$model
  initial_nmi <- select_res$nmi

  # 4. BootEGA filtering
  log_msg("Running bootEGA...")
  boot_res <- iterative_stability_check(
    embedding_matrix = selected_embedding,
    items = items,
    cut.off = 0.75,
    model = select_res$model,
    algorithm = select_res$algorithm,
    uni.method = select_res$uni.method,
    silently = silently
  )

  if (!boot_res$successful) {
    warning("[", type_name, "] BootEGA failed — returning partial result.")
    return(result)
  }

  result$bootEGA$initial_boot <- boot_res$boot1
  result$bootEGA$final_boot <- boot_res$boot2
  result$bootEGA$n_removed <- nrow(boot_res$items_removed)
  result$bootEGA$items_removed <- boot_res$items_removed

  stable_embedding <- boot_res$embedding
  stable_items <- items[items$ID %in% colnames(stable_embedding), , drop = FALSE]

  # 5. Final EGA + NMI
  log_msg("Final EGA...")
  final_res <- final_community_detection(
    embedding_matrix = stable_embedding,
    true_communities = true_communities,
    model = select_res$model,
    algorithm = select_res$algorithm,
    uni.method = select_res$uni.method,
    verbose = verbose
  )

  if (!isTRUE(final_res$success)) {
    warning("[", type_name, "] Final EGA failed — returning partial result.")
    return(result)
  }

  # Add community labels
  com_df <- data.frame(ID = names(final_res$communities),
                       EGA_com = final_res$communities,
                       stringsAsFactors = FALSE)

  result$final_items <- merge(stable_items, com_df, by = "ID")
  result$final_NMI <- final_res$final_nmi

  # Store full + sparse embeddings
  full_embeds_final <- embedding_matrix[,colnames(embedding_matrix) %in% result$final_items$ID]
  result$embeddings$full <- full_embeds_final
  result$embeddings$sparse <- sparsify_embeddings(full_embeds_final, silently = TRUE)


  return(result)
}



#' Run Full Item Reduction Pipeline Across All Item Types
#'
#' @param embedding_matrix Numeric matrix of all items (columns = items)
#' @param items Data frame of all item metadata (must include ID, type, statement, attribute)
#' @param EGA.model NULL, "glasso", or "TMFG"
#' @param EGA.algorithm Character. EGA algorithm to use (default = "walktrap")
#' @param EGA.uni.method Character. Unidimensionality method (default = "louvain")
#' @param verbose Logical. Print progress?
#'
#' @return A named list of pipeline results, one per item type
run_item_reduction_pipeline <- function(embedding_matrix,
                                        items,
                                        EGA.model = NULL,
                                        EGA.algorithm = "walktrap",
                                        EGA.uni.method = "louvain",
                                        keep.org,
                                        silently,
                                        verbose = FALSE) {

  # --- Prepare ---
  unique_types <- unique(items$type)

  # Split by type
  embedding_split <- lapply(unique_types, function(t) {
    cols <- items$ID[items$type == t]
    embedding_matrix[, cols, drop = FALSE]
  })
  items_split <- split(items, items$type)

  names(embedding_split) <- unique_types

  # --- Run pipeline ---
  results <- lapply(unique_types, function(tname) {
    tryCatch({
      run_pipeline_for_item_type(
        embedding_matrix = embedding_split[[tname]],
        items = items_split[[tname]],
        type_name = tname,
        model = EGA.model,
        algorithm = EGA.algorithm,
        uni.method = EGA.uni.method,
        keep.org = keep.org,
        verbose = verbose,
        silently = silently
      )
    }, error = function(e) {
      warning("Pipeline failed for type: ", tname, " — ", e$message)
      return(NULL)
    })
  })


  names(results) <- unique_types

  return(results)
}




