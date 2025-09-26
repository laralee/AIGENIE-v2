#' Run full pipeline for a single item type
#'
#' @param embedding_matrix Numeric matrix (columns = items for one type)
#' @param items Data frame of items for this type (must include ID, statement, attribute)
#' @param type_name Character. Type label used for tracking/logging.
#' @param model NULL, "glasso", or "TMFG"
#' @param algorithm EGA algorithm
#' @param uni.method EGA uni.method
#' @param keep.org Logical. Whether to include original items and embeddings
#' @param silently Logical. Whether to print progress statements
#' @param plot Logicial. Whether to plot the network plots at the end
#'
#' @return A named list containing pipeline results for this type
run_pipeline_for_item_type <- function(embedding_matrix,
                                       items,
                                       type_name,
                                       model = NULL,
                                       algorithm = "walktrap",
                                       uni.method = "louvain",
                                       keep.org = FALSE,
                                       silently,
                                       plot) {


  if(keep.org){
    result <- list(
      final_NMI = NULL,
      initial_NMI = NULL,
      embeddings = list(),
      UVA = list(),
      bootEGA = list(),
      EGA.model_selected = NULL,
      final_items = NULL,
      initial_items = items,
      final_EGA = NULL,
      initial_EGA = NULL,
      start_N = nrow(items),
      final_N = NULL,
      network_plot = NULL,
      stability_plot = NULL
    )} else {
      result <- list(
        final_NMI = NULL,
        initial_NMI = NULL,
        embeddings = list(),
        UVA = list(),
        bootEGA = list(),
        EGA.model_selected = NULL,
        final_items = NULL,
        final_EGA = NULL,
        initial_EGA = NULL,
        start_N = nrow(items),
        final_N = NULL,
        network_plot = NULL,
        stability_plot = NULL
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

  uva_res <- reduce_redundancy_uva(embedding_matrix, items)

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
  select_res <- select_optimal_embedding(
    embedding_matrix = reduced_matrix,
    true_communities = true_communities,
    model = model,
    algorithm = algorithm,
    uni.method = uni.method
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
  final_res <- final_community_detection(
    embedding_matrix = stable_embedding,
    true_communities = true_communities,
    model = select_res$model,
    algorithm = select_res$algorithm,
    uni.method = select_res$uni.method
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

  result$final_EGA <- final_res$ega

  # Store full + sparse embeddings
  full_embeds_final <- embedding_matrix[,colnames(embedding_matrix) %in% result$final_items$ID]
  result$embeddings$full <- full_embeds_final
  result$embeddings$sparse <- sparsify_embeddings(full_embeds_final)

  # 6. Build initial network
  if(!silently){
    cat("\nBuilding initial network based on optimal settings...")
  }


  true_communities <- as.factor(as.integer(factor(items$attribute)))
  names(true_communities) <- items$ID

  initial_res <- final_community_detection(
    embedding_matrix = embedding_matrix,
    true_communities = true_communities,
    model = select_res$model,
    algorithm = select_res$algorithm,
    uni.method = select_res$uni.method
  )

  if (!isTRUE(initial_res$success)) {
    warning("[", type_name, "] Initial EGA failed — returning partial result.")
    return(result)
  }

  # add the communities to the initial items (if retained)
  if(keep.org){
    com_df <- data.frame(ID = names(initial_res$communities),
                         EGA_com = initial_res$communities,
                         stringsAsFactors = FALSE)

    result$initial_items <- merge(items, com_df, by = "ID")
  }

  result$initial_EGA <- initial_res$ega
  result$initial_NMI <- initial_res$final_nmi


  if(result$embeddings$selected == "full"){
    data <- embedding_matrix}
  else {
    data <- sparsify_embeddings(embedding_matrix)}

  try_stab <- calc_final_stability(result,
                                   data,
                                   algorithm,
                                   uni.method,
                                   silently)

  if(try_stab$successful){
    result <- try_stab$result
  }

  # add the final number of items
  result$final_N <- nrow(result$final_items)


  if(!silently){
    cat(paste0("\nReduction for ",type_name," items complete."))
  }

 tryCatch({network_plot <- plot_comparison(
    p1 = result$initial_EGA,
    p2 = result$final_EGA,
    caption1 = "Network Plot for Items Pre-Reduction",
    caption2 = "Network Plot for Items Post-Reduction",
    nmi1 = result$initial_NMI,
    nmi2 = result$final_NMI,
    title = paste("Network Plots for", type_name, "Items Before vs After AIGENIE Reduction")
  )
  result$network_plot <- network_plot },
  error = function(e) {
    warning(paste("Failed to create network plots for", type_name, "items."))
  })





  tryCatch({stability_plot <- plot_comparison(
    p1 = result$bootEGA$initial_boot_with_redundancies,
    p2 = result$bootEGA$final_boot,
    caption1 = "Stability Plot for Items Pre-Reduction",
    caption2 = "Stability Plot for Items Post-Reduction",
    nmi1 = result$initial_NMI,
    nmi2 = result$final_NMI,
    title = paste("Bootstrapped Item Stability for", type_name, "Items Before vs After AIGENIE Reduction")
  )
  result$stability_plot <- stability_plot
  }, error = function(e) {
    warning(paste("Failed to create stability plots for", type_name, "items."))
  })



  if(plot && !is.null(result$network_plot)){
    plot(network_plot)
  }

  return(result)
}



#' Run Full Item Reduction Pipeline Across All Item Types
#'
#' @param embedding_matrix Numeric matrix of all items (columns = items)
#' @param items Data frame of all item metadata (must include ID, type, statement, attribute)
#' @param EGA.model NULL, "glasso", or "TMFG"
#' @param EGA.algorithm Character. EGA algorithm to use (default = "walktrap")
#' @param EGA.uni.method Character. Unidimensionality method (default = "louvain")
#' @param silently Logical. Print progress?
#' @param plot logical. Should the network plot be printed
#'
#' @return A named list of pipeline results, one per item type
run_item_reduction_pipeline <- function(embedding_matrix,
                                        items,
                                        EGA.model = NULL,
                                        EGA.algorithm = "walktrap",
                                        EGA.uni.method = "louvain",
                                        keep.org,
                                        silently,
                                        plot) {

  # --- Prepare ---
  unique_types <- unique(items$type)
  success <- TRUE

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
        silently = silently,
        plot = plot
      )
    }, error = function(e) {
      warning("Pipeline failed for type: ", tname, " — ", e$message)
      success <<- FALSE
      return(NULL)
    })
  })


  names(results) <- unique_types

  return(list(item_level = results,
              success = success))
}




#' Run full pipeline for all items in the sample
#'
#' @param item_level AIGENIE results on the item level
#' @param items all items generated for the initial item pool
#' @param embeddings all embeddings created for the initial item pool
#' @param model NULL, "glasso", or "TMFG"
#' @param algorithm EGA algorithm
#' @param uni.method EGA uni.method
#' @param keep.org Logical. Whether to include original items and embeddings
#' @param silently Logical. Whether to print progress statements
#' @param plot logical. Whether to plot the network plot
#'
#' @return A named list containing pipeline results for this type
run_pipeline_for_all <- function(item_level,
                                 items,
                                 embeddings,
                                 model = NULL,
                                 algorithm = "walktrap",
                                 uni.method = "louvain",
                                 keep.org = FALSE,
                                 silently,
                                 plot) {


  if(keep.org){
    overall_result <- list(
      final_NMI = NULL,
      initial_NMI = NULL,
      embeddings = list(),
      EGA.model_selected = NULL,
      final_items = NULL,
      initial_items = items,
      final_EGA = NULL,
      initial_EGA = NULL,
      start_N = nrow(items),
      final_N = NULL,
      network_plot = NULL
    )} else {
      overall_result <- list(
        final_NMI = NULL,
        initial_NMI = NULL,
        embeddings = list(),
        EGA.model_selected = NULL,
        final_items = NULL,
        final_EGA = NULL,
        initial_EGA = NULL,
        start_N = nrow(items),
        final_N = NULL,
        network_plot = NULL
      )
    }

  success <- TRUE

  # Build overall data frame and embedding matrix
  if (keep.org) {
    overall_result$embeddings$full_org <- embeddings
    overall_result$embeddings$sparse_org <- sparsify_embeddings(embeddings)
  }

  df_list <- lapply(item_level, function(x) x$final_items)
  final_items <- do.call(rbind, df_list)

  overall_result$final_items <- final_items

  df_list <- lapply(item_level, function(x) x$embeddings$full)
  final_embeddings <- do.call(cbind, df_list)

  overall_result$embeddings$full <- final_embeddings
  overall_result$embeddings$sparse <- sparsify_embeddings(final_embeddings)

  # Find the final true communities label
  communities <- paste(final_items$type, final_items$attribute, sep = "_")
  communities <- as.numeric(as.factor(communities))
  communities <- as.list(communities)
  names(communities) <- final_items$ID


  if(!silently){
    cat("\n\n")
    cat(paste("Starting analysis on all items.\n"))
    cat("-------------------\n")
  }

  # 1. Get communities... already done
  true_communities <- communities

  # 2. Optimal embedding/model selection
  select_res <- select_optimal_embedding(
    embedding_matrix = final_embeddings,
    true_communities = true_communities,
    model = model,
    algorithm = algorithm,
    uni.method = uni.method
  )

  if (!isTRUE(select_res$success)) {
    warning("Building the final overall EGA network has failed — returning partial result.")
    success <- FALSE
    return(list(overall_result = overall_result,
                success = success))
  }

  if(!silently){
    if(is.null(model)){
      cat("Optimal EGA model and embedding type found.\n")
    } else {
      cat("Optimal embedding type found.\n")
    }

  }


  overall_result$embeddings$selected <- select_res$embedding_type
  overall_result$EGA.model_selected <- select_res$model
  overall_result$final_NMI <- select_res$nmi
  overall_result$final_EGA <- select_res$ega


  # 5. Find the initial NMI given optimal settings EGA + NMI
  if(select_res$embedding_type == "sparse"){
    selected_embedding <- sparsify_embeddings(embeddings)
  } else {
    selected_embedding <- embeddings
  }


  # Find the final true communities label
  initial_communities <- paste(items$type, items$attribute, sep = "_")
  initial_communities <- as.numeric(as.factor(initial_communities))
  initial_communities <- as.list(initial_communities)
  names(initial_communities) <- items$ID


  # Run EGA on all items in the item pool

  if(!silently){
  cat("Building initial EGA network based on optimal settings...")
  }


  initial_res <- final_community_detection(
    embedding_matrix = selected_embedding,
    true_communities = initial_communities,
    model = select_res$model,
    algorithm = select_res$algorithm,
    uni.method = select_res$uni.method
  )

  if (!isTRUE(initial_res$success)) {
    warning("EGA failed on all items in the initial item pool — returning partial result.")
    success <- FALSE
    return(list(overall_result = overall_result,
                success = success))
  }

  # Add the initial community detection stats
  overall_result$initial_NMI <- initial_res$final_nmi
  overall_result$initial_EGA <- initial_res$ega

  # Add community labels
  final_items$EGA_com <- NULL # clear the communities found in previous steps
  com_df <- data.frame(ID = names(select_res$found.communities),
                       EGA_com = select_res$found.communities,
                       stringsAsFactors = FALSE)

  overall_result$final_items <- merge(final_items, com_df, by = "ID")

  # add the communities to the initial items (if retained)
  if(keep.org){
    com_df <- data.frame(ID = names(initial_res$communities),
                         EGA_com = initial_res$communities,
                         stringsAsFactors = FALSE)

    overall_result$initial_items <- merge(items, com_df, by = "ID")
  }

  # add the final number of items
  overall_result$final_N <- nrow(overall_result$final_items)

  if(!silently){
    cat("Done.")
  }


  tryCatch({network_plot <- plot_comparison(
    p1 = overall_result$initial_EGA,
    p2 = overall_result$final_EGA,
    caption1 = "Network Plot for Items Pre-Reduction",
    caption2 = "Network Plot for Items Post-Reduction",
    nmi1 = overall_result$initial_NMI,
    nmi2 = overall_result$final_NMI,
    title = "Network Plots for All Items Before vs After AIGENIE Reduction"
  )
  overall_result$network_plot <- network_plot },
  error = function(e) {
    warning(paste("Failed to create network plots for the items overall."))
  })

  if(plot && !is.null(overall_result$network_plot)){
    plot(network_plot)
  }


  return(list(overall_result = overall_result,
              success = success))
}
