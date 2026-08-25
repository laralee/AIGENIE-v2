#' Run full pipeline for a single item type
#'
#' @param embedding_matrix Numeric matrix (columns = items for one type)
#' @param items Data frame of items for this type (must include ID, statement, attribute)
#' @param type_name Character. Type label used for tracking/logging.
#' @param model NULL, "glasso", or "TMFG"
#' @param algorithm EGA algorithm
#' @param uni.method EGA uni.method
#' @param corr Character. Correlation method. Default "auto" uses EGAnet's automatic detection.
#' @param ncores Numeric. Number of cores for parallel processing. Default NULL uses EGAnet default.
#' @param boot.iter Numeric. Number of bootstrap iterations. Default 500.
#' @param uva.cut.off Numeric in `[0, 1)`. wTO threshold for `EGAnet::UVA`. Default `0.20`.
#' @param keep.org Logical. Whether to include original items and embeddings
#' @param silently Logical. Whether to print progress statements
#' @param plot Logical. Whether to plot the network plots at the end
#'
#' @return A named list containing pipeline results for this type, including a
#'   `filtering_audit` table with one row per removed item and a
#'   `reduction_summary` table describing NMI and item-count changes by stage.
run_pipeline_for_item_type <- function(embedding_matrix,
                                       items,
                                       type_name,
                                       model = NULL,
                                       algorithm = "walktrap",
                                       uni.method = "louvain",
                                       corr = "auto",
                                       ncores = NULL,
                                       boot.iter = 500,
                                       uva.cut.off = 0.20,
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
      stability_plot = NULL,
      filtering_audit = data.frame(),
      reduction_summary = data.frame()
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
        stability_plot = NULL,
        filtering_audit = data.frame(),
        reduction_summary = data.frame()
      )
  }

  # Check minimum items for meaningful analysis

  if (nrow(items) < 6) {
    warning("[", type_name, "] Too few items (", nrow(items),
            ") for meaningful network analysis. Minimum recommended is 6. Returning partial result.")
    result$final_items <- items
    result$final_N <- nrow(items)
    return(result)
  }

  if(!silently){
    cat("\n\n")
    cat(paste("Starting item pool reduction for", type_name  ,"items.\n"))
    cat("-------------------\n")
  }

  # 1. Convert attribute to numeric factor for true communities
  true_communities <- as.factor(as.integer(factor(items$attribute)))
  names(true_communities) <- items$ID

  # 2. Sparsify the full embedding matrix ONCE up front. This sparse matrix
  # is used (a) as the input to UVA and (b) as the sparse representation
  # passed to select_optimal_embedding -- thresholds are derived from the
  # pre-UVA distribution and the matrix is subset thereafter, matching the
  # AI-GENIE simulation's "sparsify once, subset thereafter" approach.
  sparse_embedding <- sparsify_embeddings(embedding_matrix)

  # 3. Redundancy reduction (UVA) on the SPARSE matrix
  uva_res <- reduce_redundancy_uva(sparse_embedding, items, corr = corr,
                                   uva.cut.off = uva.cut.off)

  if (!uva_res$success) {
    warning("[", type_name, "] UVA failed -- returning partial result.")
    return(result)
  }

  if(!silently){
    cat("Unique Variable Analysis complete.\n")
  }


  result$UVA$n_removed <- uva_res$items_removed
  result$UVA$n_sweeps <- uva_res$iterations
  result$UVA$redundant_pairs <- uva_res$redundant_pairs

  # NEW
  result$UVA$removal_log <- uva_res$removal_log
  # Apply UVA's surviving IDs to BOTH the full and sparse representations
  reduced_sparse <- uva_res$embedding_matrix
  kept_ids       <- colnames(reduced_sparse)
  reduced_full   <- embedding_matrix[, kept_ids, drop = FALSE]
  reduced_items  <- items[items$ID %in% kept_ids, , drop = FALSE]

  # Check if enough items remain after UVA
  if (ncol(reduced_full) < 4) {
    warning("[", type_name, "] Too few items (", ncol(reduced_full),
            ") remaining after UVA for further analysis. Returning partial result.")
    result$final_items <- reduced_items
    result$final_N <- nrow(reduced_items)
    return(result)
  }

  if (keep.org) {
    result$embeddings$full_org   <- embedding_matrix
    result$embeddings$sparse_org <- sparse_embedding
  }


  # 4. Optimal embedding/model selection -- pass BOTH pre-computed
  # representations so the sparse matrix retains its pre-UVA quantile
  # thresholds rather than being re-sparsified on the post-UVA distribution.
  select_res <- select_optimal_embedding(
    embedding_matrix = reduced_full,
    sparse_matrix    = reduced_sparse,
    true_communities = true_communities,
    model            = model,
    algorithm        = algorithm,
    uni.method       = uni.method,
    corr             = corr
  )

  if (!isTRUE(select_res$success)) {
    warning("[", type_name, "] Model selection failed -- returning partial result.")
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
  result$embeddings$selection_log <- select_res$log
  result$EGA.model_selected <- select_res$model
  post_uva_initial_nmi <- select_res$nmi

  # 5. BootEGA filtering
  boot_res <- iterative_stability_check(
    embedding_matrix = selected_embedding,
    items = reduced_items,
    cut.off = 0.75,
    model = select_res$model,
    algorithm = select_res$algorithm,
    uni.method = select_res$uni.method,
    corr = corr,
    ncores = ncores,
    boot.iter = boot.iter,
    silently = silently
  )

  if (!boot_res$successful) {
    warning("[", type_name, "] BootEGA failed -- returning partial result.")
    return(result)
  }

  result$bootEGA$post_uva_initial_boot <- boot_res$boot1
  result$bootEGA$post_uva_final_boot <- boot_res$boot2
  result$bootEGA$n_removed <- nrow(boot_res$items_removed)
  result$bootEGA$items_removed <- boot_res$items_removed

  stable_embedding <- boot_res$embedding
  stable_items <- items[items$ID %in% colnames(stable_embedding), , drop = FALSE]

  # 6. Final EGA + NMI
  final_res <- final_community_detection(
    embedding_matrix = stable_embedding,
    true_communities = true_communities,
    model = select_res$model,
    algorithm = select_res$algorithm,
    uni.method = select_res$uni.method,
    corr = corr
  )

  if (!isTRUE(final_res$success)) {
    warning("[", type_name, "] Final EGA failed -- returning partial result.")
    return(result)
  }

  # Add community labels
  com_df <- data.frame(ID = names(final_res$communities),
                       EGA_com = final_res$communities,
                       stringsAsFactors = FALSE)

  result$final_items <- merge(stable_items, com_df, by = "ID")
  result$final_NMI <- final_res$final_nmi

  result$final_EGA <- final_res$ega

  # Store full + sparse embeddings -- sparse is the SUBSET of the pre-UVA
  # sparse matrix (matching the "sparsify once, subset thereafter" approach).
  final_ids <- intersect(colnames(embedding_matrix), result$final_items$ID)
  result$embeddings$full   <- embedding_matrix[, final_ids, drop = FALSE]
  result$embeddings$sparse <- sparse_embedding[, final_ids, drop = FALSE]

  # 7. Build initial network
  if(!silently){
    cat("\nBuilding initial network based on optimal settings...")
  }


  true_communities <- as.factor(as.integer(factor(items$attribute)))
  names(true_communities) <- items$ID

  # Initial EGA always on the FULL pre-UVA embedding matrix, regardless of
  # which representation was selected as optimal. This isolates `initial_NMI`
  # as the dense-representation baseline so `final_NMI - initial_NMI`
  # captures the end-to-end gain of the AI-GENIE pipeline (sparsification +
  # UVA + bootEGA), matching the framing used in the AI-GENIE simulation.
  initial_res <- final_community_detection(
    embedding_matrix = embedding_matrix,
    true_communities = true_communities,
    model = select_res$model,
    algorithm = select_res$algorithm,
    uni.method = select_res$uni.method,
    corr = corr
  )

  if (!isTRUE(initial_res$success)) {
    warning("[", type_name, "] Initial EGA failed -- returning partial result.")
    return(result)
  }

  # add the communities to the initial items (if retained)
  if(keep.org){
    com_df <- data.frame(ID = names(initial_res$communities),
                         EGA_com = initial_res$communities,
                         stringsAsFactors = FALSE)

    result$initial_items <- merge(items, com_df, by = "ID", all.x = TRUE)
  }

  result$initial_EGA <- initial_res$ega
  result$initial_NMI <- initial_res$final_nmi

  # ============================================================
  # Filtering audit
  # ============================================================

  result$filtering_audit <- build_filtering_audit(
    items = items,
    type_name = type_name,
    uva_log = uva_res$removal_log,
    boot_removed = boot_res$items_removed,
    initial_ega = result$initial_EGA,
    selection_dropped = select_res$dropped_items,
    final_dropped = final_res$items_dropped,
    uva.cut.off = uva.cut.off,
    stability.cut.off = 0.75
  )

  result$reduction_summary <- data.frame(

    stage = c(
      "Initial",
      "Post-UVA / selected embedding",
      "Post-bootEGA / final"
    ),

    N = c(
      nrow(items),
      ncol(selected_embedding),
      nrow(result$final_items)
    ),

    NMI = c(
      result$initial_NMI,
      post_uva_initial_nmi,
      result$final_NMI
    ),

    n_removed_at_stage = c(
      0L,
      nrow(items) - ncol(selected_embedding),
      ncol(selected_embedding) - nrow(result$final_items)
    ),

    stringsAsFactors = FALSE
  )

  result$reduction_summary$delta_NMI <-
    result$reduction_summary$NMI -
    result$initial_NMI

  # For the stability plot's "pre-reduction" bootEGA baseline, keep the
  # SAME representation that was selected as optimal so the stability
  # comparison is apples-to-apples with the post-UVA bootEGA. (This is
  # independent of `initial_NMI`, which is anchored to the full matrix.)
  if(result$embeddings$selected == "full"){
    stability_data <- embedding_matrix
  } else {
    stability_data <- sparse_embedding
  }

  try_stab <- calc_final_stability(result,
                                   stability_data,
                                   algorithm,
                                   uni.method,
                                   corr = corr,
                                   ncores = ncores,
                                   boot.iter = boot.iter,
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


  tryCatch({stability_plot <- plot_stability_comparison(
    boot1 = result$bootEGA$initial_boot_with_redundancies,
    boot2 = result$bootEGA$post_uva_final_boot,
    caption1 = "Original Sample | EGA + TEFI",
    caption2 = "Original Sample | EGA + TEFI",
    nmi1 = result$initial_NMI,
    nmi2 = result$final_NMI,
    title = paste("Bootstrapped Item Stability for", type_name, "Items Before vs After AIGENIE Reduction")
  )
  result$stability_plot <- stability_plot
  result$network_plot <- network_plot },
  error = function(e) {
    warning(paste("Failed to create stability plots for", type_name,
                  "items. Reason:", conditionMessage(e)))
  })


  if(plot && !is.null(result$network_plot)){
    print(result$network_plot)
  }


  return(result)
}


#' Run reduction pipeline for all item types
#'
#' @param embedding_matrix Full embedding matrix (columns = all items)
#' @param items Data frame of all items (must include ID, statement, attribute, type)
#' @param EGA.model NULL, "glasso", or "TMFG"
#' @param EGA.algorithm EGA algorithm
#' @param EGA.uni.method EGA uni.method
#' @param corr Character. Correlation method. Default "auto" uses EGAnet's automatic detection.
#' @param ncores Numeric. Number of cores for parallel processing.
#' @param boot.iter Numeric. Number of bootstrap iterations. Default 500.
#' @param uva.cut.off Numeric in `[0, 1)`. wTO threshold for `EGAnet::UVA`. Default `0.20`.
#' @param keep.org Logical. Whether to include original items and embeddings
#' @param silently Logical. Whether to print progress statements
#' @param plot Logical. Whether to plot the network plots at the end
#'
#' @return A named list of pipeline results, one per item type
run_item_reduction_pipeline <- function(embedding_matrix,
                                        items,
                                        EGA.model = NULL,
                                        EGA.algorithm = "walktrap",
                                        EGA.uni.method = "louvain",
                                        corr = "auto",
                                        ncores = NULL,
                                        boot.iter = 500,
                                        uva.cut.off = 0.20,
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
        corr = corr,
        ncores = ncores,
        boot.iter = boot.iter,
        uva.cut.off = uva.cut.off,
        keep.org = keep.org,
        silently = silently,
        plot = plot
      )
    }, error = function(e) {
      warning("Pipeline failed for type: ", tname, " -- ", e$message)
      success <<- FALSE
      return(NULL)
    })
  })


  names(results) <- unique_types

  return(list(item_level = results,
              success = success))
}




#' Run a pooled post-reduction fit across all item types
#'
#' `run.overall = TRUE` is a fit-only analysis: it takes the union of items that
#' survived the type-level GENIE reductions and evaluates the pooled structure
#' without applying additional UVA or bootEGA filtering. This is intentionally
#' distinct from `all.together = TRUE`, which performs reduction on the entire
#' item pool jointly.
#'
#' @param item_level Named list of completed type-level GENIE results.
#' @param items Original item data frame.
#' @param embeddings Original full embedding matrix (columns = item IDs).
#' @param model NULL, "glasso", or "TMFG". If NULL, the model with the highest
#'   pooled post-reduction NMI on the full embeddings is selected; exact ties
#'   prefer TMFG.
#' @param algorithm EGA community detection algorithm.
#' @param uni.method EGA unidimensionality method.
#' @param corr Character. Correlation method. Default "auto".
#' @param ncores Retained for backward compatibility; no additional bootEGA is
#'   run in the fit-only overall analysis.
#' @param boot.iter Retained for backward compatibility; no additional bootEGA
#'   is run in the fit-only overall analysis. Default 500.
#' @param uva.cut.off Retained for backward compatibility; no additional UVA is
#'   run in the fit-only overall analysis.
#' @param keep.org Logical. Whether to retain original items/embeddings.
#' @param silently Logical. Whether to suppress progress output.
#' @param plot Logical. Whether to print the pooled pre/post network comparison.
#'
#' @return A list with `overall_result` and `success`. `overall_result` contains
#'   pooled pre/post EGA fits, NMI values, the union of type-level survivors, a
#'   pooled filtering audit, and a pooled reduction summary.
#' @keywords internal
run_pipeline_for_all <- function(item_level,
                                 items,
                                 embeddings,
                                 model = NULL,
                                 algorithm = "walktrap",
                                 uni.method = "louvain",
                                 corr = "auto",
                                 ncores = NULL,
                                 boot.iter = 500,
                                 uva.cut.off = 0.20,
                                 keep.org = FALSE,
                                 silently,
                                 plot) {

  # Union of the items retained by the independent type-level reductions.
  surviving_ids <- unique(unlist(lapply(item_level, function(x) {
    if (is.null(x) || is.null(x$final_items) || !"ID" %in% names(x$final_items)) {
      return(character(0))
    }
    as.character(x$final_items$ID)
  }), use.names = FALSE))

  if (length(surviving_ids) < 3L) {
    warning("Overall fit requires at least three type-level surviving items.")
    return(list(overall_result = NULL, success = FALSE))
  }

  item_ids <- as.character(items$ID)
  surviving_ids <- item_ids[item_ids %in% surviving_ids]
  post_items <- items[match(surviving_ids, item_ids), , drop = FALSE]

  if (!is.null(colnames(embeddings))) {
    embeddings_pre <- embeddings[, item_ids, drop = FALSE]
    embeddings_post <- embeddings_pre[, surviving_ids, drop = FALSE]
  } else {
    embeddings_pre <- embeddings
    colnames(embeddings_pre) <- item_ids
    embeddings_post <- embeddings_pre[, surviving_ids, drop = FALSE]
  }

  # Overall truth labels must distinguish the same attribute name appearing in
  # different item types.
  overall_labels_pre <- paste(items$type, items$attribute, sep = "::")
  true_pre <- as.factor(as.integer(factor(overall_labels_pre)))
  names(true_pre) <- item_ids

  overall_labels_post <- paste(post_items$type, post_items$attribute, sep = "::")
  true_post <- as.factor(as.integer(factor(overall_labels_post)))
  names(true_post) <- as.character(post_items$ID)

  # Select the overall network model using only the full post-reduction matrix.
  models <- if (is.null(model)) c("glasso", "TMFG") else model
  model_fits <- lapply(models, function(m) {
    fit <- final_community_detection(
      embedding_matrix = embeddings_post,
      true_communities = true_post,
      model = m,
      algorithm = algorithm,
      uni.method = uni.method,
      corr = corr
    )
    list(model = m, fit = fit)
  })

  valid <- vapply(model_fits, function(z) isTRUE(z$fit$success), logical(1))
  if (!any(valid)) {
    warning("Overall post-reduction EGA failed for all candidate models.")
    return(list(overall_result = NULL, success = FALSE))
  }

  model_fits <- model_fits[valid]
  best <- model_fits[[1L]]
  if (length(model_fits) > 1L) {
    for (z in model_fits[-1L]) {
      if (z$fit$final_nmi > best$fit$final_nmi ||
          (z$fit$final_nmi == best$fit$final_nmi &&
           z$model == "TMFG" && best$model != "TMFG")) {
        best <- z
      }
    }
  }

  model_selected <- best$model
  final_res <- best$fit

  initial_res <- final_community_detection(
    embedding_matrix = embeddings_pre,
    true_communities = true_pre,
    model = model_selected,
    algorithm = algorithm,
    uni.method = uni.method,
    corr = corr
  )

  if (!isTRUE(initial_res$success)) {
    warning("Overall pre-reduction EGA failed.")
    return(list(overall_result = NULL, success = FALSE))
  }

  final_com <- data.frame(
    ID = names(final_res$communities),
    EGA_com = final_res$communities,
    stringsAsFactors = FALSE
  )
  final_items <- merge(post_items, final_com, by = "ID", all.x = TRUE, sort = FALSE)

  # Combined type-level audit, but replace the within-type loading diagnostics
  # with pooled pre-reduction network loadings when an overall fit is available.
  filtering_audit <- combine_filtering_audits(item_level)
  if (nrow(filtering_audit) > 0L) {
    pooled_net <- network_loading_diagnostics(initial_res$ega, items)
    net_cols <- grep("^pre_reduction_", names(filtering_audit), value = TRUE)
    if (length(net_cols) > 0L) {
      filtering_audit[net_cols] <- NULL
    }
    filtering_audit$.audit_order <- seq_len(nrow(filtering_audit))
    filtering_audit <- merge(
      filtering_audit,
      pooled_net,
      by = "ID",
      all.x = TRUE,
      sort = FALSE
    )
    filtering_audit <- filtering_audit[order(filtering_audit$.audit_order), , drop = FALSE]
    filtering_audit$.audit_order <- NULL
    rownames(filtering_audit) <- NULL
  }

  reduction_summary <- data.frame(
    stage = c("Initial pooled", "Post-type-reduction pooled"),
    N = c(nrow(items), nrow(final_items)),
    NMI = c(initial_res$final_nmi, final_res$final_nmi),
    n_removed_at_stage = c(0L, nrow(items) - nrow(post_items)),
    stringsAsFactors = FALSE
  )
  reduction_summary$delta_NMI <- reduction_summary$NMI - initial_res$final_nmi

  overall_result <- list(
    final_NMI = final_res$final_nmi,
    initial_NMI = initial_res$final_nmi,
    embeddings = list(
      selected = "full",
      full = embeddings_post,
      sparse = sparsify_embeddings(embeddings_pre)[, surviving_ids, drop = FALSE]
    ),
    EGA.model_selected = model_selected,
    final_items = final_items,
    final_EGA = final_res$ega,
    initial_EGA = initial_res$ega,
    start_N = nrow(items),
    final_N = nrow(final_items),
    network_plot = NULL,
    stability_plot = NULL,
    filtering_audit = filtering_audit,
    reduction_summary = reduction_summary
  )

  if (keep.org) {
    initial_com <- data.frame(
      ID = names(initial_res$communities),
      EGA_com = initial_res$communities,
      stringsAsFactors = FALSE
    )
    overall_result$initial_items <- merge(items, initial_com, by = "ID", all.x = TRUE, sort = FALSE)
    overall_result$embeddings$full_org <- embeddings_pre
    overall_result$embeddings$sparse_org <- sparsify_embeddings(embeddings_pre)
  }

  overall_result$network_plot <- tryCatch(
    plot_comparison(
      p1 = overall_result$initial_EGA,
      p2 = overall_result$final_EGA,
      caption1 = "Network Plot for Items Pre-Reduction",
      caption2 = "Network Plot for Items Post-Reduction",
      nmi1 = overall_result$initial_NMI,
      nmi2 = overall_result$final_NMI,
      title = "Overall Network Before vs After Type-Level AIGENIE Reduction"
    ),
    error = function(e) {
      warning("Failed to create overall network plot: ", conditionMessage(e))
      NULL
    }
  )

  if (isTRUE(plot) && !is.null(overall_result$network_plot)) {
    print(overall_result$network_plot)
  }

  if (!silently) {
    cat("\nOverall post-reduction fit complete.")
  }

  list(overall_result = overall_result, success = TRUE)
}
