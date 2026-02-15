#' Sparsify Embedding Matrix
#'
#' Applies sparsification to an embedding matrix by zeroing out values between
#' specified quantiles. Includes fallback strategies if initial sparsification
#' results in all zeros.
#'
#' @param embedding_matrix Numeric matrix with items as columns, dimensions as rows
#' @param lower_quantile Lower quantile threshold (default 0.025)
#' @param upper_quantile Upper quantile threshold (default 0.975)
#' @param fallback_lower Fallback lower quantile if first attempt fails (default 0.10)
#' @param fallback_upper Fallback upper quantile if first attempt fails (default 0.90)
#'
#' @return Sparsified embedding matrix with same dimensions as input
#' @details
#' Sparsification process:
#' 1. Zero out values between lower and upper quantiles
#' 2. If result is all zeros, try fallback quantiles
#' 3. If still all zeros, return original matrix
#'
#' `silently` is always `TRUE`. It is only set to `FALSE` for developement
#' and diagnostic purposes.
#'
sparsify_embeddings <- function(embedding_matrix,
                                lower_quantile = 0.025,
                                upper_quantile = 0.975,
                                fallback_lower = 0.10,
                                fallback_upper = 0.90) {

  # Validate input
  if (!is.matrix(embedding_matrix) || !is.numeric(embedding_matrix)) {
    stop("embedding_matrix must be a numeric matrix")
  }

  # Store original for fallback
  original_embedding <- embedding_matrix

  # Helper function for applying sparsification
  apply_sparsification <- function(mat, lower, upper) {
    q <- quantile(mat, probs = c(lower, upper), na.rm = TRUE)
    mat[mat > q[1] & mat < q[2]] <- 0
    return(mat)
  }

  # First attempt with primary quantiles
  embedding_sparse <- apply_sparsification(embedding_matrix, lower_quantile, upper_quantile)

  # Check if all values are zero
  if (all(embedding_sparse == 0, na.rm = TRUE)) {

    # Try with fallback quantiles
    embedding_sparse <- apply_sparsification(embedding_matrix, fallback_lower, fallback_upper)

    # If still all zeros, return original
    if (all(embedding_sparse == 0, na.rm = TRUE)) {

      embedding_sparse <- original_embedding
      attr(embedding_sparse, "sparsification_applied") <- FALSE

    } else {

      attr(embedding_sparse, "sparsification_applied") <- TRUE
      attr(embedding_sparse, "quantiles_used") <- c(lower = fallback_lower, upper = fallback_upper)
    }

  } else {

    attr(embedding_sparse, "sparsification_applied") <- TRUE
    attr(embedding_sparse, "quantiles_used") <- c(lower = lower_quantile, upper = upper_quantile)
  }

  return(embedding_sparse)
}




#' Reduce Redundancy via Iterative UVA (with Redundant Pair Logging)
#'
#' Applies EGAnet::UVA iteratively and logs human-readable redundant item sets.
#'
#' @param embedding_matrix A numeric matrix of embeddings (columns = items).
#' @param items Data frame with `ID` and `statement` columns.
#' @param corr Character. Correlation method to use. Default "auto" uses EGAnet's
#'   automatic correlation detection. Other options: "pearson", "spearman", "cosine".
#'
#' @return A list with the reduced matrix, sweep metadata, and redundancy log.
reduce_redundancy_uva <- function(embedding_matrix, items, corr = "auto") {

  original_embedding <- embedding_matrix
  current_matrix <- embedding_matrix
  count <- 1
  success <- TRUE
  all_redundant_sets <- list()
  all_removed_items <- character(0)

  # Helper: extract redundancy sets and format them
  extract_redundancy_sets <- function(uva_object, sweep, items, current_matrix) {
    if (is.null(uva_object$redundant) || length(uva_object$redundant) == 0) {
      return(NULL)
    }

    # Get the IDs that remain after reduction
    remaining_ids <- NULL
    if (!is.null(uva_object$reduced_data)) {
      remaining_ids <- colnames(uva_object$reduced_data)
    }

    # Get the IDs that were removed
    current_ids <- colnames(current_matrix)
    removed_ids <- setdiff(current_ids, remaining_ids)

    # Process redundancies more carefully
    out <- list()
    processed_removals <- character(0)

    for (i in seq_along(uva_object$redundant)) {
      item_name <- names(uva_object$redundant)[i]
      redundant_with <- uva_object$redundant[[i]]

      # Create groups based on what was actually removed
      # If item_name was removed, find what it was redundant with that was kept
      if (item_name %in% removed_ids) {
        # This item was removed
        kept_partners <- intersect(redundant_with, remaining_ids)
        if (length(kept_partners) > 0) {
          # Use the first kept partner
          kept_id <- kept_partners[1]

          # Create the minimal group
          group <- c(kept_id, item_name)

          # Check if any other removed items are also redundant with the same kept item
          for (other_removed in setdiff(redundant_with, remaining_ids)) {
            if (!(other_removed %in% processed_removals)) {
              group <- c(group, other_removed)
              processed_removals <- c(processed_removals, other_removed)
            }
          }

          if (!(item_name %in% processed_removals)) {
            processed_removals <- c(processed_removals, item_name)

            # Map IDs to statements
            group <- unique(group)
            group_stmts <- items$statement[match(group, items$ID)]
            keep_stmt <- items$statement[match(kept_id, items$ID)]
            remove_ids <- setdiff(group, kept_id)
            remove_stmts <- items$statement[match(remove_ids, items$ID)]

            out[[length(out) + 1]] <- data.frame(
              sweep = sweep,
              items = paste(group_stmts, collapse = "\n "),
              keep = keep_stmt,
              remove = paste(remove_stmts, collapse = "\n "),
              stringsAsFactors = FALSE
            )
          }
        }
      } else if (item_name %in% remaining_ids) {
        # This item was kept - find what was removed because of it
        removed_partners <- intersect(redundant_with, removed_ids)
        if (length(removed_partners) > 0) {
          # Only include partners that haven't been processed yet
          unprocessed_partners <- setdiff(removed_partners, processed_removals)
          if (length(unprocessed_partners) > 0) {
            group <- c(item_name, unprocessed_partners)
            processed_removals <- c(processed_removals, unprocessed_partners)

            # Map IDs to statements
            group_stmts <- items$statement[match(group, items$ID)]
            keep_stmt <- items$statement[match(item_name, items$ID)]
            remove_stmts <- items$statement[match(unprocessed_partners, items$ID)]

            out[[length(out) + 1]] <- data.frame(
              sweep = sweep,
              items = paste(group_stmts, collapse = "\n "),
              keep = keep_stmt,
              remove = paste(remove_stmts, collapse = "\n "),
              stringsAsFactors = FALSE
            )
          }
        }
      }
    }

    if (length(out) == 0) return(NULL)

    return(do.call(rbind, out))
  }

  # Main iterative loop
  repeat {
    # Run UVA with reduce = TRUE to get the reduced matrix
    uva <- tryCatch({
      EGAnet::UVA(
        data = current_matrix,
        corr = corr,
        cut.off = 0.25,
        reduce = TRUE,
        reduce.method = "remove",
        auto = TRUE,
        plot = FALSE
      )
    }, error = function(e) {
      warning(paste("UVA failed at iteration", count, ":", conditionMessage(e)))
      success <- FALSE
      return(NULL)
    })

    # Check for failure or completion
    if (is.null(uva)) break
    if (is.null(uva$redundant) || length(uva$redundant) == 0) {
      # No redundancies found, we're done
      break
    }
    if (is.null(uva$reduced_data) || ncol(uva$reduced_data) == 0) {
      warning("No items remaining after UVA.")
      success <- FALSE
      break
    }

    # Extract redundancy information for this sweep
    sweep_sets <- extract_redundancy_sets(uva, sweep = count, items, current_matrix)
    if (!is.null(sweep_sets)) {
      all_redundant_sets[[count]] <- sweep_sets
    }

    # Track removed items
    current_ids <- colnames(current_matrix)
    reduced_ids <- colnames(uva$reduced_data)
    removed_this_sweep <- setdiff(current_ids, reduced_ids)
    all_removed_items <- c(all_removed_items, removed_this_sweep)

    # Update current matrix for next iteration
    current_matrix <- uva$reduced_data
    count <- count + 1

    # Safety check
    if (count > 50) {
      warning("UVA iterations exceeded safety limit (50). Stopping.")
      success <- FALSE
      break
    }
  }

  # Combine all redundant sets
  redundant_df <- if (length(all_redundant_sets) > 0) {
    do.call(rbind, all_redundant_sets)
  } else {
    NULL
  }

  # Prepare final output
  if (!is.null(current_matrix) && ncol(current_matrix) > 0) {
    attr(current_matrix, "UVA_count") <- count - 1
    attr(current_matrix, "items_removed") <- length(all_removed_items)

    return(list(
      embedding_matrix = current_matrix,
      iterations = count - 1,
      items_removed = length(all_removed_items),
      removed_items = all_removed_items,
      redundant_pairs = redundant_df,
      success = success
    ))
  } else {
    # Return original if reduction failed
    attr(original_embedding, "UVA_count") <- 0

    return(list(
      embedding_matrix = original_embedding,
      iterations = 0,
      items_removed = 0,
      removed_items = character(0),
      redundant_pairs = redundant_df,
      success = FALSE
    ))
  }
}




#' Select Optimal Embedding and EGA Model Based on NMI
#'
#' @param embedding_matrix A numeric matrix (columns = items).
#' @param true_communities A named list of known communities.
#' @param model Character. One of "glasso", "TMFG", or NULL (to test both).
#' @param algorithm Community detection algorithm (e.g., "walktrap").
#' @param uni.method Unidimensionality method (e.g., "louvain").
#' @param corr Character. Correlation method. Default "auto" uses EGAnet's automatic detection.
#' @param sparsify_function Function to sparsify the embedding.
#'
#' @return A list with best embedding, model, communities, NMI, and comparison log.
select_optimal_embedding <- function(embedding_matrix,
                                     true_communities,
                                     model = NULL,
                                     algorithm = "walktrap",
                                     uni.method = "louvain",
                                     corr = "auto",
                                     sparsify_function = sparsify_embeddings) {

  # Prepare embeddings
  full <- embedding_matrix
  sparse <- sparsify_function(embedding_matrix)

  embeddings <- list(full = full, sparse = sparse)

  # Determine which models to test
  models <- if (is.null(model)) c("glasso", "TMFG") else model

  # Setup log of results
  result_log <- data.frame(
    model = character(),
    embedding_type = character(),
    nmi = numeric(),
    stringsAsFactors = FALSE
  )

  best_nmi <- -Inf
  best_result <- NULL

  for (etype in names(embeddings)) {
    emb <- embeddings[[etype]]

    for (m in models) {

      result <- tryCatch({
        ega <- EGAnet::EGA.fit(
          data = emb,
          corr = corr,
          model = m,
          algorithm = algorithm,
          uni.method = uni.method,
          plot.EGA = FALSE,
          verbose = FALSE
        )

        wc <- ega$EGA$wc

        if (is.null(wc)) {
          stop("No community structure returned.")
        }

        # Drop NAs
        if (anyNA(wc)) {
          wc <- wc[!is.na(wc)]
        }

        if (length(wc) < 2) {
          stop("Too few items assigned to communities.")
        }

        this_nmi <- igraph::compare(
          unlist(true_communities[names(wc)]),
          wc,
          method = "nmi"
        )

        result_log <- rbind(result_log, data.frame(
          model = m,
          embedding_type = etype,
          nmi = this_nmi
        ))

        if (this_nmi > best_nmi ||
            (this_nmi == best_nmi && m == "TMFG" && best_result$model != "TMFG") ||
            (this_nmi == best_nmi && m == best_result$model && etype == "sparse" && best_result$embedding_type != "sparse"))
        {
          best_nmi <- this_nmi
          best_result <- list(
            best_embedding_matrix = emb[, names(wc), drop = FALSE],
            embedding_type        = etype,
            model                 = m,
            algorithm             = algorithm,
            uni.method            = uni.method,
            communities           = wc,
            found.communities     = ega$EGA$wc,
            nmi                   = this_nmi,
            dropped_items         = setdiff(colnames(emb), names(wc)),
            ega                   = ega
          )

        }

        NULL
      }, error = function(e) {
        NULL
      })
    }
  }

  if (is.null(best_result)) {
    return(list(
      success = FALSE,
      log = result_log
    ))
  }

  best_result$success <- TRUE
  best_result$log <- result_log
  return(best_result)
}


#' Iteratively run BootEGA to ensure structural stability of items
#'
#' @param embedding_matrix Numeric matrix of item embeddings (columns = items).
#' @param items Data frame containing at least `ID` and `statement`.
#' @param cut.off Numeric. Minimum stability required to retain an item.
#' @param model Network estimation model (e.g., "glasso", "TMFG").
#' @param algorithm Community detection algorithm.
#' @param uni.method Unidimensionality method.
#' @param corr Character. Correlation method. Default "auto" uses EGAnet's automatic detection.
#' @param ncores Numeric. Number of cores for parallel processing. Default NULL uses EGAnet default.
#' @param boot.iter Numeric. Number of bootstrap iterations. Default 100.
#' @param EGA.type Type of EGA (default "EGA.fit").
#' @param silently Logical. Suppress output.
#'
#' @return A list containing the final embedding, boot objects, items removed per iteration, etc.
iterative_stability_check <- function(embedding_matrix,
                                      items,
                                      cut.off = 0.75,
                                      model = "NULL",
                                      algorithm = "",
                                      uni.method,
                                      corr = "auto",
                                      ncores = NULL,
                                      boot.iter = 100,
                                      EGA.type = "EGA.fit",
                                      silently) {

  successful <- TRUE
  count <- 1
  current_embedding <- embedding_matrix
  all_removed <- data.frame()

  if(!silently){
    cat("Beginning BootEGA stability check... ")
    cat("\n")
  }

  # Build bootEGA arguments
  boot_args <- list(
    data = current_embedding,
    corr = corr,
    model = model,
    algorithm = algorithm,
    uni.method = uni.method,
    iter = boot.iter,
    EGA.type = EGA.type,
    plot.itemStability = FALSE,
    plot.typicalStructure = FALSE,
    verbose = !silently,
    seed = 123
  )

  # Add ncores only if specified (otherwise use EGAnet default)
  if (!is.null(ncores)) {
    boot_args$ncores <- ncores
  }

  # First run
  boot1 <- tryCatch({
    do.call(EGAnet::bootEGA, boot_args)
  }, error = function(e) {
    successful <<- FALSE
    return(NULL)
  })

  if (is.null(boot1)) {
    return(list(
      embedding = embedding_matrix,
      boot1 = NULL,
      boot2 = NULL,
      iterations = 0,
      items_removed = NULL,
      successful = FALSE
    ))
  }

  bootstrap <- boot1
  current_boot <- NULL

  repeat {
    # Safely extract empirical dimensions
    emp_dims <- tryCatch({
      bootstrap$stability$item.stability$item.stability$empirical.dimensions
    }, error = function(e) NULL)

    # Check if emp_dims is valid
    if (is.null(emp_dims) || length(emp_dims) == 0) {
      warning("Could not extract item stability. Returning current results.")
      successful <- FALSE
      break
    }

    # Remove NA - with safe check
    na_check <- is.na(emp_dims)
    if (any(na_check, na.rm = TRUE)) {
      valid_idx <- which(!na_check)
      if (length(valid_idx) == 0) {
        warning("All items have NA stability. Returning current results.")
        successful <- FALSE
        break
      }
      current_embedding <- current_embedding[, valid_idx, drop = FALSE]
      emp_dims <- emp_dims[valid_idx]
    }

    # Check minimum items for analysis
    if (ncol(current_embedding) < 3) {
      warning("Too few items remaining for stability analysis. Returning current results.")
      successful <- FALSE
      break
    }

    # Identify unstable items
    unstable_idx <- which(emp_dims < cut.off)
    unstable_ids <- colnames(current_embedding)[unstable_idx]

    if (length(unstable_ids) == 0) {
      # Done — all stable
      break
    }

    # Log removed items for this run
    removed_df <- items[items$ID %in% unstable_ids, , drop = FALSE]
    removed_df$boot_run_removed <- count
    all_removed <- rbind(all_removed, removed_df)

    # Filter matrix
    current_embedding <- current_embedding[, -unstable_idx, drop = FALSE]
    count <- count + 1

    # Update boot_args with new data
    boot_args$data <- current_embedding

    # Run again
    bootstrap <- tryCatch({
      do.call(EGAnet::bootEGA, boot_args)
    }, error = function(e) {
      successful <<- FALSE
      return(NULL)
    })

    if (is.null(bootstrap)) {
      successful <- FALSE
      break
    }

    current_boot <- bootstrap

    # Stop runaway
    if (count > 25 || ncol(current_embedding) == 0) {
      successful <- FALSE
      break
    }
  }

  # If never updated, final = initial
  if (is.null(current_boot)) current_boot <- boot1

  if(!silently){
    cat("Done.")
  }

  return(list(
    embedding = current_embedding,
    boot1 = boot1,
    boot2 = current_boot,
    iterations = count,
    items_removed = if (nrow(all_removed) > 0) all_removed else NULL,
    successful = successful
  ))
}




#' Run bootstrapped EGA on the initial set of items
#'
#' @param result The running results of the item type level so far.
#' @param data The embedding matrix to be used (either the sparse or full)
#' @param EGA.algorithm The EGA algorithm to be used
#' @param EGA.uni.method The EGA unidensinoality that should be used
#' @param corr Character. Correlation method. Default "auto".
#' @param ncores Numeric. Number of cores for parallel processing.
#' @param boot.iter Numeric. Number of bootstrap iterations.
#' @param silently A logical flag that decides whether to print output
#' @param EGA.type Type of EGA (default "EGA.fit").
#'
#' @return An updated `results` object with the initial stability object added
calc_final_stability <- function(result,
                                 data,
                                 EGA.algorithm,
                                 EGA.uni.method,
                                 corr = "auto",
                                 ncores = NULL,
                                 boot.iter = 100,
                                 silently,
                                 EGA.type = "EGA.fit"){
  if(!silently){
    cat("\n")
    cat(paste0("Finding network stability of the original item pool...\n"))
  }

  successful <- TRUE

  x <- result

  # Build bootEGA arguments
  boot_args <- list(
    data = data,
    corr = corr,
    model = x$EGA.model_selected,
    algorithm = EGA.algorithm,
    uni.method = EGA.uni.method,
    iter = boot.iter,
    EGA.type = EGA.type,
    plot.itemStability = FALSE,
    plot.typicalStructure = FALSE,
    verbose = !silently,
    seed = 123
  )

  # Add ncores only if specified
  if (!is.null(ncores)) {
    boot_args$ncores <- ncores
  }

  try_stab <- tryCatch({
    do.call(EGAnet::bootEGA, boot_args)
  }, error = function(e) {
    warning("Stability check failed. Returning partial results.")
    return(list(successful=FALSE))
  })


  # Add the initial stability
  x$bootEGA$initial_boot_with_redundancies <- try_stab


  if(!silently){
    cat("Done.")
  }


  return(list(successful = successful,
              result = x))
}





#' Print Results
#'
#' Displays a summary of the AI-GENIE analysis results, including the EGA model used, embedding type, starting and final number of items, and NMI values before and after reduction. The summary includes the number of iterations for both UVA (Unique Variable Analysis) and bootstrapped EGA steps.
#'
#' @param obj A list object containing the OVERALL analysis results returned by \code{get_results}.
#' @param obj2 A list object containing the ITEM-TYPE LEVEL analysis results returned by \code{get_results}.
#' @param run.overall A flag denoting if overall results should be printed
#' @return No return value; the function prints the results to the console.
print_results<-function(obj, obj2, run.overall){

  # Print the title
  cat("\n")
  cat("\n")
  cat(paste("                           AI-Genie Results"))
  cat("\n")
  cat("                           ----------------")


  for (i in seq_along(obj2)){

    cat("\n")
    cat("\n")

    EGA.model <- obj2[[i]][["EGA.model_selected"]]
    before_nmi <- obj2[[i]][["initial_NMI"]]
    embedding_type <- obj2[[i]][["embeddings"]][["selected"]]
    after_genie <- obj2[[i]][["final_NMI"]]
    initial_items <- obj2[[i]][["start_N"]]
    final_items <- obj2[[i]][["final_N"]]


    words <- strsplit(paste(names(obj2)[[i]], "items"), " ")[[1]]
    words <- paste0(toupper(substring(words, 1, 1)), substring(words, 2))
    words <- paste(words, collapse = " ")

    cat(paste("                          ", words))
    cat("\n")
    cat(paste("EGA Model:", EGA.model,"    Embeddings Used:", embedding_type,
              "    Staring N:", initial_items, "    Final N:", final_items))
    cat("\n")
    cat(paste0("             Initial NMI: ", round(before_nmi,4) * 100,
               "           Final NMI: ", round(after_genie,4) * 100))
  }

  if(run.overall){ # only print overall results if you have them

    cat("\n")
    cat("\n")

    EGA.model <- obj[["EGA.model_selected"]]
    before_nmi <- obj[["initial_NMI"]]
    embedding_type <- obj[["embeddings"]][["selected"]]
    after_genie <- obj[["final_NMI"]]
    initial_items <- obj[["start_N"]]
    final_items <- obj[["final_N"]]

    cat(paste("                          Overall Sample Results"))
    cat("\n")
    cat(paste("EGA Model:", EGA.model,"    Embeddings Used:", embedding_type,
              "    Staring N:", initial_items, "    Final N:", final_items))
    cat("\n")
    cat(paste0("             Initial NMI: ", round(before_nmi,4) * 100,
               "           Final NMI: ", round(after_genie,4) * 100))
  }

}








#' Plot Comparisons
#'
#' Generates a comparative plot of two network analysis results, typically representing the item network
#' before and after AI-GENIE reduction. The plot includes provided captions, displays NMI values for each network,
#' and incorporates a scale title to contextualize the comparison. The layout may be adjusted based on the
#' \code{ident} parameter.
#'
#' @param p1 An object representing the first network analysis result (e.g., the initial EGA object before reduction).
#' @param p2 An object representing the second network analysis result (e.g., the final EGA object after reduction).
#' @param caption1 A character string to be used as a caption or title for the first network (e.g., "Before AI-GENIE Network").
#' @param caption2 A character string for the second network (e.g., "After AI-GENIE Network").
#' @param nmi1 A numeric value representing the Normalized Mutual Information (NMI) of the first network.
#' @param nmi2 A numeric value representing the NMI of the second network.
#' @param title A character string specifying the title of the plot.
#'
#' @return A plot object that visually compares the two network structures. The plot will typically display
#'         the two networks (either side-by-side or in an overlaid manner) with the provided captions and NMI values.
#'         The exact type of the plot object (e.g., a \code{ggplot} object or a base R plot) depends on the implementation.
plot_comparison <- function(p1, p2, caption1, caption2, nmi2, nmi1, title){


    plot1 <- plot(p1) +
      labs(caption = paste0(caption1," (NMI: ", round(nmi1,4) * 100, ")"))

    plot2 <- plot(p2) +
      labs(caption = paste0(caption2," (NMI: ", round(nmi2,4) * 100, ")"))

    combined_plot <- plot1 + plot2 +
      plot_annotation(
        title = title,
        subtitle = paste0("Change in NMI: ", round((nmi2 - nmi1),4) * 100),
        theme = theme(
          plot.title = element_text(hjust = 0.5, size = 16),
          plot.subtitle = element_text(hjust = 0.5, size = 12)
        )
      )


  return(combined_plot)
}



#' Run Final Community Detection with EGA
#'
#' @param embedding_matrix A numeric matrix with items as columns.
#' @param true_communities Named list mapping items to known communities.
#' @param model Network estimation model (e.g., "glasso", "TMFG").
#' @param algorithm Community detection algorithm (e.g., "walktrap").
#' @param uni.method Unidimensionality method passed to EGA.
#' @param corr Character. Correlation method. Default "auto" uses EGAnet's automatic detection.
#'
#' @return A list with final communities, final NMI, dropped items, EGA object, and success flag.
final_community_detection <- function(embedding_matrix,
                                      true_communities,
                                      model = "glasso",
                                      algorithm = "walktrap",
                                      uni.method = "louvain",
                                      corr = "auto") {

  result <- tryCatch({

    ega <- EGAnet::EGA.fit(
      data = embedding_matrix,
      corr = corr,
      model = model,
      algorithm = algorithm,
      uni.method = uni.method,
      plot.EGA = FALSE,
      verbose = FALSE
    )

    wc <- ega$EGA$wc
    if (is.null(wc)) {
      stop("EGA.fit did not return community structure.")
    }

    # Drop unclustered items (NA communities)
    if (anyNA(wc)) {
      items_dropped <- names(wc)[is.na(wc)]
      wc <- wc[!is.na(wc)]
    } else {
      items_dropped <- character(0)
    }

    # Final NMI using igraph
    final_nmi <- igraph::compare(
      unlist(true_communities[names(wc)]),
      wc,
      method = "nmi"
    )

    list(
      communities   = wc,
      final_nmi     = final_nmi,
      items_dropped = items_dropped,
      ega           = ega,
      success       = TRUE
    )

  }, error = function(e) {

    list(
      communities   = NULL,
      final_nmi     = NA_real_,
      items_dropped = colnames(embedding_matrix),
      ega           = NULL,
      success       = FALSE
    )
  })

  return(result)
}


#' Modify the items data frame to run the reduction on all items together
#'
#' @param items A data frame containing the items either generated by AI or supplied by the user
#'
#' @return A data frame whose "attribute" and "type" columns have been modifies so that the entire sample runs together
run_all_together <- function(items){

  temp <- paste(items$type, items$attribute)
  items$attribute <- temp
  items$type <- rep("All", nrow(items))

  return(items)

}


#' Build the proper return object based on if the initial items should be kept and if an overall analysis was run
#'
#' @param item_type_level A named list containing the results at the item type level
#' @param overall_result A named list containing the results at the overall level
#' @param run.overall A flag saying if a fit analysis should be run on the items overall
#' @param keep.org A flag saying if the original items generated should be kept
#'
#' @return A named list of lists containing the appropriately formated return object
build_return <- function(item_type_level, overall_result,
                         run.overall, keep.org) {

  if(!run.overall){
    # Initialize containers
    full_list <- list()
    initial_full_list <- list()
    sparse_list <- list()
    initial_sparse_list <- list()

    items_list <- list()
    initial_items_list <- list()



    # Loop through each sublist
    for (i in seq_along(item_type_level)) {

      embeddings <- item_type_level[[i]]$embeddings
      items <- item_type_level[[i]]$final_items

      full_list[[i]] <- embeddings$full
      sparse_list[[i]] <- embeddings$sparse
      items_list[[i]] <- items

      if(keep.org){
        initial_items <- item_type_level[[i]]$initial_items

        initial_full_list[[i]] <- embeddings$full_org
        initial_sparse_list[[i]] <- embeddings$sparse_org
        initial_items_list[[i]] <- initial_items
      }

    }

    # Combine all matrices column-wise
    full <- do.call(cbind, full_list)
    sparse <- do.call(cbind, sparse_list)
    items <- do.call(rbind, items_list)

    if(keep.org){
      initial_full <- do.call(cbind, initial_full_list)
      initial_sparse <- do.call(cbind, initial_sparse_list)
      initial_items <- do.call(rbind, initial_items_list)
    }
  } else {
    return(list(item_type_level = item_type_level,
                overall = overall_result))
  }



  if(keep.org){
    overall_obj <- list(final_items = items,
                        initial_items = initial_items,
                        embeddings = list(full_org = initial_full,
                                          sparse_org = initial_sparse,
                                          full = full,
                                          sparse = sparse)
                        )

  } else {

    overall_obj <- list(final_items = items,
                        embeddings = list(full = full,
                                          sparse = sparse)
    )

  }

  return(list(item_type_level = item_type_level,
               overall = overall_obj))
}



