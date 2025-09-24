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

      if (!silently) {
        cat("Fallback sparsification also resulted in all zeros. Using original embeddings.\n")
      }

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
#'
#' @return A list with the reduced matrix, sweep metadata, and redundancy log.
reduce_redundancy_uva <- function(embedding_matrix, items) {


  original_embedding <- embedding_matrix
  count   <- 1
  success <- TRUE
  all_redundant_sets <- list()

  # Helper: convert keep_remove to readable set
  extract_redundancy_sets <- function(keep_remove, sweep, items) {
    if (is.null(keep_remove) || length(keep_remove) == 0) return(NULL)

    out <- list()

    # Case 1: paired vectors (two vectors of same length)
    if (length(keep_remove) == 2 &&
        is.character(keep_remove[[1]]) &&
        is.character(keep_remove[[2]]) &&
        length(keep_remove[[1]]) == length(keep_remove[[2]])) {

      keep_ids   <- keep_remove[[1]]
      remove_ids <- keep_remove[[2]]

      for (i in seq_along(keep_ids)) {
        k_id <- keep_ids[i]
        r_id <- remove_ids[i]

        k_stmt <- items$statement[match(k_id, items$ID)]
        r_stmt <- items$statement[match(r_id, items$ID)]

        out[[length(out) + 1]] <- data.frame(
          sweep  = sweep,
          items  = I(list(c(k_stmt, r_stmt))),
          keep   = k_stmt,
          remove = I(list(r_stmt)),
          stringsAsFactors = FALSE
        )
      }

    } else {
      # Case 2: list of sets where first = kept, rest = removed
      for (group in keep_remove) {
        if (length(group) < 2) next

        keep_id    <- group[1]
        remove_ids <- group[-1]

        keep_stmt    <- items$statement[match(keep_id, items$ID)]
        remove_stmts <- items$statement[match(remove_ids, items$ID)]
        group_stmts  <- items$statement[match(group, items$ID)]

        out[[length(out) + 1]] <- data.frame(
          sweep  = sweep,
          items  = I(list(group_stmts)),
          keep   = keep_stmt,
          remove = I(list(remove_stmts)),
          stringsAsFactors = FALSE
        )
      }
    }

    if (length(out) == 0) return(NULL)

    return(do.call(rbind, out))
  }




  # Initial UVA
  uva <- tryCatch({
    EGAnet::UVA(
      data = embedding_matrix,
      cut.off = 0.2,
      reduce.method = "remove",
      plot = FALSE
    )
  }, error = function(e) {
    warning(paste("Initial UVA failed:", conditionMessage(e)))
    success <<- FALSE
    return(NULL)
  })

  if (is.null(uva)) {
    attr(original_embedding, "UVA_count") <- 0
    return(list(
      embedding_matrix = original_embedding,
      iterations = 0,
      items_removed = 0,
      removed_items = character(0),
      redundant_pairs = NULL,
      success = FALSE
    ))
  }

  # Extract from first sweep
  sweep_sets <- extract_redundancy_sets(uva$keep_remove, sweep = count, items)
  if (!is.null(sweep_sets)) all_redundant_sets[[count]] <- sweep_sets

  # Handle early exit (no redundancy)
  if (is.null(uva$keep_remove)) {
    attr(original_embedding, "UVA_count") <- 0
    return(list(
      embedding_matrix = original_embedding,
      iterations = 0,
      items_removed = 0,
      removed_items = character(0),
      redundant_pairs = NULL,
      success = TRUE
    ))
  }

  if (is.null(uva$reduced_data) || ncol(uva$reduced_data) == 0) {
    log_msg("No items remaining after UVA. Returning original embeddings.")
    attr(original_embedding, "UVA_count") <- 0
    return(list(
      embedding_matrix = original_embedding,
      iterations = 0,
      items_removed = 0,
      removed_items = colnames(embedding_matrix),
      redundant_pairs = do.call(rbind, all_redundant_sets),
      success = FALSE
    ))
  }

  previous_uva <- uva

  repeat {
    if (is.null(uva$keep_remove)) break
    if (is.null(uva$reduced_data) || ncol(uva$reduced_data) == 0) {
      log_msg("No items remaining. Stopping iterations.")
      success <- FALSE
      break
    }

    previous_uva <- uva
    count <- count + 1

    uva <- tryCatch({
      EGAnet::UVA(
        data = previous_uva$reduced_data,
        cut.off = 0.2,
        reduce.method = "remove",
        plot = FALSE
      )
    }, error = function(e) {
      success <<- FALSE
      log_msg(sprintf("UVA failed at iteration %d: %s", count, conditionMessage(e)))
      return(NULL)
    })

    if (is.null(uva)) break
    if (!is.null(uva$keep_remove) && length(uva$keep_remove) == 0) break
    if (count > 50) {
      warning("UVA iterations exceeded safety limit (50). Stopping.")
      success <- FALSE
      break
    }

    sweep_sets <- extract_redundancy_sets(uva$keep_remove, sweep = count, items)
    if (!is.null(sweep_sets)) all_redundant_sets[[count]] <- sweep_sets
  }

  if (!is.null(previous_uva$reduced_data) && ncol(previous_uva$reduced_data) > 0) {
    final_mat     <- previous_uva$reduced_data
    start_items   <- colnames(embedding_matrix)
    final_items   <- colnames(final_mat)
    removed_items <- setdiff(start_items, final_items)

    attr(final_mat, "UVA_count")     <- count
    attr(final_mat, "items_removed") <- length(removed_items)

    return(list(
      embedding_matrix = final_mat,
      iterations       = count,
      items_removed    = length(removed_items),
      removed_items    = removed_items,
      redundant_pairs  = do.call(rbind, all_redundant_sets),
      success          = success
    ))
  } else {
    log_msg("No valid reduced data. Returning original embeddings.")
    attr(original_embedding, "UVA_count") <- 0
    return(list(
      embedding_matrix = original_embedding,
      iterations       = 0,
      items_removed    = 0,
      removed_items    = character(0),
      redundant_pairs  = do.call(rbind, all_redundant_sets),
      success          = FALSE
    ))
  }
}




#' Run Final Community Detection with EGA
#'
#' @param embedding_matrix A numeric matrix with items as columns.
#' @param true_communities Named list mapping items to known communities.
#' @param model Network estimation model (e.g., "glasso", "TMFG").
#' @param algorithm Community detection algorithm (e.g., "walktrap").
#' @param uni.method Unidimensionality method passed to EGA.
#'
#' @return A list with final communities, final NMI, dropped items, EGA object, and success flag.
final_community_detection <- function(embedding_matrix,
                                      true_communities,
                                      model = "glasso",
                                      algorithm = "walktrap",
                                      uni.method = "louvain") {

  result <- tryCatch({

    ega <- EGAnet::EGA.fit(
      data = embedding_matrix,
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





#' Select Optimal Embedding and EGA Model Based on NMI
#'
#' @param embedding_matrix A numeric matrix (columns = items).
#' @param true_communities A named list of known communities.
#' @param model Character. One of "glasso", "TMFG", or NULL (to test both).
#' @param algorithm Community detection algorithm (e.g., "walktrap").
#' @param uni.method Unidimensionality method (e.g., "louvain").
#' @param sparsify_function Function to sparsify the embedding.
#'
#' @return A list with best embedding, model, communities, NMI, and comparison log.
select_optimal_embedding <- function(embedding_matrix,
                                     true_communities,
                                     model = NULL,
                                     algorithm = "walktrap",
                                     uni.method = "louvain",
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
#' @param EGA.type Type of EGA (default "EGA.fit").
#'
#' @return A list containing the final embedding, boot objects, items removed per iteration, etc.
iterative_stability_check <- function(embedding_matrix,
                                      items,
                                      cut.off = 0.75,
                                      model = "NULL",
                                      algorithm = "",
                                      uni.method,
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

  # First run
  boot1 <- tryCatch({
    EGAnet::bootEGA(
      data = current_embedding,
      model = model,
      algorithm = algorithm,
      uni.method = uni.method,
      EGA.type = EGA.type,
      plot.itemStability = FALSE,
      plot.typicalStructure = FALSE,
      verbose = !silently,
      seed = 123
    )
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
    emp_dims <- bootstrap$stability$item.stability$item.stability$empirical.dimensions

    # Remove NA
    if (any(is.na(emp_dims))) {
      valid_idx <- which(!is.na(emp_dims))
      current_embedding <- current_embedding[, valid_idx, drop = FALSE]
      emp_dims <- emp_dims[valid_idx]
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

    # Run again
    bootstrap <- tryCatch({
      EGAnet::bootEGA(
        data = current_embedding,
        model = model,
        algorithm = algorithm,
        uni.method = uni.method,
        EGA.type = EGA.type,
        plot.typicalStructure = FALSE,
        plot.itemStability = FALSE,
        verbose = !silently,
        seed = 123
      )
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
#' @param algorithm Community detection algorithm.
#' @param uni.method Unidimensionality method.
#' @param silently A logical flag that decides whether to print output
#' @param EGA.type Type of EGA (default "EGA.fit").
#'
#' @return An updated `results` object with the initial stability object added
calc_final_stability <- function(result,
                                 data,
                                 EGA.algorithm,
                                 EGA.uni.method,
                                 silently,
                                 EGA.type = "EGA.fit"){
  if(!silently){
    cat("\n")
    cat(paste0("Finding network stability of the original item pool...\n"))
  }

  successful <- TRUE

  x <- result

  try_stab <- tryCatch({
      EGAnet::bootEGA(
        data = data,
        model = x$EGA.model_selected,
        algorithm = EGA.algorithm,
        uni.method = EGA.uni.method,
        EGA.type = EGA.type,
        plot.itemStability = FALSE,
        plot.typicalStructure = FALSE,
        verbose = !silently,
        seed = 123
      )
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
#' @return No return value; the function prints the results to the console.
print_results<-function(obj, obj2){

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
