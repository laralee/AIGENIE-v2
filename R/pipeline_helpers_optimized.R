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


#' Extract item-level UVA removal evidence
#'
#' Creates one row per item removed by UVA, retaining the strongest
#' redundant relationship as the primary diagnostic and all redundant
#' partners for auditability.
#'
#' @keywords internal
extract_uva_removal_details <- function(
    uva_object,
    removed_ids,
    remaining_ids,
    items,
    sweep,
    cut.off
) {

  empty <- data.frame(
    ID = character(),
    uva_sweep = integer(),
    redundant_with_ID = character(),
    redundant_with_statement = character(),
    wTO = numeric(),
    all_redundant_with_IDs = character(),
    all_redundant_wTO = character(),
    stringsAsFactors = FALSE
  )

  pairwise <- tryCatch(
    uva_object$wto$pairwise,
    error = function(e) NULL
  )

  if (is.null(pairwise) || nrow(pairwise) == 0) {
    return(empty)
  }

  required <- c("node_i", "node_j", "wto")

  if (!all(required %in% names(pairwise))) {
    return(empty)
  }

  removed_ids   <- as.character(removed_ids)
  remaining_ids <- as.character(remaining_ids)
  item_ids      <- as.character(items$ID)

  out <- lapply(removed_ids, function(rid) {

    hits <- pairwise[
      as.character(pairwise$node_i) == rid |
        as.character(pairwise$node_j) == rid,
      ,
      drop = FALSE
    ]

    if (nrow(hits) == 0) {
      return(NULL)
    }

    hits$partner <- ifelse(
      as.character(hits$node_i) == rid,
      as.character(hits$node_j),
      as.character(hits$node_i)
    )

    # Only relationships that actually exceed the UVA threshold
    hits <- hits[
      !is.na(hits$wto) &
        hits$wto >= cut.off,
      ,
      drop = FALSE
    ]

    if (nrow(hits) == 0) {
      return(NULL)
    }

    # Prefer a redundant partner that survived the current sweep.
    # This gives the most interpretable "removed because redundant with X".
    kept_hits <- hits[
      hits$partner %in% remaining_ids,
      ,
      drop = FALSE
    ]

    candidates <- if (nrow(kept_hits) > 0) {
      kept_hits
    } else {
      hits
    }

    best <- candidates[
      which.max(candidates$wto),
      ,
      drop = FALSE
    ]

    all_hits <- hits[
      order(hits$wto, decreasing = TRUE),
      ,
      drop = FALSE
    ]

    partner <- as.character(best$partner[1])

    data.frame(
      ID = rid,
      uva_sweep = as.integer(sweep),

      redundant_with_ID = partner,

      redundant_with_statement =
        items$statement[
          match(partner, item_ids)
        ],

      wTO = as.numeric(best$wto[1]),

      all_redundant_with_IDs =
        paste(
          unique(as.character(all_hits$partner)),
          collapse = "; "
        ),

      all_redundant_wTO =
        paste(
          sprintf(
            "%s=%.3f",
            as.character(all_hits$partner),
            as.numeric(all_hits$wto)
          ),
          collapse = "; "
        ),

      stringsAsFactors = FALSE
    )
  })

  out <- out[!vapply(out, is.null, logical(1))]

  if (length(out) == 0) {
    return(empty)
  }

  do.call(rbind, out)
}

#' Reduce Redundancy via Iterative UVA (with Redundant Pair Logging)
#'
#' Applies EGAnet::UVA iteratively and logs human-readable redundant item sets.
#'
#' @param embedding_matrix A numeric matrix of embeddings (columns = items).
#' @param items Data frame with `ID` and `statement` columns.
#' @param corr Character. Correlation method to use. Default "auto" uses EGAnet's
#'   automatic correlation detection. Other options: "pearson", "spearman", "cosine".
#' @param uva.cut.off Numeric in `[0, 1)`. The weighted topological overlap
#'   threshold passed to `EGAnet::UVA`. Items with pairwise wTO at or above
#'   this value are flagged as redundant. Default `0.20`.
#'
#' @return A list with the reduced matrix, sweep metadata, human-readable
#'   redundancy groups, and `removal_log`, a tidy item-level table containing
#'   removed IDs, retained redundant partners, and wTO statistics.
reduce_redundancy_uva <- function(embedding_matrix, items, corr = "auto",
                                  uva.cut.off = 0.20) {

  all_removal_details <- list()
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
        cut.off = uva.cut.off,
        reduce = TRUE,
        reduce.method = "remove",
        auto = TRUE,
        verbose = FALSE
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
    removal_details <- extract_uva_removal_details(
      uva_object = uva,
      removed_ids = removed_this_sweep,
      remaining_ids = reduced_ids,
      items = items,
      sweep = count,
      cut.off = uva.cut.off
    )

    if (nrow(removal_details) > 0) {
      all_removal_details[[length(all_removal_details) + 1L]] <-
        removal_details
    }
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

    removal_log <- if (length(all_removal_details) > 0) {
      do.call(rbind, all_removal_details)
    } else {
      data.frame(
        ID = character(),
        uva_sweep = integer(),
        redundant_with_ID = character(),
        redundant_with_statement = character(),
        wTO = numeric(),
        all_redundant_with_IDs = character(),
        all_redundant_wTO = character(),
        stringsAsFactors = FALSE
      )
    }

    return(list(
      embedding_matrix = current_matrix,
      iterations = count - 1,
      items_removed = length(all_removed_items),
      removed_items = all_removed_items,
      redundant_pairs = redundant_df,

      # NEW
      removal_log = removal_log,

      success = success
    ))
  } else {
    # Return original if reduction failed
    attr(original_embedding, "UVA_count") <- 0

    return(list(
      embedding_matrix = original_embedding,
      iterations = 0,
      items_removed = 0L,
      removed_items = character(0),
      redundant_pairs = redundant_df,
      removal_log = data.frame(
        ID = character(),
        uva_sweep = integer(),
        redundant_with_ID = character(),
        redundant_with_statement = character(),
        wTO = numeric(),
        all_redundant_with_IDs = character(),
        all_redundant_wTO = character(),
        stringsAsFactors = FALSE
      ),
      success = FALSE
    ))
  }
}




#' Select Optimal Embedding and EGA Model Based on NMI
#'
#' @param embedding_matrix A numeric matrix (columns = items). The full (dense)
#'   representation.
#' @param sparse_matrix A numeric matrix (columns = items) giving the sparse
#'   representation, aligned to `embedding_matrix` (same items and column order).
#'   This is computed once on the pre-UVA pool and then subset to the post-UVA
#'   items in the AI-GENIE pipeline; passing it in (rather than recomputing
#'   inside) preserves the pre-UVA quantile thresholds.
#' @param true_communities A named list of known communities.
#' @param model Character. One of "glasso", "TMFG", or NULL (to test both).
#' @param algorithm Community detection algorithm (e.g., "walktrap").
#' @param uni.method Unidimensionality method (e.g., "louvain").
#' @param corr Character. Correlation method. Default "auto" uses EGAnet's automatic detection.
#'
#' @details
#' Full embeddings are evaluated before sparse embeddings. Therefore, exact
#' within-model NMI ties retain the full representation. When `model = NULL`,
#' exact cross-model NMI ties prefer TMFG.
#'
#' @return A list with best embedding, model, communities, NMI, and comparison log.
select_optimal_embedding <- function(embedding_matrix,
                                     sparse_matrix,
                                     true_communities,
                                     model = NULL,
                                     algorithm = "walktrap",
                                     uni.method = "louvain",
                                     corr = "auto") {

  # Prepare embeddings
  embeddings <- list(full = embedding_matrix, sparse = sparse_matrix)

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
            (this_nmi == best_nmi && m == "TMFG" && best_result$model != "TMFG"))
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
#' @param boot.iter Numeric. Number of bootstrap iterations. Default 500.
#' @param EGA.type Type of EGA (default "EGA.fit").
#' @param silently Logical. Suppress output.
#'
#' @return A list containing the final embedding, initial/final bootEGA objects,
#'   and an `items_removed` data frame. For each removed item, the table retains
#'   the bootstrap run, empirical item stability, cutoff, stability deficit, and
#'   removal reason. Zero-removal runs return an empty data frame, not `NULL`.
iterative_stability_check <- function(embedding_matrix,
                                      items,
                                      cut.off = 0.75,
                                      model = "NULL",
                                      algorithm = "",
                                      uni.method,
                                      corr = "auto",
                                      ncores = NULL,
                                      boot.iter = 500,
                                      EGA.type = "EGA.fit",
                                      silently) {

  successful <- TRUE
  count <- 1
  current_embedding <- embedding_matrix
  # Stable empty schema so zero-removal runs return a data frame rather than NULL.
  all_removed <- items[0, , drop = FALSE]
  all_removed$boot_run_removed <- integer(0)
  all_removed$item_stability <- numeric(0)
  all_removed$stability_cutoff <- numeric(0)
  all_removed$stability_deficit <- numeric(0)
  all_removed$boot_removal_reason <- character(0)

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
      items_removed = all_removed,
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

    # Treat missing item-stability estimates as explicit removals so every
    # filtered item has an auditable reason.
    na_check <- is.na(emp_dims)
    if (any(na_check, na.rm = TRUE)) {
      na_ids <- colnames(current_embedding)[na_check]
      na_removed <- items[
        as.character(items$ID) %in% na_ids,
        ,
        drop = FALSE
      ]
      na_removed$boot_run_removed <- count
      na_removed$item_stability <- NA_real_
      na_removed$stability_cutoff <- cut.off
      na_removed$stability_deficit <- NA_real_
      na_removed$boot_removal_reason <- "missing_stability"
      all_removed <- rbind(all_removed, na_removed)

      valid_idx <- which(!na_check)
      if (length(valid_idx) == 0L) {
        warning("All items have NA stability. Returning current results.")
        current_embedding <- current_embedding[, FALSE, drop = FALSE]
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
    # Map empirical stability to item ID BEFORE changing the matrix
    stability_by_id <- stats::setNames(
      as.numeric(emp_dims),
      colnames(current_embedding)
    )

    removed_df <- items[
      as.character(items$ID) %in% unstable_ids,
      ,
      drop = FALSE
    ]

    removed_df$boot_run_removed <- count

    removed_df$item_stability <- unname(
      stability_by_id[
        as.character(removed_df$ID)
      ]
    )

    removed_df$stability_cutoff <- cut.off

    removed_df$stability_deficit <-
      cut.off - removed_df$item_stability
    removed_df$boot_removal_reason <- "below_cutoff"

    all_removed <- rbind(
      all_removed,
      removed_df
    )

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
    items_removed = all_removed,
    successful = successful
  ))
}




#' Run bootstrapped EGA on the initial set of items
#'
#' Computes a pre-reduction bootEGA baseline for stability plots using the same
#' EGA settings and bootstrap count as the reduction pipeline.
#'
#' @param result The running results object for one item type.
#' @param data Numeric embedding matrix used for the pre-reduction stability fit.
#' @param EGA.algorithm Community detection algorithm.
#' @param EGA.uni.method Unidimensionality method.
#' @param corr Character. Correlation method. Default "auto".
#' @param ncores Numeric or NULL. Number of cores for parallel processing.
#' @param boot.iter Numeric. Number of bootstrap iterations. Default 500.
#' @param silently Logical. Whether to suppress progress output.
#' @param EGA.type Type of EGA passed to `EGAnet::bootEGA`. Default "EGA.fit".
#'
#' @return A list with `successful` and the updated `result`.
#' @keywords internal
calc_final_stability <- function(result,
                                 data,
                                 EGA.algorithm,
                                 EGA.uni.method,
                                 corr = "auto",
                                 ncores = NULL,
                                 boot.iter = 500,
                                 silently,
                                 EGA.type = "EGA.fit") {

  if (!silently) {
    cat("\n")
    cat("Finding network stability of the original item pool...\n")
  }

  x <- result

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

  if (!is.null(ncores)) {
    boot_args$ncores <- ncores
  }

  boot_obj <- tryCatch(
    do.call(EGAnet::bootEGA, boot_args),
    error = function(e) {
      warning(
        "Stability check failed: ",
        conditionMessage(e),
        ". Returning partial results."
      )
      NULL
    }
  )

  if (is.null(boot_obj)) {
    return(list(successful = FALSE, result = x))
  }

  x$bootEGA$initial_boot_with_redundancies <- boot_obj

  if (!silently) {
    cat("Done.")
  }

  list(successful = TRUE, result = x)
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


#' Plot Stability Comparison (network + item stability dotplot, side by side)
#'
#' Builds a 4-panel comparison: pre-reduction network + pre-reduction item stability,
#' next to post-reduction network + post-reduction item stability. Mirrors the layout
#' of the AIGENIE simulation/reference figure.
#'
#' @param boot1,boot2 bootEGA objects (pre and post reduction).
#' @param caption1,caption2 Captions under each network panel.
#' @param nmi1,nmi2 NMI values pre/post.
#' @param title Overall title.
#'
#' @return A patchwork object combining the four panels.
plot_stability_comparison <- function(boot1, boot2,
                                      caption1, caption2,
                                      nmi1, nmi2, title){

  net1 <- plot(boot1$EGA) +
    ggplot2::labs(caption = paste0(caption1, " (NMI: ", round(nmi1, 4) * 100, ")"))
  net2 <- plot(boot2$EGA) +
    ggplot2::labs(caption = paste0(caption2, " (NMI: ", round(nmi2, 4) * 100, ")"))

  is1 <- boot1$stability$item.stability$plot
  is2 <- boot2$stability$item.stability$plot

  combined <- patchwork::wrap_plots(
    net1, is1, net2, is2,
    ncol = 4,
    widths = c(1.3, 1, 1.3, 1)
  ) +
    patchwork::plot_annotation(
      title = title,
      subtitle = paste0("Change in NMI: ", round((nmi2 - nmi1), 4) * 100),
      theme = ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, size = 16),
        plot.subtitle = ggplot2::element_text(hjust = 0.5, size = 12)
      )
    )

  return(combined)
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
#' @param items A data frame containing the items either generated by AI or supplied by the user.
#'
#' @return A data frame whose `attribute` and `type` columns are modified so the
#'   entire sample can be analyzed as one item type.
#' @keywords internal
run_all_together <- function(items) {

  temp <- paste(items$type, items$attribute)
  items$attribute <- temp
  items$type <- rep("All", nrow(items))

  items
}


#' Compute pre-reduction item-level network-loading diagnostics
#'
#' Uses `EGAnet::net.loads()` on an EGA solution and reports, for each item, the
#' loading on its assigned EGA community, its strongest loading on another
#' community, and the absolute primary-to-cross-loading gap. These statistics
#' are descriptive audit information; they are not item-removal criteria.
#'
#' @param ega_object An `EGAnet::EGA.fit` object or a standard EGA object.
#' @param items Data frame containing at least `ID`.
#'
#' @return A data frame with one row per item represented in the EGA network.
#' @keywords internal
network_loading_diagnostics <- function(ega_object, items) {

  empty <- data.frame(
    ID = character(),
    pre_reduction_EGA_community = integer(),
    pre_reduction_primary_network_loading = numeric(),
    pre_reduction_primary_network_loading_abs = numeric(),
    pre_reduction_strongest_cross_community = character(),
    pre_reduction_strongest_cross_loading = numeric(),
    pre_reduction_strongest_cross_loading_abs = numeric(),
    pre_reduction_loading_gap = numeric(),
    stringsAsFactors = FALSE
  )

  if (is.null(ega_object)) {
    return(empty)
  }

  ega <- if (!is.null(ega_object$EGA)) ega_object$EGA else ega_object

  if (is.null(ega$network) || is.null(ega$wc)) {
    return(empty)
  }

  network <- ega$network
  wc <- ega$wc
  ids <- colnames(network)

  if (is.null(ids)) {
    ids <- rownames(network)
  }

  if (is.null(ids)) {
    return(empty)
  }

  netloads <- tryCatch(
    EGAnet::net.loads(network, wc = wc),
    error = function(e) NULL
  )

  if (is.null(netloads) || is.null(netloads$std)) {
    return(empty)
  }

  L <- as.matrix(netloads$std)

  # Align loading rows to the network item IDs.
  if (!is.null(rownames(L))) {
    keep <- ids %in% rownames(L)
    ids <- ids[keep]
    if (length(ids) == 0L) return(empty)
    L <- L[ids, , drop = FALSE]
  } else {
    if (nrow(L) != length(ids)) return(empty)
    rownames(L) <- ids
  }

  # Align community memberships to the same IDs.
  if (!is.null(names(wc))) {
    wc <- wc[ids]
  } else {
    if (length(wc) != nrow(L)) return(empty)
    names(wc) <- ids
  }

  get_loading_column <- function(community) {
    if (is.na(community)) return(NA_integer_)

    if (!is.null(colnames(L)) && as.character(community) %in% colnames(L)) {
      return(match(as.character(community), colnames(L)))
    }

    idx <- suppressWarnings(as.integer(community))
    if (is.na(idx) || idx < 1L || idx > ncol(L)) NA_integer_ else idx
  }

  out <- lapply(seq_along(ids), function(i) {
    community <- wc[i]
    primary_col <- get_loading_column(community)

    if (is.na(primary_col)) return(NULL)

    primary <- as.numeric(L[i, primary_col])
    other_cols <- setdiff(seq_len(ncol(L)), primary_col)

    if (length(other_cols) > 0L) {
      vals <- abs(L[i, other_cols])
      if (all(is.na(vals))) {
        cross_col <- NA_integer_
      } else {
        cross_col <- other_cols[which.max(replace(vals, is.na(vals), -Inf))]
      }
    } else {
      cross_col <- NA_integer_
    }

    if (is.na(cross_col)) {
      cross <- NA_real_
      cross_community <- NA_character_
    } else {
      cross <- as.numeric(L[i, cross_col])
      cross_community <- if (!is.null(colnames(L))) {
        as.character(colnames(L)[cross_col])
      } else {
        as.character(cross_col)
      }
    }

    data.frame(
      ID = as.character(ids[i]),
      pre_reduction_EGA_community = suppressWarnings(as.integer(community)),
      pre_reduction_primary_network_loading = primary,
      pre_reduction_primary_network_loading_abs = abs(primary),
      pre_reduction_strongest_cross_community = cross_community,
      pre_reduction_strongest_cross_loading = cross,
      pre_reduction_strongest_cross_loading_abs = abs(cross),
      pre_reduction_loading_gap = if (is.na(cross)) NA_real_ else abs(primary) - abs(cross),
      stringsAsFactors = FALSE
    )
  })

  out <- Filter(Negate(is.null), out)
  if (length(out) == 0L) return(empty)

  out <- do.call(rbind, out)
  rownames(out) <- NULL
  out
}


#' Build GENIE item-filtering audit table
#'
#' Combines the stage-specific evidence that caused item removal with
#' pre-reduction network-loading diagnostics. UVA rows report wTO redundancy
#' evidence and the retained redundant partner. bootEGA rows report empirical
#' item stability. Network loadings are descriptive context and are not used as
#' filtering thresholds.
#'
#' @param items Data frame for one item type with `ID`, `statement`, and `attribute`.
#' @param type_name Character label for the item type.
#' @param uva_log Item-level removal log returned by `reduce_redundancy_uva()`.
#' @param boot_removed Data frame of removals returned by `iterative_stability_check()`.
#' @param initial_ega Pre-reduction EGA object computed on the full dense embeddings.
#' @param selection_dropped Character vector of items left unassigned during embedding/model selection.
#' @param final_dropped Character vector of items left unassigned by the final EGA.
#' @param uva.cut.off Numeric wTO cutoff used by UVA.
#' @param stability.cut.off Numeric item-stability cutoff used by bootEGA.
#'
#' @return A tidy data frame with one row per filtered item and the evidence for
#'   its removal.
#' @keywords internal
build_filtering_audit <- function(items,
                                  type_name,
                                  uva_log,
                                  boot_removed,
                                  initial_ega,
                                  uva.cut.off,
                                  stability.cut.off = 0.75,
                                  selection_dropped = character(0),
                                  final_dropped = character(0)) {

  item_ids <- as.character(items$ID)

  uva_rows <- NULL
  if (!is.null(uva_log) && is.data.frame(uva_log) && nrow(uva_log) > 0L) {
    uva_rows <- data.frame(
      ID = as.character(uva_log$ID),
      type = type_name,
      attribute = as.character(items$attribute[match(as.character(uva_log$ID), item_ids)]),
      statement = as.character(items$statement[match(as.character(uva_log$ID), item_ids)]),
      removal_stage = "UVA",
      reason = sprintf("Redundancy: wTO = %.3f >= %.2f", uva_log$wTO, uva.cut.off),
      diagnostic_name = "wTO",
      diagnostic_value = as.numeric(uva_log$wTO),
      cutoff = as.numeric(uva.cut.off),
      uva_sweep = as.integer(uva_log$uva_sweep),
      redundant_with_ID = as.character(uva_log$redundant_with_ID),
      redundant_with_statement = as.character(uva_log$redundant_with_statement),
      redundant_wTO = as.numeric(uva_log$wTO),
      all_redundant_with_IDs = as.character(uva_log$all_redundant_with_IDs),
      all_redundant_wTO = as.character(uva_log$all_redundant_wTO),
      boot_run = NA_integer_,
      item_stability = NA_real_,
      stability_deficit = NA_real_,
      stringsAsFactors = FALSE
    )
  }

  boot_rows <- NULL
  if (!is.null(boot_removed) && is.data.frame(boot_removed) && nrow(boot_removed) > 0L) {
    stability <- if ("item_stability" %in% names(boot_removed)) {
      as.numeric(boot_removed$item_stability)
    } else {
      rep(NA_real_, nrow(boot_removed))
    }

    boot_cutoff <- if ("stability_cutoff" %in% names(boot_removed)) {
      as.numeric(boot_removed$stability_cutoff)
    } else {
      rep(as.numeric(stability.cut.off), nrow(boot_removed))
    }

    deficit <- if ("stability_deficit" %in% names(boot_removed)) {
      as.numeric(boot_removed$stability_deficit)
    } else {
      boot_cutoff - stability
    }

    boot_kind <- if ("boot_removal_reason" %in% names(boot_removed)) {
      as.character(boot_removed$boot_removal_reason)
    } else {
      rep("below_cutoff", nrow(boot_removed))
    }

    boot_reason <- ifelse(
      boot_kind == "missing_stability",
      "Instability: item stability could not be estimated (NA)",
      sprintf("Instability: item stability = %.3f < %.2f", stability, boot_cutoff)
    )

    boot_rows <- data.frame(
      ID = as.character(boot_removed$ID),
      type = type_name,
      attribute = as.character(boot_removed$attribute),
      statement = as.character(boot_removed$statement),
      removal_stage = "bootEGA",
      reason = boot_reason,
      diagnostic_name = "item_stability",
      diagnostic_value = stability,
      cutoff = boot_cutoff,
      uva_sweep = NA_integer_,
      redundant_with_ID = NA_character_,
      redundant_with_statement = NA_character_,
      redundant_wTO = NA_real_,
      all_redundant_with_IDs = NA_character_,
      all_redundant_wTO = NA_character_,
      boot_run = as.integer(boot_removed$boot_run_removed),
      item_stability = stability,
      stability_deficit = deficit,
      stringsAsFactors = FALSE
    )
  }

  make_unassigned_rows <- function(ids, stage, reason_text) {
    ids <- unique(as.character(ids))
    ids <- ids[nzchar(ids) & !is.na(ids)]
    if (length(ids) == 0L) return(NULL)

    data.frame(
      ID = ids,
      type = type_name,
      attribute = as.character(items$attribute[match(ids, item_ids)]),
      statement = as.character(items$statement[match(ids, item_ids)]),
      removal_stage = stage,
      reason = reason_text,
      diagnostic_name = "EGA_community",
      diagnostic_value = NA_real_,
      cutoff = NA_real_,
      uva_sweep = NA_integer_,
      redundant_with_ID = NA_character_,
      redundant_with_statement = NA_character_,
      redundant_wTO = NA_real_,
      all_redundant_with_IDs = NA_character_,
      all_redundant_wTO = NA_character_,
      boot_run = NA_integer_,
      item_stability = NA_real_,
      stability_deficit = NA_real_,
      stringsAsFactors = FALSE
    )
  }

  selection_rows <- make_unassigned_rows(
    selection_dropped,
    "EGA_selection",
    "Unassigned: no EGA community returned during embedding/model selection"
  )

  final_rows <- make_unassigned_rows(
    final_dropped,
    "final_EGA",
    "Unassigned: no EGA community returned in the final structural solution"
  )

  rows <- Filter(
    Negate(is.null),
    list(uva_rows, selection_rows, boot_rows, final_rows)
  )
  if (length(rows) == 0L) {
    return(data.frame())
  }

  audit <- do.call(rbind, rows)

  # Add network-loadings from the full pre-reduction EGA. These statistics are
  # explanatory diagnostics and should not be interpreted as removal rules.
  net <- network_loading_diagnostics(initial_ega, items)

  audit$.audit_order <- seq_len(nrow(audit))
  audit <- merge(audit, net, by = "ID", all.x = TRUE, sort = FALSE)
  audit <- audit[order(audit$.audit_order), , drop = FALSE]
  audit$.audit_order <- NULL
  rownames(audit) <- NULL

  audit
}


#' Combine filtering audits across item types
#'
#' @param item_type_level Named list of item-type pipeline results.
#'
#' @return A single data frame containing all available item-level filtering
#'   audit rows across item types.
#' @keywords internal
combine_filtering_audits <- function(item_type_level) {

  audits <- lapply(item_type_level, function(x) {
    if (is.null(x) ||
        is.null(x$filtering_audit) ||
        !is.data.frame(x$filtering_audit) ||
        nrow(x$filtering_audit) == 0L) {
      return(NULL)
    }
    x$filtering_audit
  })

  audits <- Filter(Negate(is.null), audits)
  if (length(audits) == 0L) return(data.frame())

  out <- do.call(rbind, audits)
  rownames(out) <- NULL
  out
}


#' Build the final GENIE/AIGENIE return object
#'
#' @param item_type_level Named list containing results at the item-type level.
#' @param overall_result Named list containing results at the overall level when
#'   `run.overall = TRUE`.
#' @param run.overall Logical. Whether an overall post-reduction fit was run.
#' @param keep.org Logical. Retained for compatibility with callers; original
#'   items are already handled inside each pipeline result.
#'
#' @return A named list containing `item_type_level` and a combined
#'   `filtering_audit`; when `run.overall = TRUE`, also includes `overall`.
#' @keywords internal
build_return <- function(item_type_level,
                         overall_result,
                         run.overall,
                         keep.org) {

  filtering_audit <- combine_filtering_audits(item_type_level)

  # When an overall post-reduction fit was requested, prefer its pooled
  # pre-reduction network-loading diagnostics in the top-level audit.
  if (isTRUE(run.overall) &&
      !is.null(overall_result) &&
      is.data.frame(overall_result$filtering_audit) &&
      nrow(overall_result$filtering_audit) > 0L) {
    filtering_audit <- overall_result$filtering_audit
  }

  out <- list(
    item_type_level = item_type_level,
    filtering_audit = filtering_audit
  )

  if (isTRUE(run.overall)) {
    out$overall <- overall_result
  }

  out
}
