#' Validate Items Data Frame for GENIE
#'
#' @description
#' Validates that the items data frame meets all requirements for GENIE processing.
#' Ensures proper structure, column presence, data types, and content validity.
#'
#' @param items A data frame that should contain columns: statement, attribute, type, ID
#'
#' @return A cleaned and validated items data frame with standardized formatting
#'
items_validate_GENIE <- function(items) {

  # Helper function for normalizing strings
  norm_str <- function(x) trimws(tolower(x))
  trim_str <- function(x) trimws(x)

  # ---- 1. Check that items is a data frame ----
  if (!is.data.frame(items)) {
    stop(
      "GENIE expects 'items' to be a data frame with columns: 'statement', 'attribute', 'type', 'ID'.",
      call. = FALSE
    )
  }

  # ---- 2. Check for required columns ----
  required_cols <- c("statement", "attribute", "type", "ID")
  missing_cols <- setdiff(required_cols, names(items))

  if (length(missing_cols) > 0) {
    stop(
      paste0(
        "GENIE expects 'items' to contain the following columns: ",
        paste(sprintf("'%s'", required_cols), collapse = ", "),
        ".\nMissing columns: ",
        paste(sprintf("'%s'", missing_cols), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---- 3. Check that data frame is not empty ----
  if (nrow(items) == 0) {
    stop(
      "GENIE expects 'items' to contain at least one row of data.",
      call. = FALSE
    )
  }

  # ---- 4. Validate statement column ----
  if (!is.character(items$statement)) {
    items$statement <- as.character(items$statement)
  }

  if (any(is.na(items$statement))) {
    stop(
      "GENIE expects items$statement to contain no NA values.",
      call. = FALSE
    )
  }

  statements_trimmed <- trim_str(items$statement)
  if (any(statements_trimmed == "")) {
    empty_rows <- which(statements_trimmed == "")
    stop(
      paste0(
        "GENIE expects items$statement to contain no empty strings.\n",
        "Empty statements found at row(s): ",
        paste(empty_rows, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---- 5. Validate attribute column ----
  if (!is.character(items$attribute)) {
    items$attribute <- as.character(items$attribute)
  }

  if (any(is.na(items$attribute))) {
    stop(
      "GENIE expects items$attribute to contain no NA values.",
      call. = FALSE
    )
  }

  attributes_trimmed <- trim_str(items$attribute)
  if (any(attributes_trimmed == "")) {
    empty_rows <- which(attributes_trimmed == "")
    stop(
      paste0(
        "GENIE expects items$attribute to contain no empty strings.\n",
        "Empty attributes found at row(s): ",
        paste(empty_rows, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---- 6. Validate type column ----
  if (!is.character(items$type)) {
    items$type <- as.character(items$type)
  }

  if (any(is.na(items$type))) {
    stop(
      "GENIE expects items$type to contain no NA values.",
      call. = FALSE
    )
  }

  types_trimmed <- trim_str(items$type)
  if (any(types_trimmed == "")) {
    empty_rows <- which(types_trimmed == "")
    stop(
      paste0(
        "GENIE expects items$type to contain no empty strings.\n",
        "Empty types found at row(s): ",
        paste(empty_rows, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---- 7. Validate ID column ----
  # ID can be numeric or character
  if (is.factor(items$ID)) {
    items$ID <- as.character(items$ID)
  }

  if (any(is.na(items$ID))) {
    stop(
      "GENIE expects items$ID to contain no NA values.",
      call. = FALSE
    )
  }

  # Check for unique IDs
  if (any(duplicated(items$ID))) {
    dup_ids <- unique(items$ID[duplicated(items$ID)])
    stop(
      paste0(
        "GENIE expects items$ID to contain unique values.\n",
        "Duplicate IDs found: ",
        paste(sprintf("'%s'", dup_ids[1:min(10, length(dup_ids))]), collapse = ", "),
        ifelse(length(dup_ids) > 10, paste0(" ... and ", length(dup_ids) - 10, " more"), "")
      ),
      call. = FALSE
    )
  }

  # Convert ID to character for consistency
  items$ID <- as.character(items$ID)

  # Check for empty IDs after conversion
  ids_trimmed <- trim_str(items$ID)
  if (any(ids_trimmed == "")) {
    empty_rows <- which(ids_trimmed == "")
    stop(
      paste0(
        "GENIE expects items$ID to contain no empty strings.\n",
        "Empty IDs found at row(s): ",
        paste(empty_rows, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---- 8. Create cleaned data frame ----
  cleaned_items <- data.frame(
    statement = statements_trimmed,
    attribute = norm_str(items$attribute),  # Normalize attribute (lowercase + trim)
    type = norm_str(items$type),            # Normalize type (lowercase + trim)
    ID = ids_trimmed,                       # Just trim IDs, preserve case
    stringsAsFactors = FALSE
  )

  # ---- 9. Check for minimum items per type-attribute combination ----
  # This ensures meaningful analysis can be performed
  type_attr_counts <- table(cleaned_items$type, cleaned_items$attribute)

  # Find combinations with too few items
  low_counts <- which(type_attr_counts > 0 & type_attr_counts < 15, arr.ind = TRUE)

  if (nrow(low_counts) > 0) {
    warning(
      paste0(
        "GENIE detected type-attribute combinations with fewer than 15 items.\n",
        "Consider adding more items for meaningful analysis:\n",
        paste(
          sprintf("  - Type '%s', Attribute '%s': %d item(s)",
                  rownames(type_attr_counts)[low_counts[,1]],
                  colnames(type_attr_counts)[low_counts[,2]],
                  type_attr_counts[low_counts]),
          collapse = "\n"
        )
      ),
      call. = FALSE,
      immediate. = TRUE
    )
  }

  return(cleaned_items)
}



#' Validate Embedding Matrix for GENIE
#'
#' @description
#' Validates that the optional embedding matrix meets all requirements for GENIE processing.
#' Ensures proper structure, dimensions, column names match item IDs, and numeric content.
#'
#' @param embedding.matrix A numeric matrix or data frame with rows as embedding dimensions
#'   and columns as items. Can be NULL if embeddings will be generated.
#' @param items A validated items data frame (already processed by items_validate_GENIE)
#' @param silently Logical. If FALSE, displays informational messages
#'
#' @return A validated embedding matrix (always as matrix type) or NULL if not provided
#'
embedding_matrix_validate_GENIE <- function(embedding.matrix, items, silently = FALSE) {

  # ---- 1. Handle NULL case (embeddings will be generated) ----
  if (is.null(embedding.matrix)) {
    return(NULL)
  }

  # ---- 2. Convert data frame to matrix if needed ----
  if (is.data.frame(embedding.matrix)) {
    if (!silently) {
      message("Converting embedding data frame to matrix format...")
    }

    # Check that all columns are numeric
    non_numeric <- sapply(embedding.matrix, function(x) !is.numeric(x))
    if (any(non_numeric)) {
      stop(
        paste0(
          "GENIE expects all columns in embedding.matrix to be numeric.\n",
          "Non-numeric columns found: ",
          paste(sprintf("'%s'", names(embedding.matrix)[non_numeric]), collapse = ", ")
        ),
        call. = FALSE
      )
    }

    embedding.matrix <- as.matrix(embedding.matrix)
  }

  # ---- 3. Check that it's a matrix ----
  if (!is.matrix(embedding.matrix)) {
    stop(
      "GENIE expects embedding.matrix to be a numeric matrix or data frame.",
      call. = FALSE
    )
  }

  # ---- 4. Check that it's numeric ----
  if (!is.numeric(embedding.matrix)) {
    stop(
      "GENIE expects embedding.matrix to contain only numeric values.",
      call. = FALSE
    )
  }

  # ---- 5. Check for NAs or infinite values ----
  if (any(is.na(embedding.matrix))) {
    na_count <- sum(is.na(embedding.matrix))
    stop(
      paste0(
        "GENIE expects embedding.matrix to contain no NA values.\n",
        "Found ", na_count, " NA value(s)."
      ),
      call. = FALSE
    )
  }

  if (any(is.infinite(embedding.matrix))) {
    inf_count <- sum(is.infinite(embedding.matrix))
    stop(
      paste0(
        "GENIE expects embedding.matrix to contain no infinite values.\n",
        "Found ", inf_count, " infinite value(s)."
      ),
      call. = FALSE
    )
  }

  # ---- 6. Check dimensions ----
  if (nrow(embedding.matrix) == 0 || ncol(embedding.matrix) == 0) {
    stop(
      paste0(
        "GENIE expects embedding.matrix to have at least one row and one column.\n",
        "Current dimensions: ", nrow(embedding.matrix), " rows × ", ncol(embedding.matrix), " columns"
      ),
      call. = FALSE
    )
  }

  # ---- 7. Check column names exist ----
  if (is.null(colnames(embedding.matrix))) {
    stop(
      "GENIE expects embedding.matrix to have column names that match items$ID.",
      call. = FALSE
    )
  }

  # ---- 8. Check column names match item IDs ----
  matrix_ids <- colnames(embedding.matrix)
  item_ids <- as.character(items$ID)  # Ensure character comparison

  # Find mismatches
  missing_from_matrix <- setdiff(item_ids, matrix_ids)
  extra_in_matrix <- setdiff(matrix_ids, item_ids)

  if (length(missing_from_matrix) > 0) {
    stop(
      paste0(
        "GENIE expects embedding.matrix to have columns for all items.\n",
        "Missing embeddings for ", length(missing_from_matrix), " item(s):\n",
        paste(sprintf("  - '%s'", head(missing_from_matrix, 10)), collapse = "\n"),
        ifelse(length(missing_from_matrix) > 10,
               paste0("\n  ... and ", length(missing_from_matrix) - 10, " more"), "")
      ),
      call. = FALSE
    )
  }

  if (length(extra_in_matrix) > 0) {
    if (!silently) {
      warning(
        paste0(
          "embedding.matrix contains ", length(extra_in_matrix),
          " column(s) not present in items. These will be removed:\n",
          paste(sprintf("  - '%s'", head(extra_in_matrix, 10)), collapse = "\n"),
          ifelse(length(extra_in_matrix) > 10,
                 paste0("\n  ... and ", length(extra_in_matrix) - 10, " more"), "")
        ),
        call. = FALSE,
        immediate. = TRUE
      )
    }

    # Remove extra columns
    embedding.matrix <- embedding.matrix[, matrix_ids %in% item_ids, drop = FALSE]
  }

  # ---- 9. Reorder columns to match items order ----
  embedding.matrix <- embedding.matrix[, item_ids, drop = FALSE]

  # ---- 10. Check embedding dimensions are reasonable ----
  n_dims <- nrow(embedding.matrix)
  n_items <- ncol(embedding.matrix)

  if (n_dims < 2) {
    stop(
      paste0(
        "GENIE expects embedding.matrix to have at least 2 dimensions (rows).\n",
        "Current dimensions: ", n_dims
      ),
      call. = FALSE
    )
  }


  # ---- 11. Check for zero variance columns ----
  col_vars <- apply(embedding.matrix, 2, var)
  zero_var_cols <- which(col_vars == 0)

  if (length(zero_var_cols) > 0) {
    warning(
      paste0(
        "GENIE detected ", length(zero_var_cols), " item(s) with zero variance in embeddings.\n",
        "Items: ", paste(sprintf("'%s'", colnames(embedding.matrix)[zero_var_cols][1:min(5, length(zero_var_cols))]),
                         collapse = ", "),
        ifelse(length(zero_var_cols) > 5, paste0(" ... and ", length(zero_var_cols) - 5, " more"), ""),
        "\nThese items may cause issues in network analysis."
      ),
      call. = FALSE,
      immediate. = TRUE
    )
  }

  # ---- 12. Check for duplicate embeddings ----
  # Use correlation to find nearly identical embeddings
  cor_matrix <- cor(embedding.matrix)
  diag(cor_matrix) <- 0  # Remove self-correlations

  # Find pairs with correlation > 0.999 (nearly identical)
  high_cor_pairs <- which(abs(cor_matrix) > 0.999, arr.ind = TRUE)

  if (nrow(high_cor_pairs) > 0) {
    # Get unique pairs (remove duplicates due to symmetry)
    high_cor_pairs <- high_cor_pairs[high_cor_pairs[,1] < high_cor_pairs[,2], , drop = FALSE]

    if (nrow(high_cor_pairs) > 0) {
      warning(
        paste0(
          "GENIE detected ", nrow(high_cor_pairs), " pair(s) of items with nearly identical embeddings (r > 0.999).\n",
          "Example pairs:\n",
          paste(
            sprintf("  - '%s' and '%s'",
                    colnames(embedding.matrix)[high_cor_pairs[1:min(3, nrow(high_cor_pairs)), 1]],
                    colnames(embedding.matrix)[high_cor_pairs[1:min(3, nrow(high_cor_pairs)), 2]]),
            collapse = "\n"
          ),
          ifelse(nrow(high_cor_pairs) > 3, paste0("\n  ... and ", nrow(high_cor_pairs) - 3, " more pairs"), ""),
          "\nConsider reviewing these items for redundancy."
        ),
        call. = FALSE,
        immediate. = TRUE
      )
    }
  }

  if (!silently) {
    message(paste0(
      "Embedding matrix validated: ",
      n_dims, " dimensions × ",
      n_items, " items"
    ))
  }

  return(embedding.matrix)
}


#' Build item.attributes Object from Items Data Frame
#'
#' @description
#' Reverse engineers the `item.attributes` object structure required by AIGENIE
#' from a validated items data frame. This allows GENIE to work with user-provided
#' items by reconstructing the expected attribute structure.
#'
#' @param items A validated data frame with columns: statement, attribute, type, ID
#'   (already processed by items_validate_GENIE)
#'
#' @return A named list where:
#'   - Names are the unique item types from items$type
#'   - Each element is a character vector of unique attributes for that type
#'   - All values are normalized (lowercase, trimmed) to match AIGENIE expectations
#'
build_item_attributes_from_items <- function(items) {

  # Get unique types (should already be normalized from validation)
  unique_types <- unique(items$type)

  # Initialize the result list
  item_attributes <- list()

  # For each type, get the unique attributes
  for (type in unique_types) {
    # Filter items for this type
    type_items <- items[items$type == type, ]

    # Get unique attributes for this type (should already be normalized)
    unique_attributes <- unique(type_items$attribute)

    # Store in the result
    item_attributes[[type]] <- unique_attributes
  }

  # Ensure the list is properly named
  names(item_attributes) <- unique_types

  return(item_attributes)
}
