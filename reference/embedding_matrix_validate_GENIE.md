# Validate Embedding Matrix for GENIE

Validates that the optional embedding matrix meets all requirements for
GENIE processing. Ensures proper structure, dimensions, column names
match item IDs, and numeric content.

## Usage

``` r
embedding_matrix_validate_GENIE(embedding.matrix, items, silently = FALSE)
```

## Arguments

- embedding.matrix:

  A numeric matrix or data frame with rows as embedding dimensions and
  columns as items. Can be NULL if embeddings will be generated.

- items:

  A validated items data frame (already processed by
  items_validate_GENIE)

- silently:

  Logical. If FALSE, displays informational messages

## Value

A validated embedding matrix (always as matrix type) or NULL if not
provided
