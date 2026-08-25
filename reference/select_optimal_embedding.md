# Select Optimal Embedding and EGA Model Based on NMI

Select Optimal Embedding and EGA Model Based on NMI

## Usage

``` r
select_optimal_embedding(
  embedding_matrix,
  sparse_matrix,
  true_communities,
  model = NULL,
  algorithm = "walktrap",
  uni.method = "louvain",
  corr = "auto"
)
```

## Arguments

- embedding_matrix:

  A numeric matrix (columns = items). The full (dense) representation.

- sparse_matrix:

  A numeric matrix (columns = items) giving the sparse representation,
  aligned to `embedding_matrix` (same items and column order). This is
  computed once on the pre-UVA pool and then subset to the post-UVA
  items in the AI-GENIE pipeline; passing it in (rather than recomputing
  inside) preserves the pre-UVA quantile thresholds.

- true_communities:

  A named list of known communities.

- model:

  Character. One of "glasso", "TMFG", or NULL (to test both).

- algorithm:

  Community detection algorithm (e.g., "walktrap").

- uni.method:

  Unidimensionality method (e.g., "louvain").

- corr:

  Character. Correlation method. Default "auto" uses EGAnet's automatic
  detection.

## Value

A list with best embedding, model, communities, NMI, and comparison log.

## Details

Full embeddings are evaluated before sparse embeddings. Therefore, exact
within-model NMI ties retain the full representation. When
`model = NULL`, exact cross-model NMI ties prefer TMFG.
