# Reduce Redundancy via Iterative UVA (with Redundant Pair Logging)

Applies EGAnet::UVA iteratively and logs human-readable redundant item
sets.

## Usage

``` r
reduce_redundancy_uva(
  embedding_matrix,
  items,
  corr = "auto",
  uva.cut.off = 0.2
)
```

## Arguments

- embedding_matrix:

  A numeric matrix of embeddings (columns = items).

- items:

  Data frame with `ID` and `statement` columns.

- corr:

  Character. Correlation method to use. Default "auto" uses EGAnet's
  automatic correlation detection. Other options: "pearson", "spearman",
  "cosine".

- uva.cut.off:

  Numeric in `[0, 1)`. The weighted topological overlap threshold passed
  to [`EGAnet::UVA`](https://rdrr.io/pkg/EGAnet/man/UVA.html). Items
  with pairwise wTO at or above this value are flagged as redundant.
  Default `0.20`.

## Value

A list with the reduced matrix, sweep metadata, human-readable
redundancy groups, and `removal_log`, a tidy item-level table containing
removed IDs, retained redundant partners, and wTO statistics.
