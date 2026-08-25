# Run a pooled post-reduction fit across all item types

`run.overall = TRUE` is a fit-only analysis: it takes the union of items
that survived the type-level GENIE reductions and evaluates the pooled
structure without applying additional UVA or bootEGA filtering. This is
intentionally distinct from `all.together = TRUE`, which performs
reduction on the entire item pool jointly.

## Usage

``` r
run_pipeline_for_all(
  item_level,
  items,
  embeddings,
  model = NULL,
  algorithm = "walktrap",
  uni.method = "louvain",
  corr = "auto",
  ncores = NULL,
  boot.iter = 500,
  uva.cut.off = 0.2,
  keep.org = FALSE,
  silently,
  plot
)
```

## Arguments

- item_level:

  Named list of completed type-level GENIE results.

- items:

  Original item data frame.

- embeddings:

  Original full embedding matrix (columns = item IDs).

- model:

  NULL, "glasso", or "TMFG". If NULL, the model with the highest pooled
  post-reduction NMI on the full embeddings is selected; exact ties
  prefer TMFG.

- algorithm:

  EGA community detection algorithm.

- uni.method:

  EGA unidimensionality method.

- corr:

  Character. Correlation method. Default "auto".

- ncores:

  Retained for backward compatibility; no additional bootEGA is run in
  the fit-only overall analysis.

- boot.iter:

  Retained for backward compatibility; no additional bootEGA is run in
  the fit-only overall analysis. Default 500.

- uva.cut.off:

  Retained for backward compatibility; no additional UVA is run in the
  fit-only overall analysis.

- keep.org:

  Logical. Whether to retain original items/embeddings.

- silently:

  Logical. Whether to suppress progress output.

- plot:

  Logical. Whether to print the pooled pre/post network comparison.

## Value

A list with `overall_result` and `success`. `overall_result` contains
pooled pre/post EGA fits, NMI values, the union of type-level survivors,
a pooled filtering audit, and a pooled reduction summary.
