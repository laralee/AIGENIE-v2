# Run full pipeline for a single item type

Run full pipeline for a single item type

## Usage

``` r
run_pipeline_for_item_type(
  embedding_matrix,
  items,
  type_name,
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

- embedding_matrix:

  Numeric matrix (columns = items for one type)

- items:

  Data frame of items for this type (must include ID, statement,
  attribute)

- type_name:

  Character. Type label used for tracking/logging.

- model:

  NULL, "glasso", or "TMFG"

- algorithm:

  EGA algorithm

- uni.method:

  EGA uni.method

- corr:

  Character. Correlation method. Default "auto" uses EGAnet's automatic
  detection.

- ncores:

  Numeric. Number of cores for parallel processing. Default NULL uses
  EGAnet default.

- boot.iter:

  Numeric. Number of bootstrap iterations. Default 500.

- uva.cut.off:

  Numeric in `[0, 1)`. wTO threshold for
  [`EGAnet::UVA`](https://rdrr.io/pkg/EGAnet/man/UVA.html). Default
  `0.20`.

- keep.org:

  Logical. Whether to include original items and embeddings

- silently:

  Logical. Whether to print progress statements

- plot:

  Logical. Whether to plot the network plots at the end

## Value

A named list containing pipeline results for this type, including a
`filtering_audit` table with one row per removed item and a
`reduction_summary` table describing NMI and item-count changes by
stage.
