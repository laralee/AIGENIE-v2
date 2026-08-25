# Run reduction pipeline for all item types

Run reduction pipeline for all item types

## Usage

``` r
run_item_reduction_pipeline(
  embedding_matrix,
  items,
  EGA.model = NULL,
  EGA.algorithm = "walktrap",
  EGA.uni.method = "louvain",
  corr = "auto",
  ncores = NULL,
  boot.iter = 500,
  uva.cut.off = 0.2,
  keep.org,
  silently,
  plot
)
```

## Arguments

- embedding_matrix:

  Full embedding matrix (columns = all items)

- items:

  Data frame of all items (must include ID, statement, attribute, type)

- EGA.model:

  NULL, "glasso", or "TMFG"

- EGA.algorithm:

  EGA algorithm

- EGA.uni.method:

  EGA uni.method

- corr:

  Character. Correlation method. Default "auto" uses EGAnet's automatic
  detection.

- ncores:

  Numeric. Number of cores for parallel processing.

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

A named list of pipeline results, one per item type
