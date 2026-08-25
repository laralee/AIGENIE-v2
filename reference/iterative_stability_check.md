# Iteratively run BootEGA to ensure structural stability of items

Iteratively run BootEGA to ensure structural stability of items

## Usage

``` r
iterative_stability_check(
  embedding_matrix,
  items,
  cut.off = 0.75,
  model = "NULL",
  algorithm = "",
  uni.method,
  corr = "auto",
  ncores = NULL,
  boot.iter = 500,
  EGA.type = "EGA.fit",
  silently
)
```

## Arguments

- embedding_matrix:

  Numeric matrix of item embeddings (columns = items).

- items:

  Data frame containing at least `ID` and `statement`.

- cut.off:

  Numeric. Minimum stability required to retain an item.

- model:

  Network estimation model (e.g., "glasso", "TMFG").

- algorithm:

  Community detection algorithm.

- uni.method:

  Unidimensionality method.

- corr:

  Character. Correlation method. Default "auto" uses EGAnet's automatic
  detection.

- ncores:

  Numeric. Number of cores for parallel processing. Default NULL uses
  EGAnet default.

- boot.iter:

  Numeric. Number of bootstrap iterations. Default 500.

- EGA.type:

  Type of EGA (default "EGA.fit").

- silently:

  Logical. Suppress output.

## Value

A list containing the final embedding, initial/final bootEGA objects,
and an `items_removed` data frame. For each removed item, the table
retains the bootstrap run, empirical item stability, cutoff, stability
deficit, and removal reason. Zero-removal runs return an empty data
frame, not `NULL`.
