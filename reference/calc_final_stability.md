# Run bootstrapped EGA on the initial set of items

Computes a pre-reduction bootEGA baseline for stability plots using the
same EGA settings and bootstrap count as the reduction pipeline.

## Usage

``` r
calc_final_stability(
  result,
  data,
  EGA.algorithm,
  EGA.uni.method,
  corr = "auto",
  ncores = NULL,
  boot.iter = 500,
  silently,
  EGA.type = "EGA.fit"
)
```

## Arguments

- result:

  The running results object for one item type.

- data:

  Numeric embedding matrix used for the pre-reduction stability fit.

- EGA.algorithm:

  Community detection algorithm.

- EGA.uni.method:

  Unidimensionality method.

- corr:

  Character. Correlation method. Default "auto".

- ncores:

  Numeric or NULL. Number of cores for parallel processing.

- boot.iter:

  Numeric. Number of bootstrap iterations. Default 500.

- silently:

  Logical. Whether to suppress progress output.

- EGA.type:

  Type of EGA passed to
  [`EGAnet::bootEGA`](https://rdrr.io/pkg/EGAnet/man/bootEGA.html).
  Default "EGA.fit".

## Value

A list with `successful` and the updated `result`.
