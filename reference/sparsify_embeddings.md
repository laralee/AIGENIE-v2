# Sparsify Embedding Matrix

Applies sparsification to an embedding matrix by zeroing out values
between specified quantiles. Includes fallback strategies if initial
sparsification results in all zeros.

## Usage

``` r
sparsify_embeddings(
  embedding_matrix,
  lower_quantile = 0.025,
  upper_quantile = 0.975,
  fallback_lower = 0.1,
  fallback_upper = 0.9
)
```

## Arguments

- embedding_matrix:

  Numeric matrix with items as columns, dimensions as rows

- lower_quantile:

  Lower quantile threshold (default 0.025)

- upper_quantile:

  Upper quantile threshold (default 0.975)

- fallback_lower:

  Fallback lower quantile if first attempt fails (default 0.10)

- fallback_upper:

  Fallback upper quantile if first attempt fails (default 0.90)

## Value

Sparsified embedding matrix with same dimensions as input

## Details

Sparsification process:

1.  Zero out values between lower and upper quantiles

2.  If result is all zeros, try fallback quantiles

3.  If still all zeros, return original matrix

`silently` is always `TRUE`. It is only set to `FALSE` for developement
and diagnostic purposes.
