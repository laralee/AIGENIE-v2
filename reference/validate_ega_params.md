# Validate EGA Parameters

Validates and normalizes the EGA algorithm, unidimensionality method,
and model parameters. Trims whitespace and performs case-insensitive
matching. Returns canonical-cased values.

## Usage

``` r
validate_ega_params(EGA.algorithm, EGA.uni.method, EGA_model)
```

## Arguments

- EGA.algorithm:

  A string: one of "leiden", "louvain", "walktrap" (or NULL, in which
  case default behavior takes over)

- EGA.uni.method:

  A string: one of "expand", "LE", "louvain"

- EGA_model:

  A string or NULL: one of "glasso", "TMFG"

## Value

A named list with cleaned and correctly-cased values.
