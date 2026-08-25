# Validate Boolean Arguments

Validates that all arguments passed to the function are scalar boolean
values (`TRUE` or `FALSE`). If any argument is not a boolean, an error
is thrown that identifies the offending variable by name and instructs
the user to set it to either `TRUE` or `FALSE`.

## Usage

``` r
validate_booleans(...)
```

## Arguments

- ...:

  One or more variables to check. We are expecting each to be a logical
  scalar (`TRUE` or `FALSE`). In `AIGENIE`, these variables would be
  `items.only`, `adaptive`, `plot`, `keep.org`, `silently`, and
  `embeddings.only`.
