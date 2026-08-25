# Validate and Clean `response.options`

Validates that `response.options` is an atomic vector of non-empty
strings, with no missing or invalid values. Whitespace is trimmed from
each string.

## Usage

``` r
response.options_validate(response.options)
```

## Arguments

- response.options:

  An atomic character vector of response labels.
