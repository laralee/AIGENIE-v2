# Validate Items Data Frame for GENIE

Validates that the items data frame meets all requirements for GENIE
processing. Ensures proper structure, column presence, data types, and
content validity.

## Usage

``` r
items_validate_GENIE(items)
```

## Arguments

- items:

  A data frame that should contain columns: statement, attribute, type,
  ID

## Value

A cleaned and validated items data frame with standardized formatting
