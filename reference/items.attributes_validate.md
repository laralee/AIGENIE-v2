# Validate `items.attributes`

Validates that `items.attributes` is a **named list** whose names are
truly unique after trimming whitespace and ignoring case, and that each
element is itself a list **containing only strings**, with **at least
two** truly unique strings (same trimming + case-insensitive rule).

## Usage

``` r
items.attributes_validate(items.attributes)
```

## Arguments

- items.attributes:

  A named list. Each element must be a list containing only character
  scalars (strings). Each of those inner lists must contain at least two
  truly unique strings after trimming and case-folding.

## Value

A cleaned version of `items.attributes` with normalized names and
values. Errors are thrown if validation fails.
