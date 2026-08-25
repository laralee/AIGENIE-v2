# Validate and Clean `item.examples` Against Cleaned `items.attributes`

Ensures `item.examples` is a data frame with required string columns and
that the values in `type` and `attribute` align with the cleaned
structure of `items.attributes`. Returns a cleaned version of the data
frame with normalized values:

- `type` and `attribute` are trimmed and lowercased

- `statement` is trimmed (case preserved)

## Usage

``` r
item.examples_validate(item.examples, items.attributes)
```

## Arguments

- item.examples:

  A data frame with columns `type`, `attribute`, `statement`. All values
  must be non-empty strings.

- items.attributes:

  A cleaned list from `validate_items.attributes()`. All names and
  values must be normalized (lowercased and trimmed).

## Value

A cleaned version of `item.examples` with normalized values.
