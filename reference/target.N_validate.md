# Validate and Expand `target.N` for Each Item Attribute

Ensures that `target.N` is either:

- NULL -\> defaults to 60 per attribute

- A single integer -\> repeated for each attribute

- A list/vector of integers -\> must match number of attributes

## Usage

``` r
target.N_validate(
  target.N,
  items.attributes,
  items.only,
  embeddings.only,
  silently
)
```

## Arguments

- target.N:

  An integer, list/vector of integers, or NULL.

- items.attributes:

  A cleaned list returned from `validate_items.attributes()`.

- items.only:

  A flag used to determine if only items need to be generated

- embeddings.only:

  A flag used to determine if only embeddings need to be generated

- silently:

  A flag used to determine if warnings should be printed

## Value

A list of integers, one per attribute (named).
