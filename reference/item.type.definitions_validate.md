# Validate and Clean `item.type.definitions`

Validates that `item.type.definitions` is a named list where:

- Names are unique (after trim + case-fold)

- Names exist in `items.attributes`

- Values are non-empty strings

## Usage

``` r
item.type.definitions_validate(item.type.definitions, items.attributes)
```

## Arguments

- item.type.definitions:

  A named list of strings, where each name must correspond to a name in
  `items.attributes` and each value must be a non-empty string.

- items.attributes:

  A cleaned list from `validate_items.attributes()`.

## Value

A cleaned version of `item.type.definitions`.

## Details

Returns a cleaned version with:

- Normalized names (trimmed and lowercased)

- Trimmed values (case preserved)
