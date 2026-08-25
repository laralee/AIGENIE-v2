# Build item.attributes Object from Items Data Frame

Reverse engineers the `item.attributes` object structure required by
AIGENIE from a validated items data frame. This allows GENIE to work
with user-provided items by reconstructing the expected attribute
structure.

## Usage

``` r
build_item_attributes_from_items(items)
```

## Arguments

- items:

  A validated data frame with columns: statement, attribute, type, ID
  (already processed by items_validate_GENIE)

## Value

A named list where:

- Names are the unique item types from items\$type

- Each element is a character vector of unique attributes for that type

- All values are normalized (lowercase, trimmed) to match AIGENIE
  expectations
