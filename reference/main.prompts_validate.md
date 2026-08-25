# Validate and Normalize `main.prompts`

Validates that `main.prompts` is a named list of non-empty strings, one
for each attribute in `items.attributes`, matched by normalized name.

## Usage

``` r
main.prompts_validate(main.prompts, items.attributes, silently)
```

## Arguments

- main.prompts:

  A named list of prompt strings, one per attribute.

- items.attributes:

  A cleaned list from `validate_items.attributes()`.

- silently:

  A flag determining wheter a warning message should be printed

## Value

A cleaned and ordered named list of trimmed prompt strings. Also returns
the appropriate 'custom' flag (TRUE if custom ok, FALSE if not)
