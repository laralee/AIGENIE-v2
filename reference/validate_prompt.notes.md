# Validate and Normalize `prompt.notes`

Accepts a string, NULL, or a named list of strings/NULLs. Ensures one
entry per attribute in `items.attributes`, returning a fully named and
cleaned list.

## Usage

``` r
validate_prompt.notes(prompt.notes, items.attributes)
```

## Arguments

- prompt.notes:

  A single string, NULL, or named list of strings/NULLs.

- items.attributes:

  A cleaned list from `validate_items.attributes()`.

## Value

A named list of strings, one per attribute, with NULLs replaced by "".
