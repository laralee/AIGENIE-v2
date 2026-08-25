# Checks `system.role` and `prompts` for the `chat` function

Checks `system.role` and `prompts` for the `chat` function

## Usage

``` r
validate_system.role_prompts(system.role, prompts)
```

## Arguments

- system.role:

  The persona for the model. Either a string, `NULL` (default), or a
  list of strings

- prompts:

  The prompts to be given to the model. Either a string or a list of
  strings

## Value

if valid, a list with the `system.role` and `prompts` objects
