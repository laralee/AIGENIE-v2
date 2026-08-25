# Modify Main Prompts with Contextual Enhancements

This function appends structured context and formatting instructions to
a list of main prompts used for item generation. It ensures that each
prompt includes relevant domain information, audience guidance, scale
definitions, JSON formatting rules, and optionally example items or
critical author notes – but only if these elements are not already
present in the prompt (checked case-insensitively and with whitespace
trimmed).

## Usage

``` r
modify_main.prompts(
  main.prompts,
  item.attributes,
  item.type.definitions,
  domain,
  scale.title,
  prompt.notes,
  audience,
  item.examples
)
```

## Arguments

- main.prompts:

  A named list of character strings, where each element is a prompt
  associated with an item type.

- item.attributes:

  A named list where each element is a character vector of attribute
  names for an item type.

- item.type.definitions:

  (Optional) A named list of definitions corresponding to each item
  type. Used to append conceptual clarity.

- domain:

  (Optional) A string describing the content domain (e.g.,
  "psychological", "clinical"). Included in the prompt if not already
  present.

- scale.title:

  (Optional) The name of the scale (e.g., "Social Anxiety Scale") for
  which items are being generated.

- prompt.notes:

  (Optional) A named list of author-supplied notes for each item type
  that should be emphasized in the prompt.

- audience:

  (Optional) A string describing the target population (e.g., "adults",
  "high school students").

- item.examples:

  (Optional) A data frame of example items. Must contain a column
  matching each item type to extract examples.

## Value

A modified list of character strings, with each prompt updated to
include relevant metadata, instructions, and formatting examples as
needed.
