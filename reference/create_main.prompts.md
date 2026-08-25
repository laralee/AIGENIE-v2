# Create Initial Main Prompts for Item Generation

Constructs structured prompts for an LLM to generate scale items based
on a list of item attributes, optional item type definitions, audience
and domain context, and example items. Each resulting prompt includes
strict formatting requirements, attribute listings, and item-generation
instructions.

## Usage

``` r
create_main.prompts(
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

- item.attributes:

  A named list where each element is a character vector of attribute
  names for an item type.

- item.type.definitions:

  (Optional) A named list of textual definitions for each item type,
  used to provide conceptual clarity in the prompt.

- domain:

  (Optional) A string specifying the domain (e.g., "psychological",
  "clinical") the items belong to.

- scale.title:

  (Optional) The title of the scale (e.g., "Emotion Regulation
  Inventory").

- prompt.notes:

  (Optional) A named list of additional instructions or warnings to
  include per item type.

- audience:

  (Optional) A string describing the target audience or population
  (e.g., "adolescents", "working adults").

- item.examples:

  (Optional) A data frame of existing high-quality example items. Used
  to guide item phrasing and structure. Must be compatible with the
  helper
  [`construct_item.examples_string()`](https://laralee.github.io/AIGENIE/reference/construct_item.examples_string.md).

## Value

A named list of character strings. Each entry corresponds to one item
type and contains a complete prompt to guide an LLM in generating two
distinct items per attribute, formatted as a JSON array.
