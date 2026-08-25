# Construct Formatted String of Example Items for Prompts

Given a validated item examples data frame, this function constructs a
JSON string of example items for a given type, to be used in prompt
building.

## Usage

``` r
construct_item.examples_string_for_prompt(item.examples, current_type)
```

## Arguments

- item.examples:

  A validated data frame with `type`, `attribute`, `statement`.

- current_type:

  A string specifying which type to filter for.

## Value

A single JSON-formatted string (or NULL if no matches).
