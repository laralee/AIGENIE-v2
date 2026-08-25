# Clean and Parse LLM Response

Parses LLM-generated text to extract structured item data. Handles JSON
format and falls back to text parsing.

## Usage

``` r
cleaning_function(raw_text, item_type)
```

## Arguments

- raw_text:

  Character string with LLM response

- item_type:

  Character string with the item type

## Value

Data frame with type, attribute, statement columns
