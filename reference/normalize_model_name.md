# Normalize Model Name (Legacy Compatibility)

Validates and normalizes model names. This function maintains backward
compatibility with existing code.

Converts model names to the standardized format: Provider/model-name
Maintains backward compatibility with existing model names.

## Usage

``` r
normalize_model_name(
  model,
  groq.API = NULL,
  openai.API = NULL,
  anthropic.API = NULL,
  silently = FALSE
)

normalize_model_name(
  model,
  groq.API = NULL,
  openai.API = NULL,
  anthropic.API = NULL,
  silently = FALSE
)
```

## Arguments

- model:

  Character string of the model name

- groq.API:

  Optional Groq API key

- openai.API:

  Optional OpenAI API key

- anthropic.API:

  Optional Anthropic API key

- silently:

  Logical, suppress warnings

## Value

Normalized model name string

List with normalized model name and detected provider
