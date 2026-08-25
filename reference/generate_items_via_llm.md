# Generate Items via LLM

Generates scale items using the specified LLM provider. Supports OpenAI,
Groq, and local GGUF models.

## Usage

``` r
generate_items_via_llm(
  main.prompts,
  system.role,
  model,
  top.p,
  temperature,
  adaptive,
  silently,
  groq.API,
  openai.API,
  anthropic.API = NULL,
  target.N
)
```

## Arguments

- main.prompts:

  Named list of prompts for each item type

- system.role:

  Character string defining the system role

- model:

  Character string specifying the model

- top.p:

  Numeric. Nucleus sampling parameter

- temperature:

  Numeric. Sampling temperature

- adaptive:

  Logical. Use adaptive generation with previous items?

- silently:

  Logical. Suppress progress messages?

- groq.API:

  Optional Groq API key

- openai.API:

  Optional OpenAI API key

- target.N:

  Named list of target item counts per type

## Value

A list with 'items' data frame and 'successful' flag
