# Generate Text Using Any Supported LLM Provider

Unified interface for text generation that automatically routes to the
appropriate provider (OpenAI, Groq, Anthropic, or HuggingFace).

## Usage

``` r
generate_text_llm(
  prompt,
  system.role = NULL,
  model = "gpt-4o",
  temperature = 1,
  top.p = 1,
  max_tokens = 2048,
  openai.API = NULL,
  groq.API = NULL,
  anthropic.API = NULL,
  hf.token = NULL
)
```

## Arguments

- prompt:

  Character string with the user prompt

- system.role:

  Character string with the system prompt

- model:

  Character string specifying the model

- temperature:

  Numeric. Sampling temperature (0-2)

- top.p:

  Numeric. Nucleus sampling parameter (0-1)

- max_tokens:

  Integer. Maximum tokens to generate

- openai.API:

  Optional OpenAI API key

- groq.API:

  Optional Groq API key

- anthropic.API:

  Optional Anthropic API key

- hf.token:

  Optional HuggingFace token

## Value

Character string with the generated text
