# Generate Text Using Groq API

Generate Text Using Groq API

## Usage

``` r
generate_text_groq(
  prompt,
  system.role = NULL,
  model = "llama-3.3-70b-versatile",
  temperature = 1,
  top.p = 1,
  max_tokens = 2048,
  api_key
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

  Numeric. Sampling temperature

- top.p:

  Numeric. Nucleus sampling parameter

- max_tokens:

  Integer. Maximum tokens to generate

- api_key:

  Groq API key

## Value

Character string with the generated text
