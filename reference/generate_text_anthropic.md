# Generate Text Using Anthropic Messages API

Generates text using Anthropic's Claude models via the /v1/messages
endpoint. Uses the requests library directly (no extra SDK dependency).

## Usage

``` r
generate_text_anthropic(
  prompt,
  system.role = NULL,
  model = "claude-sonnet-4-5-20250929",
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

  Character string specifying the Claude model

- temperature:

  Numeric. Sampling temperature (0-1)

- top.p:

  Numeric. Nucleus sampling parameter (0-1)

- max_tokens:

  Integer. Maximum tokens to generate

- api_key:

  Anthropic API key

## Value

Character string with the generated text
