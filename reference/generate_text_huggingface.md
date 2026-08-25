# Generate Text Using HuggingFace Inference API

Generate Text Using HuggingFace Inference API

## Usage

``` r
generate_text_huggingface(
  prompt,
  system.role = NULL,
  model,
  temperature = 1,
  top.p = 1,
  max_tokens = 2048,
  hf_token = NULL
)
```

## Arguments

- prompt:

  Character string with the user prompt

- system.role:

  Character string with the system prompt

- model:

  Character string specifying the HuggingFace model ID

- temperature:

  Numeric. Sampling temperature

- top.p:

  Numeric. Nucleus sampling parameter

- max_tokens:

  Integer. Maximum tokens to generate

- hf_token:

  Optional HuggingFace token

## Value

Character string with the generated text
