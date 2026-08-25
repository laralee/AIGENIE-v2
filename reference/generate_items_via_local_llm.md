# Generate Items Using Local LLM (GGUF)

Generates items using a locally installed GGUF model via
llama-cpp-python.

## Usage

``` r
generate_items_via_local_llm(
  main.prompts,
  system.role,
  model.path,
  temperature,
  top.p,
  adaptive,
  silently,
  target.N,
  n.ctx = 4096,
  n.gpu.layers = -1,
  max.tokens = 1024
)
```

## Arguments

- main.prompts:

  Named list of prompts

- system.role:

  Character string with system role

- model.path:

  Path to local GGUF model file

- temperature:

  Numeric. Sampling temperature

- top.p:

  Numeric. Nucleus sampling parameter

- adaptive:

  Logical. Use adaptive generation?

- silently:

  Logical. Suppress messages?

- target.N:

  Named list of target counts

- n.ctx:

  Integer. Context window size

- n.gpu.layers:

  Integer. GPU layers (-1 for all)

- max.tokens:

  Integer. Max tokens per generation

## Value

A list with 'items' data frame and 'successful' flag
