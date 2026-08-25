# Validate Local LLM Generation Parameters

Validates parameters specific to local LLM generation

## Usage

``` r
validate_local_llm_params(n.ctx, n.gpu.layers, max.tokens)
```

## Arguments

- n.ctx:

  Context window size

- n.gpu.layers:

  Number of layers to offload to GPU

- max.tokens:

  Maximum tokens for generation

## Value

A list of validated parameters
