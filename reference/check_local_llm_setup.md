# Check Local LLM Setup

Verifies that all requirements for local LLM inference are met,
including Python environment, llama-cpp-python installation, and model
file accessibility.

## Usage

``` r
check_local_llm_setup(model.path, silently = FALSE)
```

## Arguments

- model.path:

  Path to the GGUF model file

- silently:

  Logical. Suppress progress messages?

## Value

Logical. TRUE if setup is complete, FALSE otherwise.
