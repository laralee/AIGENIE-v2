# Ensure AI-GENIE Python Environment is Ready

Sets up the Python environment with all required dependencies for
AI-GENIE. Uses UV for fast, reliable package management. This function
is called automatically when needed, but can also be called directly.

## Usage

``` r
ensure_aigenie_python(
  force_reinstall = FALSE,
  include_huggingface = TRUE,
  include_local_llm = FALSE,
  gpu = FALSE
)
```

## Arguments

- force_reinstall:

  Logical. Force complete reinstallation?

- include_huggingface:

  Logical. Include HuggingFace packages? Default TRUE.

- include_local_llm:

  Logical. Include local LLM support? Default FALSE.

- gpu:

  Logical. Install GPU-enabled PyTorch? Default FALSE.

## Value

TRUE invisibly on success
