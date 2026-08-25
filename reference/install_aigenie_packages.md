# Install AI-GENIE Python Packages Using UV

Install AI-GENIE Python Packages Using UV

## Usage

``` r
install_aigenie_packages(
  env_path,
  include_huggingface = TRUE,
  include_local_llm = FALSE,
  gpu = FALSE
)
```

## Arguments

- env_path:

  Path to the virtual environment

- include_huggingface:

  Logical. Include HuggingFace packages?

- include_local_llm:

  Logical. Include local LLM (llama-cpp) support?

- gpu:

  Logical. Install GPU-enabled PyTorch?

## Value

TRUE invisibly on success
