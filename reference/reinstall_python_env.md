# Reinstall AI-GENIE Python Environment

Removes and recreates the Python virtual environment with all required
dependencies. Use this function if you encounter Python-related errors,
want to update Python packages, or need to change the environment
configuration.

AIGENIE uses UV (<https://docs.astral.sh/uv/>) for fast, reliable Python
environment management.

## Usage

``` r
reinstall_python_env(
  include_huggingface = TRUE,
  include_local_llm = FALSE,
  gpu = FALSE
)
```

## Arguments

- include_huggingface:

  Logical. Include HuggingFace packages (transformers,
  sentence-transformers, torch). Required for local embeddings with
  HuggingFace models. Default `TRUE`.

- include_local_llm:

  Logical. Include llama-cpp-python for running local GGUF models with
  [`local_AIGENIE`](https://laralee.github.io/AIGENIE/reference/local_AIGENIE.md).
  Default `FALSE`.

- gpu:

  Logical. Install GPU-enabled PyTorch. Requires CUDA-compatible NVIDIA
  GPU and proper driver installation. Default `FALSE`.

## Value

Invisible `TRUE` on success.

## See also

[`python_env_info`](https://laralee.github.io/AIGENIE/reference/python_env_info.md)
to check environment status,
[`install_gpu_support`](https://laralee.github.io/AIGENIE/reference/install_gpu_support.md)
for GPU setup,
[`install_local_llm_support`](https://laralee.github.io/AIGENIE/reference/install_local_llm_support.md)
for local model setup.

## Examples

``` r
if (FALSE) { # \dontrun{
# Fix Python environment issues
reinstall_python_env()

# Reinstall with GPU support
reinstall_python_env(gpu = TRUE)

# Minimal install (API-only, no HuggingFace - faster)
reinstall_python_env(include_huggingface = FALSE)

# Full install with local LLM support
reinstall_python_env(include_huggingface = TRUE, include_local_llm = TRUE)
} # }
```
