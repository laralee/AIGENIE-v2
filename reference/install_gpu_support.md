# Install GPU Support for AI-GENIE

Reinstalls the Python environment with GPU-enabled PyTorch for faster
inference with local embedding models. Requires a CUDA-compatible NVIDIA
GPU and proper driver installation.

## Usage

``` r
install_gpu_support()
```

## Value

Invisible `TRUE` on success.

## Details

This function:

1.  Removes the existing Python environment

2.  Creates a new environment with GPU-enabled PyTorch

3.  Installs all HuggingFace dependencies

On Apple Silicon Macs, MPS (Metal Performance Shaders) acceleration is
used automatically without needing this function.

## See also

[`reinstall_python_env`](https://laralee.github.io/AIGENIE/reference/reinstall_python_env.md),
[`python_env_info`](https://laralee.github.io/AIGENIE/reference/python_env_info.md).

## Examples

``` r
if (FALSE) { # \dontrun{
# Enable GPU acceleration (requires NVIDIA GPU + CUDA)
install_gpu_support()
} # }
```
