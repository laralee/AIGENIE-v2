# Install Local LLM Support

Installs llama-cpp-python for running local GGUF models with
[`local_AIGENIE`](https://laralee.github.io/AIGENIE/reference/local_AIGENIE.md).
On Apple Silicon Macs, this includes Metal acceleration support for fast
inference.

## Usage

``` r
install_local_llm_support()
```

## Value

Invisible `TRUE` on success.

## Details

After installation, you can use any GGUF model file with
[`local_AIGENIE()`](https://laralee.github.io/AIGENIE/reference/local_AIGENIE.md).
Download GGUF models from HuggingFace (search for "GGUF" format).

Popular model recommendations:

- **Llama 3 8B**: Good balance of quality and speed

- **Mistral 7B**: Fast with good quality

- **Qwen 2.5**: Strong multilingual support

## See also

[`local_AIGENIE`](https://laralee.github.io/AIGENIE/reference/local_AIGENIE.md),
[`reinstall_python_env`](https://laralee.github.io/AIGENIE/reference/reinstall_python_env.md).

## Examples

``` r
if (FALSE) { # \dontrun{
# Install local LLM support
install_local_llm_support()

# Then download a GGUF model and use with local_AIGENIE
results <- local_AIGENIE(
  item.attributes = my_traits,
  model.path = "~/models/llama-3-8b.Q4_K_M.gguf",
  embedding.model = "bert-base-uncased"
)
} # }
```
