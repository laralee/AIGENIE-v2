# Download a Local LLM Model

Downloads a GGUF model from HuggingFace for use with local_AIGENIE.
Models are saved to a user-specified directory or the default AIGENIE
models directory.

## Usage

``` r
get_local_llm(repo_id, filename, save_dir = NULL, hf.token = NULL)
```

## Arguments

- repo_id:

  HuggingFace repository ID (e.g.,
  "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")

- filename:

  Specific GGUF filename to download (e.g.,
  "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

- save_dir:

  Directory to save the model. If NULL, uses the default AIGENIE models
  directory.

- hf.token:

  Optional HuggingFace token for gated models

## Value

Character string with the full path to the downloaded model file.

## Examples

``` r
if (FALSE) { # \dontrun{
# Download a Mistral 7B model
model_path <- get_local_llm(
  repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
  filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

# Use it with local_AIGENIE
results <- local_AIGENIE(
  item.attributes = my_attributes,
  model.path = model_path
)
} # }
```
