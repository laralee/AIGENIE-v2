# Set Hugging Face Token

Configure your HuggingFace API token for accessing gated models like
Google's EmbeddingGemma or other restricted models.

## Usage

``` r
set_huggingface_token(token, save = TRUE)
```

## Arguments

- token:

  Character. Your HuggingFace API token from
  <https://huggingface.co/settings/tokens>.

- save:

  Logical. If `TRUE` (default), saves the token to the HuggingFace cache
  for future sessions. If `FALSE`, sets it only for the current R
  session.

## Value

Invisible `TRUE` on success.

## Details

Some embedding models on HuggingFace require authentication:

- `google/embeddinggemma-300m`

- Other gated models

Before using these models, you must:

1.  Create an account at <https://huggingface.co>

2.  Accept the model's license on its model page

3.  Generate an access token at <https://huggingface.co/settings/tokens>

4.  Call this function with your token

## Examples

``` r
if (FALSE) { # \dontrun{
# Set token (saved permanently for future sessions)
set_huggingface_token("hf_xxxxxxxxxxxxxxxxx")

# Set token for this session only (not saved)
set_huggingface_token("hf_xxxxxxxxxxxxxxxxx", save = FALSE)

# Now you can use gated HuggingFace models
results <- GENIE(
  items = my_items,
  embedding.model = "BAAI/bge-large-en-v1.5",
  hf.token = "hf_xxxxxxxxxxxxxxxxx"
)
} # }
```
