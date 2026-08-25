# List Available Models

Queries the OpenAI, Groq, Anthropic, and/or Jina AI APIs to retrieve
currently available models. Requires API keys for live provider queries.
Jina AI models are returned from a curated static list (no list
endpoint).

## Usage

``` r
list_available_models(
  provider = NULL,
  openai.API = NULL,
  groq.API = NULL,
  anthropic.API = NULL,
  type = NULL
)
```

## Arguments

- provider:

  Optional. Filter by provider: "openai", "groq", "anthropic", "jina",
  or NULL for all.

- openai.API:

  Optional OpenAI API key. If NULL, checks OPENAI_API_KEY env var.

- groq.API:

  Optional Groq API key. If NULL, checks GROQ_API_KEY env var.

- anthropic.API:

  Optional Anthropic API key. If NULL, checks ANTHROPIC_API_KEY env var.

- type:

  Filter by model type: "chat", "embedding", or NULL for all. Default is
  NULL (show everything).

## Value

A data frame with columns: provider, model, type, display_name, created
