# Check for users who pasted the example code but didn't add an API key

Check for users who pasted the example code but didn't add an API key

## Usage

``` r
check_for_default_APIs(
  hf.token,
  groq.API = NULL,
  openai.API,
  anthropic.API = NULL,
  jina.API = NULL
)
```

## Arguments

- hf.token:

  The hugging face token provided

- groq.API:

  The Groq API key provided

- openai.API:

  The OpenAI API key provided

- anthropic.API:

  Character. Anthropic API key. Can be NULL when Anthropic models are
  not used.

- jina.API:

  Character. Jina AI API key. Can be NULL when Jina embeddings are not
  used.
