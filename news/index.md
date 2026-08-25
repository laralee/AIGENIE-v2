# Changelog

## AIGENIE 2.1.0

### New Features

- **Anthropic Claude support**: Generate items using Claude Sonnet 4.5,
  Opus 4, and Haiku 4.5 models via `anthropic.API`.
- **Jina AI embeddings**: Compute embeddings using Jina models (v2, v3,
  v4) via `jina.API`, including Matryoshka truncation and task-specific
  adapters.
- **Slash-style Groq models**: Route HuggingFace-style model IDs (e.g.,
  `"meta-llama/llama-4-scout-17b-16e-instruct"`, `"qwen/qwen3-32b"`) to
  Groq when `groq.API` is provided.
- **[`chat()`](https://laralee.github.io/AIGENIE/reference/chat.md)
  function**: Send arbitrary prompts to any supported LLM provider with
  optional repetitions, temperature control, and system role
  customization.
- **[`list_available_models()`](https://laralee.github.io/AIGENIE/reference/list_available_models.md)**:
  Query available models across all providers, with optional filtering
  by type (`"chat"` or `"embedding"`).
- **Multi-provider mixing**: Use one provider for item generation and
  another for embeddings (e.g., Anthropic items + Jina embeddings).

### Improvements

- Comprehensive input validation across all user-facing functions.
- Unified embedding dispatch via
  [`generate_embeddings()`](https://laralee.github.io/AIGENIE/reference/generate_embeddings.md)
  supporting OpenAI, Jina AI, HuggingFace (API and local), and
  sentence-transformers.
- Model alias resolution for common shorthand names (`"sonnet"`,
  `"llama3"`, `"deepseek"`, `"qwen"`, etc.).
- Improved error messages with provider-specific guidance.

### Bug Fixes

- Fixed provider detection for slash-style model names that were
  incorrectly routed to HuggingFace instead of Groq.

------------------------------------------------------------------------

## AIGENIE 2.0.0

### Major Changes

- Complete rewrite of the pipeline architecture.
- Introduced
  [`GENIE()`](https://laralee.github.io/AIGENIE/reference/GENIE.md) for
  validation of user-provided item sets (embedding → EGA → UVA →
  bootEGA).
- Added Groq API support for open-source LLM item generation.
- Added local LLM support via llama-cpp-python
  ([`local_AIGENIE()`](https://laralee.github.io/AIGENIE/reference/local_AIGENIE.md),
  [`local_GENIE()`](https://laralee.github.io/AIGENIE/reference/local_GENIE.md),
  [`local_chat()`](https://laralee.github.io/AIGENIE/reference/local_chat.md)).
- Python environment management via UV and reticulate.
- HuggingFace embedding support (API and local sentence-transformers).

------------------------------------------------------------------------

## AIGENIE 1.0.0

- Initial release.
- OpenAI-based item generation and embedding.
- EGA-based dimensional structure estimation.
- UVA for redundancy detection.
- Bootstrap EGA for stability assessment.
