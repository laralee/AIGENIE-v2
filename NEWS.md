# AIGENIE 2.1.2

## New features
- Exposed `boot.iter` and `ncores` in `AIGENIE()`, `GENIE()`, `local_AIGENIE()`, and `local_GENIE()`. The `boot.iter` default remains 500, matching the current reduction pipeline; `ncores = NULL` preserves EGAnet's existing default core behavior.
- Added a publication-ready `filtering_audit` with item-level filtering
  provenance for UVA and bootEGA decisions.
- Added per-type `reduction_summary` outputs.
- Added pre-reduction network-loading diagnostics for filtered items.
- Added bundled GPT-5.4 items and frozen embeddings for reproducible GENIE
  examples.
- Added an official filtering-audit vignette and pkgdown documentation site.

## Reproducibility
- Restored correct UVA redundancy detection with current EGAnet versions.
- Set bootEGA reduction to 500 bootstrap iterations.
- Full embeddings now win exact full/sparse NMI ties within an EGA model.
- `run.overall = TRUE` now performs a pooled post-reduction fit without
  applying a second item-reduction pass.

## CRAN preparation
- Updated `Authors@R` so Lara Russell-Lasalandra and Hudson Golino are
  explicitly represented as package authors/co-creators and copyright
  holders; Alexander Christensen remains a full author and copyright holder.
- Added a formal AIGENIE software citation with the CRAN package DOI.
- Updated the methodology citation to the published 2026
  *Behavior Research Methods* article.

# AIGENIE 2.1.0

## New Features

* **Anthropic Claude support**: Generate items using Claude Sonnet 4.5, Opus 4, and Haiku 4.5 models via `anthropic.API`.
* **Jina AI embeddings**: Compute embeddings using Jina models (v2, v3, v4) via `jina.API`, including Matryoshka truncation and task-specific adapters.
* **Slash-style Groq models**: Route HuggingFace-style model IDs (e.g., `"meta-llama/llama-4-scout-17b-16e-instruct"`, `"qwen/qwen3-32b"`) to Groq when `groq.API` is provided.
* **`chat()` function**: Send arbitrary prompts to any supported LLM provider with optional repetitions, temperature control, and system role customization.
* **`list_available_models()`**: Query available models across all providers, with optional filtering by type (`"chat"` or `"embedding"`).
* **Multi-provider mixing**: Use one provider for item generation and another for embeddings (e.g., Anthropic items + Jina embeddings).

## Improvements

* Comprehensive input validation across all user-facing functions.
* Unified embedding dispatch via `generate_embeddings()` supporting OpenAI, Jina AI, HuggingFace (API and local), and sentence-transformers.
* Model alias resolution for common shorthand names (`"sonnet"`, `"llama3"`, `"deepseek"`, `"qwen"`, etc.).
* Improved error messages with provider-specific guidance.

## Bug Fixes
* Fixed provider detection for slash-style model names that were incorrectly routed to HuggingFace instead of Groq.

---

# AIGENIE 2.0.0

## Major Changes

* Complete rewrite of the pipeline architecture.
* Introduced `GENIE()` for validation of user-provided item sets (embedding → EGA → UVA → bootEGA).
* Added Groq API support for open-source LLM item generation.
* Added local LLM support via llama-cpp-python (`local_AIGENIE()`, `local_GENIE()`, `local_chat()`).
* Python environment management via UV and reticulate.
* HuggingFace embedding support (API and local sentence-transformers).

---

# AIGENIE 1.0.0

* Initial release.
* OpenAI-based item generation and embedding.
* EGA-based dimensional structure estimation.
* UVA for redundancy detection.
* Bootstrap EGA for stability assessment.

