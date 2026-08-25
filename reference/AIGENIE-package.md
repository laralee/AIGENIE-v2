# AIGENIE: Automatic Item Generation and Validation via Network-Integrated Evaluation

Automated psychological scale development and structural validation
using large language models (LLMs) and network psychometric methods.
Implements the AI-GENIE framework (Automatic Item Generation and
Validation via Network-Integrated Evaluation) to generate candidate
items, compute embedding representations, and estimate dimensional
structure using Exploratory Graph Analysis (EGA). Item quality is
evaluated using Unique Variable Analysis to identify redundant items and
Bootstrap EGA to assess item and dimension stability. Supports both
fully automated item generation and analysis of user-provided item sets,
facilitating efficient, theory-informed measurement development prior to
empirical data collection.

AIGENIE is an R package for automated psychological scale development
and structural validation using large language models (LLMs) and network
psychometric methods. The package implements the AI-GENIE framework
(Automatic Item Generation and Validation via Network-Integrated
Evaluation) to generate candidate items, compute embedding
representations, and estimate dimensional structure using Exploratory
Graph Analysis (EGA). Item quality is further evaluated using Unique
Variable Analysis to identify redundant items and Bootstrap EGA to
assess item and dimension stability. AI-GENIE supports both fully
automated item generation and analysis of user-provided item sets,
facilitating efficient, theory-informed measurement development prior to
empirical data collection.

## Main Functions

- [`AIGENIE`](https://laralee.github.io/AIGENIE/reference/AIGENIE.md):

  Full pipeline: generate items using LLMs, compute embeddings, and
  perform EGA-based redundancy and stability item pool reduction and
  structural validation.

- [`GENIE`](https://laralee.github.io/AIGENIE/reference/GENIE.md):

  Embedding and item-pool reduction (or filtering) only: takes existing
  items, computes embeddings, and performs EGA, UVA, and bootEGA. Use
  when you already have candidate items.

- [`local_AIGENIE`](https://laralee.github.io/AIGENIE/reference/local_AIGENIE.md):

  Full pipeline using local GGUF models instead of cloud APIs. Requires
  downloading model files.

- [`local_GENIE`](https://laralee.github.io/AIGENIE/reference/local_GENIE.md):

  GENIE with local embedding models.

## Supported LLM Providers

For item generation, AIGENIE supports:

- **OpenAI**: GPT-4o, GPT-4, GPT-3.5-turbo, o1 series, plus newer
  models.

- **Anthropic**: Claude Sonnet 4.5, Opus 4.5, Haiku 4.5, plus newer
  models

- **Groq**: Llama 3.3 70b versatile, Llama 4 Maverick 17b 128e instruct,
  GPT-OSS-120b, GPT-OSS-20b, plus other models

- **Local**: Any GGUF model via llama-cpp-python

## Supported Embedding Providers

For computing semantic embeddings, AIGENIE supports:

- **OpenAI**: text-embedding-3-small, text-embedding-3-large

- **Jina AI**: jina-embeddings-v3, jina-embeddings-v4 (with task
  adapters), and others

- **HuggingFace**: BAAI/bge, thenlper/gte, sentence-transformers models,
  and others

- **Local**: BERT, RoBERTa, DistilBERT, and other transformer models

## Environment Setup Functions

- [`reinstall_python_env`](https://laralee.github.io/AIGENIE/reference/reinstall_python_env.md):

  Reinstall the Python environment if you encounter issues.

- [`install_gpu_support`](https://laralee.github.io/AIGENIE/reference/install_gpu_support.md):

  Enable GPU acceleration for faster local model inference.

- [`install_local_llm_support`](https://laralee.github.io/AIGENIE/reference/install_local_llm_support.md):

  Install llama-cpp-python for running local GGUF models.

- [`python_env_info`](https://laralee.github.io/AIGENIE/reference/python_env_info.md):

  Display diagnostic information about the Python environment.

- [`set_huggingface_token`](https://laralee.github.io/AIGENIE/reference/set_huggingface_token.md):

  Configure HuggingFace authentication for gated models.

## Quick Start


    # Define what you want to measure
    item.attributes <- list(
      anxiety = c("worry", "nervousness", "fear"),
      depression = c("sadness", "hopelessness", "fatigue")
    )

    # Generate and reduce items (OpenAI)
    results <- AIGENIE(
      item.attributes = item.attributes,
      openai.API = Sys.getenv("OPENAI_API_KEY"),
      domain = "clinical psychology",
      scale.title = "Mood Assessment Scale",
      target.N = 30
    )

    # Or use Anthropic Claude with Jina embeddings
    results <- AIGENIE(
      item.attributes = item.attributes,
      anthropic.API = Sys.getenv("ANTHROPIC_API_KEY"),
      jina.API = Sys.getenv("JINA_API_KEY"),
      model = "sonnet",
      embedding.model = "jina-embeddings-v3",
      domain = "clinical psychology",
      scale.title = "Mood Assessment Scale",
      target.N = 30
    )

    # Or use free Groq API
    results <- AIGENIE(
      item.attributes = item.attributes,
      groq.API = Sys.getenv("GROQ_API_KEY"),
      openai.API = Sys.getenv("OPENAI_API_KEY"),
      model = "llama-3.3-70b-versatile",
      domain = "clinical psychology",
      scale.title = "Mood Assessment Scale",
      target.N = 30
    )

## Getting API Keys

- **OpenAI**: <https://platform.openai.com/api-keys>

- **Anthropic**: <https://console.anthropic.com/>

- **Groq**: <https://console.groq.com/> (free tier available)

- **Jina AI**: <https://jina.ai/> (free tier available)

- **HuggingFace**: <https://huggingface.co/settings/tokens>

## Dependencies

AIGENIE uses a Python backend managed via UV for embedding generation
and some LLM interactions. The Python environment is automatically
configured on first use. Required R packages include EGAnet for network
analysis and reticulate for Python integration.

## References

Russell-Lasalandra, L. L., Christensen, A. P., & Golino, H. (2026).
Generative psychometrics via AI-GENIE: Automatic item generation and
validation with network-integrated evaluation. *Behavior Research
Methods*, *58*(8), 217. <https://doi.org/10.3758/s13428-026-03082-1>

Russell-Lasalandra, L. L., & Golino, H. (2026). Prompt engineering for
scale development in generative psychometrics. *PsyArXiv*.
<https://osf.io/preprints/psyarxiv/znqkm_v2>

Russell-Lasalandra, L. L., Golino, H., Garrido, L. E., & Christensen, A.
P. (2026). The ultimate tutorial for AI-driven scale development in
generative psychometrics: Releasing AIGENIE from its bottle. *PsyArXiv*.
<https://osf.io/preprints/psyarxiv/arfg3_v1>

Garrido, L. E., Russell-Lasalandra, L. L., & Golino, H. (2025).
Estimating dimensional structure in generative psychometrics: Comparing
PCA and network methods using large language model item embeddings.
*PsyArXiv*. <https://osf.io/preprints/psyarxiv/2s7pw_v1>

Golino, H., & Christensen, A. P. (2024). *EGAnet: Exploratory Graph
Analysis*. R package. <https://r-ega.net>

Golino, H. (2025). What I Learned with John: On the Depth of Language
and How to Measure It with Large Language Models and Algorithm
(Kolmogorov) Complexity. *PsyArXiv*.
<https://doi.org/10.31234/osf.io/b92n5>

Golino, H., Garrido, L., & Russell-Lasalandra, L. L. (2026). Optimizing
the Landscape of LLM Embeddings with Dynamic Exploratory Graph Analysis
for Generative Psychometrics: A Monte Carlo Study. *arXiv*.
arXiv:2601.17010. <https://doi.org/10.48550/arXiv.2601.17010>

Christensen, A. P., & Golino, H. (2021a). Estimating the stability of
the number of factors via Bootstrap Exploratory Graph Analysis: A
tutorial. *Psych*, *3*(3), 479-500.  

Christensen, A. P., Garrido, L. E., & Golino, H. (2023). Unique variable
analysis: A network psychometrics method to detect local dependence.
*Multivariate Behavioral Research*.  

Golino, H., Moulder, R., Shi, D., Christensen, A. P., Garrido, L. E.,
Nieto, M. D., Nesselroade, J., Sadana, R., Thiyagarajan, J. A., & Boker,
S. M. (2020). Entropy fit indices: New fit measures for assessing the
structure and dimensionality of multiple latent variables. *Multivariate
Behavioral Research*.  

Golino, H., Shi, D., Christensen, A. P., Garrido, L. E., Nieto, M. D.,
Sadana, R., Thiyagarajan, J. A., & Martinez-Molina, A. (2020).
Investigating the performance of exploratory graph analysis and
traditional techniques to identify the number of latent factors: A
simulation and tutorial. *Psychological Methods*, *25*, 292-320.  

## See also

Useful links:

- <https://laralee.github.io/AIGENIE/>

- <https://github.com/laralee/AIGENIE>

- Report bugs at <https://github.com/laralee/AIGENIE/issues>

[`AIGENIE`](https://laralee.github.io/AIGENIE/reference/AIGENIE.md) for
the main function,
[`GENIE`](https://laralee.github.io/AIGENIE/reference/GENIE.md) for
embedding-only analysis,
[`EGAnet`](https://rdrr.io/pkg/EGAnet/man/EGAnet-package.html) for the
underlying EGA analysis methods.

## Author

**Maintainer**: Lara Russell-Lasalandra <llr7cb@virginia.edu>
([ORCID](https://orcid.org/0009-0000-3014-1937))

Authors:

- Alexander Christensen ([ORCID](https://orcid.org/0000-0002-9798-7037))

- Hudson Golino ([ORCID](https://orcid.org/0000-0002-1601-1447))

Lara Lee Russell-Lasalandra, Hudson Golino, Alexander P. Christensen
