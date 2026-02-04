#' @keywords internal
"_PACKAGE"

#' AIGENIE: Automatic Item Generation and Validation (Structural Validity) with Network-Integrated Evaluation. AI-Guided Exploration and Network Inference for Items and Embeddings.
#'
#' @description
#' AIGENIE is an R package for automated psychological scale development and structural validation using large language models (LLMs) and network psychometric methods. The package implements the AI-GENIE framework (Automatic Item Generation and Validation via Network-Integrated Evaluation) to generate candidate items, compute embedding representations, and estimate dimensional structure using Exploratory Graph Analysis (EGA). Item quality is further evaluated using Unique Variable Analysis to identify redundant items and Bootstrap EGA to assess item and dimension stability. AI-GENIE supports both fully automated item generation and analysis of user-provided item sets, facilitating efficient, theory-informed measurement development prior to empirical data collection.
#'
#' @section Main Functions:
#' \describe{
#'   \item{\code{\link{AIGENIE}}}{Full pipeline: generate items using LLMs,
#'     compute embeddings, and perform EGA-based redundancy and stability item pool reduction and structural validation.}
#'   \item{\code{\link{GENIE}}}{Embedding and item-pool reduction (or filtering) only: takes existing items,
#'     computes embeddings, and performs EGA, UVA, and bootEGA. Use when you already have
#'     candidate items.
#'   }
#'   \item{\code{\link{local_AIGENIE}}}{Full pipeline using local GGUF models
#'     instead of cloud APIs. Requires downloading model files.}
#'   \item{\code{\link{local_GENIE}}}{GENIE with local embedding models.}
#' }
#'
#' @section Supported LLM Providers:
#' For item generation, AIGENIE supports:
#' \itemize{
#'   \item \strong{OpenAI}: GPT-4o, GPT-4, GPT-3.5-turbo, o1 series, plus newer models.
#'   \item \strong{Anthropic}: Claude Sonnet 4.5, Opus 4.5, Haiku 4.5, plus newer models
#'   \item \strong{Groq}: Llama 3.3 70b versatile, Llama 4 Maverick 17b 128e instruct, GPT-OSS-120b, GPT-OSS-20b, plus other models
#'   \item \strong{Local}: Any GGUF model via llama-cpp-python
#' }
#'
#' @section Supported Embedding Providers:
#' For computing semantic embeddings, AIGENIE supports:
#' \itemize{
#'   \item \strong{OpenAI}: text-embedding-3-small, text-embedding-3-large
#'   \item \strong{Jina AI}: jina-embeddings-v3, jina-embeddings-v4 (with task adapters), and others
#'   \item \strong{HuggingFace}: BAAI/bge, thenlper/gte, sentence-transformers models, and others
#'   \item \strong{Local}: BERT, RoBERTa, DistilBERT, and other transformer models
#' }
#'
#' @section Environment Setup Functions:
#' \describe{
#'   \item{\code{\link{reinstall_python_env}}}{Reinstall the Python environment
#'     if you encounter issues.}
#'   \item{\code{\link{install_gpu_support}}}{Enable GPU acceleration for
#'     faster local model inference.}
#'   \item{\code{\link{install_local_llm_support}}}{Install llama-cpp-python
#'     for running local GGUF models.}
#'   \item{\code{\link{python_env_info}}}{Display diagnostic information about
#'     the Python environment.}
#'   \item{\code{\link{set_huggingface_token}}}{Configure HuggingFace authentication
#'     for gated models.}
#' }
#'
#' @section Quick Start:
#' \preformatted{
#' # Define what you want to measure
#' item.attributes <- list(
#'   anxiety = c("worry", "nervousness", "fear"),
#'   depression = c("sadness", "hopelessness", "fatigue")
#' )
#'
#' # Generate and reduce items (OpenAI)
#' results <- AIGENIE(
#'   item.attributes = item.attributes,
#'   openai.API = Sys.getenv("OPENAI_API_KEY"),
#'   domain = "clinical psychology",
#'   scale.title = "Mood Assessment Scale",
#'   target.N = 30
#' )
#'
#' # Or use Anthropic Claude with Jina embeddings
#' results <- AIGENIE(
#'   item.attributes = item.attributes,
#'   anthropic.API = Sys.getenv("ANTHROPIC_API_KEY"),
#'   jina.API = Sys.getenv("JINA_API_KEY"),
#'   model = "sonnet",
#'   embedding.model = "jina-embeddings-v3",
#'   domain = "clinical psychology",
#'   scale.title = "Mood Assessment Scale",
#'   target.N = 30
#' )
#'
#' # Or use free Groq API
#' results <- AIGENIE(
#'   item.attributes = item.attributes,
#'   groq.API = Sys.getenv("GROQ_API_KEY"),
#'   openai.API = Sys.getenv("OPENAI_API_KEY"),
#'   model = "llama-3.3-70b-versatile",
#'   domain = "clinical psychology",
#'   scale.title = "Mood Assessment Scale",
#'   target.N = 30
#' )
#' }
#'
#' @section Getting API Keys:
#' \itemize{
#'   \item \strong{OpenAI}: \url{https://platform.openai.com/api-keys}
#'   \item \strong{Anthropic}: \url{https://console.anthropic.com/}
#'   \item \strong{Groq}: \url{https://console.groq.com/} (free tier available)
#'   \item \strong{Jina AI}: \url{https://jina.ai/} (free tier available)
#'   \item \strong{HuggingFace}: \url{https://huggingface.co/settings/tokens}
#' }
#'
#' @section Dependencies:
#' AIGENIE uses a Python backend managed via UV for embedding generation and
#' some LLM interactions. The Python environment is automatically configured
#' on first use. Required R packages include \pkg{EGAnet} for network analysis
#' and \pkg{reticulate} for Python integration.
#'
#' @author  Lara Lee Russell-Lasalandra, Hudson Golino, Alexander P. Christensen
#' @references
#'
#' Russell-Lasalandra, L. L., Christensen, A. P., & Golino, H. (2025).
#' Generative Psychometrics via AI-GENIE: Automatic Item Generation with Network-Integrated Evaluation.
#' \emph{PsyArXiv}. \url{https://doi.org/10.31234/osf.io/fgbj4}
#'
#' Golino, H., & Christensen, A. P. (2024).
#' \emph{EGAnet: Exploratory Graph Analysis}.
#' R package. \url{https://r-ega.net}
#'
#' Garrido, L., Russell-Lasalandra, L. L., & Golino, H. (2025).
#' Estimating Dimensional Structure in Generative Psychometrics: Comparing PCA and Network Methods Using Large Language Model Item Embeddings.
#' \emph{PsyArXiv} \url{https://doi.org/10.31234/osf.io/2s7pw}
#'
#' Golino, H. (2025).
#' What I Learned with John: On the Depth of Language and How to Measure It with Large Language Models and Algorithm (Kolmogorov) Complexity.
#' \emph{PsyArXiv}. \url{https://doi.org/10.31234/osf.io/b92n5}
#'
#' Golino, H., Garrido, L., & Russell-Lasalandra, L. L. (2026).
#' Optimizing the Landscape of LLM Embeddings with Dynamic Exploratory Graph Analysis for Generative Psychometrics: A Monte Carlo Study.
#' \emph{arXiv}. arXiv:2601.17010. \url{https://doi.org/10.48550/arXiv.2601.17010}
#'
#' Christensen, A. P., & Golino, H. (2021a).
#' Estimating the stability of the number of factors via Bootstrap Exploratory Graph Analysis: A tutorial.
#' \emph{Psych}, \emph{3}(3), 479-500. \cr
#'
#' Christensen, A. P., Garrido, L. E., & Golino, H. (2023).
#' Unique variable analysis: A network psychometrics method to detect local dependence.
#' \emph{Multivariate Behavioral Research}. \cr
#'
#' Golino, H., Moulder, R., Shi, D., Christensen, A. P., Garrido, L. E., Nieto, M. D., Nesselroade, J., Sadana, R., Thiyagarajan, J. A., & Boker, S. M. (2020).
#' Entropy fit indices: New fit measures for assessing the structure and dimensionality of multiple latent variables.
#' \emph{Multivariate Behavioral Research}. \cr
#'
#' Golino, H., Shi, D., Christensen, A. P., Garrido, L. E., Nieto, M. D., Sadana, R., Thiyagarajan, J. A., & Martinez-Molina, A. (2020).
#' Investigating the performance of exploratory graph analysis and traditional techniques to identify the number of latent factors:
#' A simulation and tutorial.
#' \emph{Psychological Methods}, \emph{25}, 292-320. \cr
#'
#' @seealso
#' \code{\link{AIGENIE}} for the main function,
#' \code{\link{GENIE}} for embedding-only analysis,
#' \code{\link{EGAnet}} for the underlying EGA analysis methods.
#'
#' @name AIGENIE-package
NULL
