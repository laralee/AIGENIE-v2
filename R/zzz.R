# zzz.R
# Python Environment Management for AIGENIE using UV
# ============================================================================

# Define the environment name
.aigenie_env_name <- "aigenie_python_env"

# ============================================================================
# Environment Path Helpers
# ============================================================================

#' Get the Path to AI-GENIE Python Environment
#'
#' @return Character string with the path to the virtual environment
#' @keywords internal
get_aigenie_env_path <- function() {

  # Store in user's home directory for persistence across projects
  env_dir <- file.path(tools::R_user_dir("AIGENIE", which = "data"), .aigenie_env_name)
  return(env_dir)
}

#' Get Python Executable Path Based on OS
#'
#' @param env_path Path to the virtual environment
#' @return Character string with the path to the Python executable
#' @keywords internal
get_python_path <- function(env_path) {
  if (.Platform$OS.type == "windows") {
    file.path(env_path, "Scripts", "python.exe")
  } else {
    file.path(env_path, "bin", "python")
  }
}

# ============================================================================
# UV Installation Check
# ============================================================================

#' Check if UV is Available
#'
#' @return TRUE if UV is available, otherwise stops with an error
#' @keywords internal
check_uv_available <- function() {
  # 1) Try Sys.which()
  uv_path <- Sys.which("uv")
  uv_path <- as.character(uv_path)[1]

  # 2) If empty, try common install locations
  if (!nzchar(uv_path)) {
    possible <- c("~/.local/bin/uv", "/usr/local/bin/uv", "/opt/homebrew/bin/uv", "/usr/bin/uv")
    possible <- normalizePath(possible, mustWork = FALSE)
    found <- possible[file.exists(possible)]
    if (length(found) > 0) uv_path <- found[1]
  }

  # 3) If still empty, try asking a login shell (best-effort)
  if (!nzchar(uv_path)) {
    try_shell <- tryCatch(
      system2("bash", c("-lc", "which uv"), stdout = TRUE, stderr = FALSE),
      error = function(e) character(0)
    )
    if (length(try_shell) && nzchar(try_shell[1])) {
      uv_path <- try_shell[1]
    }
  }

  # 4) Final check by invoking --version
  if (nzchar(uv_path) && file.exists(uv_path)) {
    ok <- FALSE
    try({
      out <- system2(uv_path, "--version", stdout = TRUE, stderr = TRUE)
      ok <- length(out) > 0 && nzchar(out[1])
    }, silent = TRUE)
    if (ok) return(invisible(TRUE))
  }

  stop(
    "UV is not installed or not found in PATH.\n",
    "Please install UV first: https://docs.astral.sh/uv/getting-started/installation/\n",
    "  - On macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh\n",
    "  - On Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"\n",
    "  - After installation, resart your R session.",
    "Troubleshooting: ensure the directory containing uv (e.g. ~/.local/bin) is on R's PATH or add it to ~/.Renviron.\n\n",
    "Run `python_env_info()` to check the status of your UV installation.")
}


# ============================================================================
# Environment Creation
# ============================================================================

#' Create the AI-GENIE Python Virtual Environment
#'
#' @param env_path Path where the environment should be created
#' @return TRUE invisibly on success
#' @keywords internal
create_aigenie_env <- function(env_path) {
  if (!dir.exists(env_path)) {
    message("Creating AI-GENIE Python environment with UV...")

    # Create parent directory if needed
    dir.create(dirname(env_path), recursive = TRUE, showWarnings = FALSE)

    # Create virtual environment
    result <- system2("uv", args = c("venv", shQuote(env_path)),
                      stdout = TRUE, stderr = TRUE)

    if (!dir.exists(env_path)) {
      stop("Failed to create virtual environment: ", paste(result, collapse = "\n"),
           call. = FALSE)
    }
    message("Virtual environment created at: ", env_path)
  }
  invisible(TRUE)
}

# ============================================================================
# Package Installation
# ============================================================================

#' Get Core Python Packages Required by AI-GENIE
#'
#' @return Character vector of package specifications
#' @keywords internal
get_core_packages <- function() {
  c(
    "openai==0.28",
    "groq",
    "requests",
    "'numpy<2.0'"
  )
}

#' Get HuggingFace-Related Python Packages
#'
#' @return Character vector of package specifications
#' @keywords internal
get_huggingface_packages <- function() {
  c(
    "transformers",
    "huggingface_hub",
    "sentence-transformers",
    "tokenizers",
    "datasets",
    "accelerate",
    "safetensors"
  )
}

#' Get Local LLM Support Packages
#'
#' @return Character vector of package specifications
#' @keywords internal
get_local_llm_packages <- function() {

  c("llama-cpp-python")
}

#' Install AI-GENIE Python Packages Using UV
#'
#' @param env_path Path to the virtual environment
#' @param include_huggingface Logical. Include HuggingFace packages?
#' @param include_local_llm Logical. Include local LLM (llama-cpp) support?
#' @param gpu Logical. Install GPU-enabled PyTorch?
#' @return TRUE invisibly on success
#' @keywords internal
install_aigenie_packages <- function(env_path,
                                      include_huggingface = TRUE,
                                      include_local_llm = FALSE,
                                      gpu = FALSE) {
  python_path <- get_python_path(env_path)

  # Start with core packages
  packages_to_install <- get_core_packages()

  message("Installing core Python packages...")

  # Install main packages
  result <- system2("uv",
                    args = c("pip", "install",
                             "--python", shQuote(python_path),
                             packages_to_install),
                    stdout = TRUE, stderr = TRUE)

  exit_status <- attr(result, "status")
  if (!is.null(exit_status) && exit_status != 0) {
    stop("Failed to install core packages: ", paste(result, collapse = "\n"),
         call. = FALSE)
  }

  # Install HuggingFace packages if requested
  if (include_huggingface) {
    message("Installing HuggingFace packages...")

    hf_packages <- get_huggingface_packages()
    hf_result <- system2("uv",
                         args = c("pip", "install",
                                  "--python", shQuote(python_path),
                                  hf_packages),
                         stdout = TRUE, stderr = TRUE)

    exit_status <- attr(hf_result, "status")
    if (!is.null(exit_status) && exit_status != 0) {
      warning("Some HuggingFace packages may not have installed correctly: ",
              paste(hf_result, collapse = "\n"))
    }

    # Install PyTorch
    message("Installing PyTorch (", ifelse(gpu, "GPU", "CPU"), " version)...")

    if (gpu) {
      torch_result <- system2("uv",
                              args = c("pip", "install",
                                       "--python", shQuote(python_path),
                                       "torch", "torchvision", "torchaudio"),
                              stdout = TRUE, stderr = TRUE)
    } else {
      torch_result <- system2("uv",
                              args = c("pip", "install",
                                       "--python", shQuote(python_path),
                                       "--index-url", "https://download.pytorch.org/whl/cpu",
                                       "torch", "torchvision", "torchaudio"),
                              stdout = TRUE, stderr = TRUE)
    }

    exit_status <- attr(torch_result, "status")
    if (!is.null(exit_status) && exit_status != 0) {
      warning("PyTorch installation may have issues: ", paste(torch_result, collapse = "\n"))
    }
  }

  # Install local LLM packages if requested
  if (include_local_llm) {
    message("Installing local LLM support (llama-cpp-python)...")

    # Check for Apple Silicon
    sys_info <- Sys.info()
    if (sys_info["sysname"] == "Darwin" && grepl("arm64|aarch64", sys_info["machine"])) {
      Sys.setenv(CMAKE_ARGS = "-DLLAMA_METAL=on")
    }

    llm_result <- system2("uv",
                          args = c("pip", "install",
                                   "--python", shQuote(python_path),
                                   get_local_llm_packages()),
                          stdout = TRUE, stderr = TRUE)

    Sys.unsetenv("CMAKE_ARGS")

    exit_status <- attr(llm_result, "status")
    if (!is.null(exit_status) && exit_status != 0) {
      warning("llama-cpp-python installation may have issues: ",
              paste(llm_result, collapse = "\n"))
    }
  }

  message("Python packages installed successfully!")
  invisible(TRUE)
}

# ============================================================================
# Main Environment Setup Function
# ============================================================================

#' Ensure AI-GENIE Python Environment is Ready
#'
#' @description
#' Sets up the Python environment with all required dependencies for AI-GENIE.
#' Uses UV for fast, reliable package management. This function is called
#' automatically when needed, but can also be called directly.
#'
#' @param force_reinstall Logical. Force complete reinstallation?
#' @param include_huggingface Logical. Include HuggingFace packages? Default TRUE.
#' @param include_local_llm Logical. Include local LLM support? Default FALSE.
#' @param gpu Logical. Install GPU-enabled PyTorch? Default FALSE.
#'
#' @return TRUE invisibly on success
#' @export
ensure_aigenie_python <- function(force_reinstall = FALSE,
                                   include_huggingface = TRUE,
                                   include_local_llm = FALSE,
                                   gpu = FALSE) {

  # Check if already initialized in this session
  if (isTRUE(getOption("aigenie.python_initialized", FALSE)) && !force_reinstall) {
    return(invisible(TRUE))
  }

  message("Setting up AI-GENIE Python environment...")

  # Check UV is available
  check_uv_available()

  # Get environment paths

  env_path <- get_aigenie_env_path()
  python_path <- get_python_path(env_path)

  # Create environment if needed
  env_exists <- dir.exists(env_path) && file.exists(python_path)

  if (!env_exists || force_reinstall) {
    if (force_reinstall && dir.exists(env_path)) {
      message("Removing existing environment for reinstall...")
      unlink(env_path, recursive = TRUE)
    }
    create_aigenie_env(env_path)
    install_aigenie_packages(env_path,
                             include_huggingface = include_huggingface,
                             include_local_llm = include_local_llm,
                             gpu = gpu)
  }

  # Configure reticulate to use our environment
  reticulate::use_python(python_path, required = TRUE)

  # Initialize Python
  reticulate::py_available(initialize = TRUE)

  # Verify core installations
  tryCatch({
    openai <- reticulate::import("openai")
    groq <- reticulate::import("groq")
    requests <- reticulate::import("requests")
    message("Core packages ready!")

    # Verify HuggingFace if installed
    if (include_huggingface && reticulate::py_module_available("transformers")) {
      transformers <- reticulate::import("transformers")
      message("HuggingFace packages ready!")
    }

    message("AI-GENIE: Python environment ready!")
  }, error = function(e) {
    if (!force_reinstall) {
      message("Package verification failed. Attempting reinstall...")
      return(ensure_aigenie_python(force_reinstall = TRUE,
                                    include_huggingface = include_huggingface,
                                    include_local_llm = include_local_llm,
                                    gpu = gpu))
    }
    stop("Failed to set up Python packages: ", e$message,
         "\nPlease try restarting R or run: AIGENIE::reinstall_python_env()",
         call. = FALSE)
  })

  # Mark as initialized for this session
  options(aigenie.python_initialized = TRUE)
  invisible(TRUE)
}

#' Ensure Python Environment for Local Models (Alias)
#'
#' @description
#' Legacy compatibility function. Now calls ensure_aigenie_python with
#' appropriate settings for local model support.
#'
#' @param force Logical. Force reinstallation?
#' @param silently Logical. Suppress messages?
#'
#' @return TRUE invisibly on success
#' @keywords internal
ensure_aigenie_python_local <- function(force = FALSE, silently = FALSE) {
  if (silently) {
    suppressMessages(
      ensure_aigenie_python(force_reinstall = force,
                            include_huggingface = TRUE,
                            include_local_llm = FALSE)
    )
  } else {
    ensure_aigenie_python(force_reinstall = force,
                          include_huggingface = TRUE,
                          include_local_llm = FALSE)
  }
}

# ============================================================================
# User-Facing Environment Management Functions
# ============================================================================

#' Reinstall AI-GENIE Python Environment
#'
#' @description
#' Removes and recreates the Python virtual environment with all required
#' dependencies. Use this function if you encounter Python-related errors,
#' want to update Python packages, or need to change the environment configuration.
#'
#' AIGENIE uses UV (\url{https://docs.astral.sh/uv/}) for fast, reliable
#' Python environment management.
#'
#' @param include_huggingface Logical. Include HuggingFace packages (transformers,
#'   sentence-transformers, torch). Required for local embeddings with HuggingFace
#'   models. Default \code{TRUE}.
#' @param include_local_llm Logical. Include llama-cpp-python for running local
#'   GGUF models with \code{\link{local_AIGENIE}}. Default \code{FALSE}.
#' @param gpu Logical. Install GPU-enabled PyTorch. Requires CUDA-compatible
#'   NVIDIA GPU and proper driver installation. Default \code{FALSE}.
#'
#' @return Invisible \code{TRUE} on success.
#'
#' @examples
#' \dontrun{
#' # Fix Python environment issues
#' reinstall_python_env()
#'
#' # Reinstall with GPU support
#' reinstall_python_env(gpu = TRUE)
#'
#' # Minimal install (API-only, no HuggingFace - faster)
#' reinstall_python_env(include_huggingface = FALSE)
#'
#' # Full install with local LLM support
#' reinstall_python_env(include_huggingface = TRUE, include_local_llm = TRUE)
#' }
#'
#' @seealso
#' \code{\link{python_env_info}} to check environment status,
#' \code{\link{install_gpu_support}} for GPU setup,
#' \code{\link{install_local_llm_support}} for local model setup.
#'
#' @export
reinstall_python_env <- function(include_huggingface = TRUE,
                                  include_local_llm = FALSE,
                                  gpu = FALSE) {
  options(aigenie.python_initialized = FALSE)
  ensure_aigenie_python(force_reinstall = TRUE,
                        include_huggingface = include_huggingface,
                        include_local_llm = include_local_llm,
                        gpu = gpu)
}

#' Install GPU Support for AI-GENIE
#'
#' @description
#' Reinstalls the Python environment with GPU-enabled PyTorch for faster
#' inference with local embedding models. Requires a CUDA-compatible NVIDIA
#' GPU and proper driver installation.
#'
#' @details
#' This function:
#' \enumerate{
#'   \item Removes the existing Python environment
#'   \item Creates a new environment with GPU-enabled PyTorch
#'   \item Installs all HuggingFace dependencies
#' }
#'
#' On Apple Silicon Macs, MPS (Metal Performance Shaders) acceleration is
#' used automatically without needing this function.
#'
#' @return Invisible \code{TRUE} on success.
#'
#' @examples
#' \dontrun{
#' # Enable GPU acceleration (requires NVIDIA GPU + CUDA)
#' install_gpu_support()
#' }
#'
#' @seealso
#' \code{\link{reinstall_python_env}},
#' \code{\link{python_env_info}}.
#'
#' @export
install_gpu_support <- function() {
  message("Installing GPU-enabled packages. This requires CUDA drivers.")
  reinstall_python_env(include_huggingface = TRUE, gpu = TRUE)
}

#' Install Local LLM Support
#'
#' @description
#' Installs llama-cpp-python for running local GGUF models with
#' \code{\link{local_AIGENIE}}. On Apple Silicon Macs, this includes
#' Metal acceleration support for fast inference.
#'
#' @details
#' After installation, you can use any GGUF model file with \code{local_AIGENIE()}.
#' Download GGUF models from HuggingFace (search for "GGUF" format).
#'
#' Popular model recommendations:
#' \itemize{
#'   \item \strong{Llama 3 8B}: Good balance of quality and speed
#'   \item \strong{Mistral 7B}: Fast with good quality
#'   \item \strong{Qwen 2.5}: Strong multilingual support
#' }
#'
#' @return Invisible \code{TRUE} on success.
#'
#' @examples
#' \dontrun{
#' # Install local LLM support
#' install_local_llm_support()
#'
#' # Then download a GGUF model and use with local_AIGENIE
#' results <- local_AIGENIE(
#'   item.attributes = my_traits,
#'   model.path = "~/models/llama-3-8b.Q4_K_M.gguf",
#'   embedding.model = "bert-base-uncased"
#' )
#' }
#'
#' @seealso
#' \code{\link{local_AIGENIE}},
#' \code{\link{reinstall_python_env}}.
#'
#' @export
install_local_llm_support <- function() {
  message("Installing local LLM support (llama-cpp-python)...")
  options(aigenie.python_initialized = FALSE)
  ensure_aigenie_python(force_reinstall = TRUE,
                        include_huggingface = TRUE,
                        include_local_llm = TRUE,
                        gpu = FALSE)
}

#' Get AI-GENIE Python Environment Info
#'
#' @description
#' Returns diagnostic information about the AIGENIE Python environment,
#' including paths, installation status, and installed packages.
#' Useful for troubleshooting Python-related issues.
#'
#' @return A list with the following elements:
#'   \item{env_path}{Path to the virtual environment directory.}
#'   \item{python_path}{Path to the Python executable.}
#'   \item{env_exists}{Logical. Whether the environment directory exists.}
#'   \item{python_exists}{Logical. Whether the Python executable exists.}
#'   \item{initialized}{Logical. Whether Python has been initialized this session.}
#'   \item{uv_available}{Logical. Whether UV is installed and accessible.}
#'   \item{installed_packages}{Character vector of installed packages (if environment exists).}
#'
#' @examples
#' \dontrun{
#' # Check environment status
#' info <- python_env_info()
#'
#' # Is the environment set up?
#' info$env_exists
#'
#' # What packages are installed?
#' cat(info$installed_packages, sep = "\n")
#'
#' # Is UV available?
#' info$uv_available
#' }
#'
#' @seealso
#' \code{\link{reinstall_python_env}} to fix environment issues.
#'
#' @export
python_env_info <- function() {
  env_path <- get_aigenie_env_path()
  python_path <- get_python_path(env_path)

  info <- list(
    env_path = env_path,
    python_path = python_path,
    env_exists = dir.exists(env_path),
    python_exists = file.exists(python_path),
    initialized = isTRUE(getOption("aigenie.python_initialized", FALSE)),
    uv_available = tryCatch({
      length(system("uv --version", intern = TRUE, ignore.stderr = TRUE)) > 0
    }, error = function(e) FALSE)
  )

  # Check installed packages if environment exists
  if (info$python_exists) {
    tryCatch({
      result <- system2("uv",
                        args = c("pip", "list", "--python", shQuote(python_path)),
                        stdout = TRUE, stderr = TRUE)
      info$installed_packages <- result
    }, error = function(e) {
      info$installed_packages <- "Unable to list packages"
    })
  }

  info
}

# ============================================================================
# HuggingFace Token Management
# ============================================================================

#' Set Hugging Face Token
#'
#' @description
#' Configure your HuggingFace API token for accessing gated models like
#' Google's EmbeddingGemma or other restricted models.
#'
#' @param token Character. Your HuggingFace API token from
#'   \url{https://huggingface.co/settings/tokens}.
#' @param save Logical. If \code{TRUE} (default), saves the token to the
#'   HuggingFace cache for future sessions. If \code{FALSE}, sets it only
#'   for the current R session.
#'
#' @details
#' Some embedding models on HuggingFace require authentication:
#' \itemize{
#'   \item \code{google/embeddinggemma-300m}
#'   \item Other gated models
#' }
#'
#' Before using these models, you must:
#' \enumerate{
#'   \item Create an account at \url{https://huggingface.co}
#'   \item Accept the model's license on its model page
#'   \item Generate an access token at \url{https://huggingface.co/settings/tokens}
#'   \item Call this function with your token
#' }
#'
#' @return Invisible \code{TRUE} on success.
#'
#' @examples
#' \dontrun{
#' # Set token (saved permanently for future sessions)
#' set_huggingface_token("hf_xxxxxxxxxxxxxxxxx")
#'
#' # Set token for this session only (not saved)
#' set_huggingface_token("hf_xxxxxxxxxxxxxxxxx", save = FALSE)
#'
#' # Now you can use gated HuggingFace models
#' results <- GENIE(
#'   items = my_items,
#'   embedding.model = "BAAI/bge-large-en-v1.5",
#'   hf.token = "hf_xxxxxxxxxxxxxxxxx"
#' )
#' }
#'
#' @export
set_huggingface_token <- function(token, save = TRUE) {
  ensure_aigenie_python()

  huggingface_hub <- reticulate::import("huggingface_hub")

  if (save) {
    huggingface_hub$login(token = token)
    message("Hugging Face token saved. You won't need to set it again.")
  } else {
    Sys.setenv(HUGGINGFACE_TOKEN = token)
    Sys.setenv(HF_TOKEN = token)
    message("Hugging Face token set for this session only.")
  }

  invisible(TRUE)
}

# ============================================================================
# Package Load/Unload Hooks
# ============================================================================

#' @keywords internal
.onLoad <- function(libname, pkgname) {
  # Set initialization flags to FALSE
  options(aigenie.python_initialized = FALSE)
  options(aigenie.python_local_initialized = FALSE)

  # Check for old conda environment and suggest cleanup
  tryCatch({
    if (requireNamespace("reticulate", quietly = TRUE)) {
      if (reticulate::condaenv_exists("AIGENIE_python_env")) {
        packageStartupMessage(
          "Note: Old conda environment 'AIGENIE_python_env' detected.\n",
          "You can remove it to save space with: reticulate::conda_remove('AIGENIE_python_env')\n",
          "AI-GENIE now uses UV for faster, lighter Python environment management."
        )
      }
    }
  }, error = function(e) {
    # Silently ignore errors during load
  })

  packageStartupMessage(
    "AI-GENIE loaded. Python dependencies will be configured on first use.\n",
    "For GPU support, run: AIGENIE::install_gpu_support()\n",
    "For local LLM support, run: AIGENIE::install_local_llm_support()"
  )
}

#' @keywords internal
.onUnload <- function(libpath) {
  options(aigenie.python_initialized = NULL)
  options(aigenie.python_local_initialized = NULL)
}
