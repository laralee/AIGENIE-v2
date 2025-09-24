# zzz.R

# Helper function to ensure Python environment is set up
ensure_aigenie_python <- function() {
  # Check if we've already initialized in this session
  if (isTRUE(getOption("aigenie.python_initialized", FALSE))) {
    return(invisible(TRUE))
  }

  message("AI-GENIE: Setting up Python environment (one-time setup)...")

  # Configure Python version
  reticulate::py_config()

  # Check and install packages
  if (!reticulate::py_module_available("openai")) {
    message("Installing openai==0.28...")
    reticulate::py_install("openai==0.28", pip = TRUE)
  }

  if (!reticulate::py_module_available("groq")) {
    message("Installing groq...")
    reticulate::py_install("groq", pip = TRUE)
  }

  if (!reticulate::py_module_available("requests")) {
    message("Installing requests...")
    reticulate::py_install("requests", pip = TRUE)
  }

  # Initialize Python
  reticulate::py_available(initialize = TRUE)

  # Verify installations
  tryCatch({
    openai <- reticulate::import("openai")
    groq <- reticulate::import("groq")
    requests <- reticulate::import("requests")
    message("AI-GENIE: Python environment ready!")
  }, error = function(e) {
    stop("Failed to set up Python packages. Please try restarting R and reinstalling.")
  })

  # Mark as initialized for this session
  options(aigenie.python_initialized = TRUE)

  invisible(TRUE)
}

# Package load hook
.onLoad <- function(libname, pkgname) {
  # Set the initialization flag to FALSE
  options(aigenie.python_initialized = FALSE)

  # Optional: Check for old conda environment and suggest cleanup
  if (reticulate::condaenv_exists("AIGENIE_python_env")) {
    packageStartupMessage(
      "Note: Old conda environment 'AIGENIE_python_env' detected.\n",
      "You can remove it to save space with: reticulate::conda_remove('AIGENIE_python_env')"
    )
  }

  # Don't initialize Python here - wait until it's actually needed
  packageStartupMessage("AI-GENIE loaded. Python dependencies will be configured on first use.")
}

# Optional: Package unload hook to clean up
.onUnload <- function(libpath) {
  # Reset the initialization flag
  options(aigenie.python_initialized = NULL)
}
