
# Version control for openai - MUST be version 0.28

.onLoad <- function(libname, pkgname) {

  venv_name <- "AIGENIE_python_env"

  # Check if the virtual environment already exists, otherwise create it
  if (!reticulate::condaenv_exists(venv_name)) {
    message("Welcome to AI-GENIE! You need a virtual environment to access Python through R. Once installed, this will not need to be done again. Creating the virtual environment...")
    reticulate::conda_create(envname = venv_name, python_version = 3.11)
  }

  # Use the virtual environment
  reticulate::use_condaenv(venv_name, required = TRUE)

  # python packages and versions required
  python_packages <- list(
    groq = NULL, # no specific version requirement for groq
    openai = "0.28" # specific version for openai
  )

  # check if each python package is available already in the correct version, and install the correct version if not
  for (pkg in names(python_packages)) {
    if (!reticulate::py_module_available(pkg)) {
      message(sprintf("Installing Python package '%s' in virtual environment...", pkg))
      if (is.null(python_packages[[pkg]])) {
        reticulate::py_install(pkg, envname = venv_name)
      } else {
        reticulate::py_install(paste(pkg, "==", python_packages[[pkg]], sep = ""), envname = venv_name)
      }
    }
  }
}

