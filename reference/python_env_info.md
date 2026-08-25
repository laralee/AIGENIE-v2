# Get AI-GENIE Python Environment Info

Returns diagnostic information about the AIGENIE Python environment,
including paths, installation status, and installed packages. Useful for
troubleshooting Python-related issues.

## Usage

``` r
python_env_info()
```

## Value

A list with the following elements:

- env_path:

  Path to the virtual environment directory.

- python_path:

  Path to the Python executable.

- env_exists:

  Logical. Whether the environment directory exists.

- python_exists:

  Logical. Whether the Python executable exists.

- initialized:

  Logical. Whether Python has been initialized this session.

- uv_available:

  Logical. Whether UV is installed and accessible.

- installed_packages:

  Character vector of installed packages (if environment exists).

## See also

[`reinstall_python_env`](https://laralee.github.io/AIGENIE/reference/reinstall_python_env.md)
to fix environment issues.

## Examples

``` r
if (FALSE) { # \dontrun{
# Check environment status
info <- python_env_info()

# Is the environment set up?
info$env_exists

# What packages are installed?
cat(info$installed_packages, sep = "\n")

# Is UV available?
info$uv_available
} # }
```
