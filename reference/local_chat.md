# Chat with a local LLM (no API calls)

Send one or more prompts to a locally available large-language model
(LLM) without making remote API calls. The local model must be
installed/available on the machine as a local model directory. This
function is intended for fully local inference (no API key required).

## Usage

``` r
local_chat(
  prompts,
  model.path,
  n.ctx = 4096,
  n.gpu.layers = -1,
  max.tokens = 1024,
  system.role = NULL,
  reps = 1,
  temperature = 1,
  top.p = 1,
  silently = FALSE
)
```

## Arguments

- prompts:

  A character string or character vector. The main prompt(s) given to
  the model. If multiple prompts are supplied, each will be sent
  separately to the model.

- model.path:

  A character string. Path for the local model file. The function does
  not download models; ensure the model is present locally before using
  this function.

- n.ctx:

  Integer, default `4096`. The context window (number of tokens)
  available to the model for a single generation.

- n.gpu.layers:

  Integer, default `-1`. Number of model layers to place on GPU (if
  supported). Use `-1` to let the runtime choose automatically.

- max.tokens:

  Integer, default `1024L`. Maximum number of tokens requested from the
  local model for a single generation.

- system.role:

  A character string or character vector, default `NULL`. The system
  role(s) (model persona). If only one system role is provided and
  multiple prompts are supplied, the same role will be used for each
  prompt. If multiple system roles are provided, they should align with
  the prompts.

- reps:

  Integer, default `1`. The number of times each prompt will be given to
  the model (independent generations).

- temperature:

  Numeric, default `1`. Sampling temperature controlling response
  randomness.

- top.p:

  Numeric, default `1`. Top-p (nucleus) sampling parameter.

- silently:

  Logical, default `FALSE`. If `FALSE`, progress messages are printed to
  the console. If `TRUE`, the function runs quietly.

## Value

A `data.frame` with one row per generation (i.e., per prompt ×
repetition) containing:

- `rep` — repetition index

- `prompt` — the prompt text sent to the model

- `response` — the raw text response returned by the model

## Details

Before running this function check that you are able to run the local
environment via
[`check_local_llm_setup()`](https://laralee.github.io/AIGENIE/reference/check_local_llm_setup.md).

For each prompt × repetition the function constructs a `full_prompt`
that includes the `system.role` and the user prompt, sets a
deterministic generation seed per generation, and calls the local model.
A retry loop (up to 5 attempts, with brief waits) handles transient
failures; if all attempts fail the function aborts with an informative
error.

## Important / Warnings

- Local model inference can be resource intensive. Large models may
  require substantial disk space, RAM, and (optionally) GPU support.
  Performance and feasibility depend on model size and hardware.

- `model.path` must point to a model already present; this function will
  not download remote models.

## See also

[`chat`](https://laralee.github.io/AIGENIE/reference/chat.md)

## Examples

``` r
if (FALSE) { # \dontrun{
#################################################
### Example 1: Writing a Very Basic Prompt  #####
#################################################

# For local_chat you do NOT need an API key, but you DO need a text generation
# model available locally.
model <- "path/to/local-model" # replace with your local model path

# Then, write your prompt. This will be given to the model directly.
prompt <- "Why does the planet Saturn have rings? Give a 100 word explanation."

# Optionally, add a system role (a model persona)
system.role <- "You specialize in tutoring astronomy for high school students."

# Add the number of prompt repetitions. By default, this is set to 1. But it
# may be useful to increase the number of repetitions to get a sense of how
# consistent your output might be.
reps <- 3

# Now you are ready to chat with the local model
first_chat <- local_chat(
  model.path = model, # local model identifier or path
  prompts = prompt,
  system.role = system.role,
  reps = reps
)

# Check how the output changes from iteration to iteration
first_chat$response[[1]] # first iteration output
first_chat$response[[2]] # second iteration output
first_chat$response[[3]] # third iteration output


####################################################################
### Example 2: Send multiple prompts in a single function call #####
####################################################################

# You are able to send more than one prompt in a single call
prompt1 <- "Why does the planet Saturn have rings? Give a 100 word explanation."
prompt2 <- "Which planet is the hottest in our solar system? How do we know?"

# Aggregate the prompts in a single object
prompts <- c(prompt1, prompt2)

# Ask the model the questions
second_chat <- local_chat(
  model.path = model, # defined above
  prompts = prompts, # NEW
  system.role = system.role, # defined above
  reps = reps # defined above
)

# The outputted data frame for this example will have 6 rows
# since the number of prompts (2) times the number of reps (3)
# gives a total of 6 generations.
second_chat$response[second_chat$prompt == prompt1] # the responses from prompt 1
second_chat$response[second_chat$prompt == prompt2] # the responses from prompt 2


####################################################################
### Example 3: Send multiple prompts with different System Roles ###
####################################################################

# Perhaps your prompts are not related. In that case, you would probably want
# to set a different system role for each prompt.
prompt2 <- "What is the difference between eukaryotes and prokaryotes? Why?"

# This new second prompt does not fit with the astronomy tutor persona. Let's
# write a persona to match this new prompt topic.
system.role2 <- "You specialize in tutoring biology for middle school students."

# Now, let's combine the system roles into a single object
system.role <- c(system.role, # defined earlier: the astronomy tutor
                 system.role2 # defined above: the biology tutor
                 )

# Aggregate our prompts in a single object again
prompts <- c(prompt1, # Asks about Saturn's rings (needs astronomy tutor)
             prompt2 # Asks about types of cells (needs biology tutor)
             )

# Ask the model the questions
third_chat <- local_chat(
  model.path = model,
  prompts = prompts,
  system.role = system.role,
  reps = reps
)

# View the outputted data frame to examine the responses
View(third_chat)
} # }
```
