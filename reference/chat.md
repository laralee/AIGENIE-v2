# Chat with an LLM via API Calls

Send one or more prompts to a remote large-language model (LLM) using
the appropriate provider API (OpenAI, Hugging Face, Groq, or Anthropic).
A valid API key for at least one provider is required. To use a local
model (no API call), see
[`local_chat()`](https://laralee.github.io/AIGENIE/reference/local_chat.md).

## Usage

``` r
chat(
  prompts,
  model,
  system.role = NULL,
  openai.API = NULL,
  hf.token = NULL,
  groq.API = NULL,
  anthropic.API = NULL,
  reps = 1,
  top.p = 1,
  temperature = 1,
  max.tokens = 2048L,
  silently = FALSE
)
```

## Arguments

- prompts:

  A character string or character vector. The main prompt(s) given to
  the model. If multiple prompts are supplied, each will be sent
  separately to the model.

- model:

  A character string specifying the LLM model name (e.g., `"gpt4o"`).
  The model must correspond to the API key provided.

- system.role:

  A character string or character vector, default `NULL`. The system
  role(s) (model persona). If only one system role is provided and
  multiple prompts are supplied, the same role will be used for each
  prompt. If multiple system roles are provided, they should align with
  the prompts.

- openai.API:

  A character string, default `NULL`. Your OpenAI API key (required when
  using an OpenAI model).

- hf.token:

  A character string, default `NULL`. Your Hugging Face token (required
  when using a Hugging Face-hosted model).

- groq.API:

  A character string, default `NULL`. Your Groq API key (required when
  using a Groq-hosted model).

- anthropic.API:

  A character string, default `NULL`. Your Anthropic API key (required
  when using an Anthropic model).

- reps:

  Integer, default `1`. The number of times each prompt will be given to
  the model.

- top.p:

  Numeric, default `1`. Top-p (nucleus) sampling parameter.

- temperature:

  Numeric, default `1`. Sampling temperature controlling response
  randomness.

- max.tokens:

  Integer, default `2048L`. Maximum number of tokens requested from the
  model.

- silently:

  Logical, default `FALSE`. If `FALSE`, progress messages are printed.
  If `TRUE`, the function runs quietly.

## Value

A `data.frame` with one row per API call (i.e., per prompt × repetition)
containing:

- `rep` — repetition index

- `prompt` — the prompt text sent to the model

- `response` — the raw text response returned by the model

## Details

The function includes a retry mechanism (up to 5 attempts) for transient
API failures. If all attempts fail, the function stops with an
informative error.

## Important

This function requires a valid API key corresponding to the selected
model. Network access is required. For local (non-API) models, use
[`local_chat()`](https://laralee.github.io/AIGENIE/reference/local_chat.md).

## See also

[`local_chat`](https://laralee.github.io/AIGENIE/reference/local_chat.md)

## Examples

``` r
if (FALSE) { # \dontrun{
#################################################
### Example 1: Writing a Very Basic Prompt  #####
#################################################

# First, define your API key
key <- "INSERT YOUR KEY HERE"
# Then, define your LLM model. This model should correspond to your API key.
model <- "gpt4o" # in this example, you would need an OpenAI API key.

# Then, write your prompt. This will be given to the model directly.
prompt <- "Why does the planet Saturn have rings? Give a 100 word explanation."

# Optionally, add a system role (a model persona)
system.role <- "You specialize in tutoring astronomy for high school students."

# Add the number of prompt repetitions. By default, this is set to 1. But it
# may bu useful to increase the number of repetitions to get a sense of how
# consistent your output might be.
reps <- 3

# Now you are ready to chat with an LLM
first_chat <- chat(
  # Set your own API key. If you are not using OpenAI, change the 1st
  # argument to match your API key. Choices are `hf.token`, `groq.API`,
  # `anthropic.API`, and `openai.API`. In this example, I'm using
  # `openai.API` since I want to chat with a GPT model.
  openai.API = key,
  model = model, # Ensure your model corresponds to your API key.
  prompts = prompt,
  system.role = system.role,
  reps = reps
)

# Check how the output changes from iteration to iteration
first_chat$response[[1]] # first iteration output
first_chat$response[[2]] # second iteration output
first_chat$response[[2]] # third iteration output


####################################################################
### Example 2: Send multiple prompts in a single function call #####
####################################################################

# You are also able to send more than one prompt in a single call
# Let's pull the first prompt from Example 1:
prompt1 <- "Why does the planet Saturn have rings? Give a 100 word explanation."
prompt2 <- "Which planet is the hottest in our solar system? How do we know?"

# Aggregate the prompts in a single object
prompts <- c(prompt1, prompt2)

# Ask the model the questions
second_chat <- chat(
  openai.API = key, # defined in Example 1
  model = model, # defined in Example 1
  prompts = prompts, # NEW
  system.role = system.role, # defined in Example 1
  reps = reps # defined in Example 1
)

# The outputted data frame for this example will have 6 rows
# since the number of prompts (2) times the number of reps (3)
# gives a total of 6 API calls.
second_chat$response[second_chat$prompt==prompt1] # the responses from prompt 1
second_chat$response[second_chat$prompt==prompt2] # the responses from prompt 2


####################################################################
### Example 3: Send multiple prompts with different System Roles ###
####################################################################

# Perhaps your prompts are not related. In that case, you would probably want
# to set a different system role for each prompt.
# Let's change `prompt 2` to be entirely unrelated to astronomy.
prompt2 <- "What is the difference between eukaryotes and prokaryotes? Why?"

# This new second prompt does not fit with the astronomy tutor persona. Let's
# write a persona to match this new prompt topic.
system.role2 <- "You specialize in tutoring biology for middle school students."

# Now, let's combine the system roles into a single object
system.role <- c(system.role, # defined in Example 1: the astronomy tutor
                 system.role2 # defined above: the biology tutor
                 )

# Aggregate our prompts in a single object again
prompts <- c(prompt1, # Asks about Saturn's rings (needs astronomy tutor)
             prompt2 # Asks about types of cells (needs biology tutor)
             )

# Ask the model the questions
third_chat <- chat(
  openai.API = key, # defined in Example 1
  model = model, # defined in Example 1
  prompts = prompts, # NEW
  system.role = system.role, # NEW
  reps = reps # defined in Example 1
)

# View the outputted data frame to examine the responses
View(third_chat)
} # }
```
