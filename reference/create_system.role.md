# Create a System Role Prompt for an LLM Item Writer

Constructs a system-level prompt to guide an LLM in behaving like an
expert scale developer. The prompt communicates role identity, domain
expertise, scale context, audience constraints, and response option
considerations.

## Usage

``` r
create_system.role(
  domain,
  scale.title,
  audience,
  response.options,
  system.role
)
```

## Arguments

- domain:

  (Optional) A string indicating the scale's conceptual or applied
  domain (e.g., "clinical psychology", "behavioral economics").

- scale.title:

  (Optional) A string providing the title of the scale (e.g., "Emotion
  Regulation Index").

- audience:

  (Optional) A string specifying the target respondent group (e.g.,
  "adolescents", "working adults").

- response.options:

  (Optional) A character vector of response choices that the LLM should
  consider when phrasing items (e.g.,
  `c("Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree")`).

- system.role:

  (Optional) A custom system prompt provided directly by the user. If
  supplied, it will be used as-is.

## Value

A single character string representing the full system prompt to be
passed to an LLM interface (e.g., OpenAI Chat API). If `system.role` is
not provided, the function dynamically constructs one based on the other
parameters.
