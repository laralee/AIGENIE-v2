#' @title GPT-5.4 Example Item Pool
#'
#' @description
#' An example item pool generated with GPT-5.4 for demonstrating the
#' psychometric reduction workflow implemented in \code{\link{GENIE}}.
#' The data contain 180 personality items: 90 conscientiousness items
#' and 90 openness items.
#'
#' @name items.gpt5.4.example
#'
#' @docType data
#'
#' @usage data("items.gpt5.4.example")
#'
#' @format A data frame with 180 rows and 4 variables:
#' \describe{
#'   \item{\code{ID}}{Unique item identifier.}
#'   \item{\code{statement}}{The generated item statement.}
#'   \item{\code{type}}{Higher-order item type: conscientiousness or openness.}
#'   \item{\code{attribute}}{Target attribute represented by the item.}
#' }
#'
#' @details
#' The item pool is the GPT-5.4 example used to illustrate AI-GENIE/GENIE
#' item reduction. Conscientiousness items represent self-efficacy,
#' achievement-striving, and perseverance. Openness items represent
#' introspection, aesthetics, and abstract-thinking.
#'
#' The corresponding embedding matrix is available as
#' \code{\link{embeddings.gpt5.4.example}}.
#'
#' @keywords datasets
#'
#' @seealso
#' \code{\link{embeddings.gpt5.4.example}}, \code{\link{GENIE}}
#'
#' @examples
#' data("items.gpt5.4.example")
#'
#' dim(items.gpt5.4.example)
#' head(items.gpt5.4.example)
#' table(items.gpt5.4.example$type)
#' table(items.gpt5.4.example$attribute)
#'
NULL
