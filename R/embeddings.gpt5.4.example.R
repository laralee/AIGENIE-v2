#' @title GPT-5.4 Example Item Embeddings
#'
#' @description
#' A numeric embedding matrix corresponding to
#' \code{\link{items.gpt5.4.example}}, provided for demonstrating
#' \code{\link{GENIE}} without requiring an external embedding API call.
#'
#' @name embeddings.gpt5.4.example
#'
#' @docType data
#'
#' @usage data("embeddings.gpt5.4.example")
#'
#' @format
#' A 1536 x 180 numeric matrix. Rows are embedding dimensions and columns
#' are items. Column names correspond to the item IDs in
#' \code{items.gpt5.4.example}.
#'
#' @details
#' The embeddings were generated from the GPT-5.4 example item pool using
#' OpenAI's \code{text-embedding-3-small} embedding model. The matrix is
#' oriented in the format expected by \code{\link{GENIE}}: embedding
#' dimensions in rows and items in columns.
#'
#' The corresponding item metadata are available as
#' \code{\link{items.gpt5.4.example}}.
#'
#' @keywords datasets
#'
#' @seealso
#' \code{\link{items.gpt5.4.example}}, \code{\link{GENIE}}
#'
#' @examples
#' data("embeddings.gpt5.4.example")
#'
#' dim(embeddings.gpt5.4.example)
#'
#' data("items.gpt5.4.example")
#' all(
#'   colnames(embeddings.gpt5.4.example) %in%
#'     as.character(items.gpt5.4.example$ID)
#' )
#'
NULL
