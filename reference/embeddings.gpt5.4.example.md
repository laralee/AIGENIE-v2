# GPT-5.4 Example Item Embeddings

A numeric embedding matrix corresponding to
[`items.gpt5.4.example`](https://laralee.github.io/AIGENIE/reference/items.gpt5.4.example.md),
provided for demonstrating
[`GENIE`](https://laralee.github.io/AIGENIE/reference/GENIE.md) without
requiring an external embedding API call.

## Usage

``` r
data("embeddings.gpt5.4.example")
```

## Format

A 1536 x 180 numeric matrix. Rows are embedding dimensions and columns
are items. Column names correspond to the item IDs in
`items.gpt5.4.example`.

## Details

The embeddings were generated from the GPT-5.4 example item pool using
OpenAI's `text-embedding-3-small` embedding model. The matrix is
oriented in the format expected by
[`GENIE`](https://laralee.github.io/AIGENIE/reference/GENIE.md):
embedding dimensions in rows and items in columns.

The corresponding item metadata are available as
[`items.gpt5.4.example`](https://laralee.github.io/AIGENIE/reference/items.gpt5.4.example.md).

## See also

[`items.gpt5.4.example`](https://laralee.github.io/AIGENIE/reference/items.gpt5.4.example.md),
[`GENIE`](https://laralee.github.io/AIGENIE/reference/GENIE.md)

## Examples

``` r
data("embeddings.gpt5.4.example")

dim(embeddings.gpt5.4.example)
#> [1] 1536  180

data("items.gpt5.4.example")
all(
  colnames(embeddings.gpt5.4.example) %in%
    as.character(items.gpt5.4.example$ID)
)
#> [1] TRUE
```
