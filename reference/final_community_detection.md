# Run Final Community Detection with EGA

Run Final Community Detection with EGA

## Usage

``` r
final_community_detection(
  embedding_matrix,
  true_communities,
  model = "glasso",
  algorithm = "walktrap",
  uni.method = "louvain",
  corr = "auto"
)
```

## Arguments

- embedding_matrix:

  A numeric matrix with items as columns.

- true_communities:

  Named list mapping items to known communities.

- model:

  Network estimation model (e.g., "glasso", "TMFG").

- algorithm:

  Community detection algorithm (e.g., "walktrap").

- uni.method:

  Unidimensionality method passed to EGA.

- corr:

  Character. Correlation method. Default "auto" uses EGAnet's automatic
  detection.

## Value

A list with final communities, final NMI, dropped items, EGA object, and
success flag.
