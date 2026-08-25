# GPT-5.4 Example Item Pool

An example item pool generated with GPT-5.4 for demonstrating the
psychometric reduction workflow implemented in
[`GENIE`](https://laralee.github.io/AIGENIE/reference/GENIE.md). The
data contain 180 personality items: 90 conscientiousness items and 90
openness items.

## Usage

``` r
data("items.gpt5.4.example")
```

## Format

A data frame with 180 rows and 4 variables:

- `ID`:

  Unique item identifier.

- `statement`:

  The generated item statement.

- `type`:

  Higher-order item type: conscientiousness or openness.

- `attribute`:

  Target attribute represented by the item.

## Details

The item pool is the GPT-5.4 example used to illustrate AI-GENIE/GENIE
item reduction. Conscientiousness items represent self-efficacy,
achievement-striving, and perseverance. Openness items represent
introspection, aesthetics, and abstract-thinking.

The corresponding embedding matrix is available as
[`embeddings.gpt5.4.example`](https://laralee.github.io/AIGENIE/reference/embeddings.gpt5.4.example.md).

## See also

[`embeddings.gpt5.4.example`](https://laralee.github.io/AIGENIE/reference/embeddings.gpt5.4.example.md),
[`GENIE`](https://laralee.github.io/AIGENIE/reference/GENIE.md)

## Examples

``` r
data("items.gpt5.4.example")

dim(items.gpt5.4.example)
#> [1] 180   4
head(items.gpt5.4.example)
#>                type            attribute
#> 1 conscientiousness        self-efficacy
#> 2 conscientiousness        self-efficacy
#> 3 conscientiousness achievement-striving
#> 4 conscientiousness achievement-striving
#> 5 conscientiousness         perseverance
#> 6 conscientiousness         perseverance
#>                                                        statement ID
#> 1       I am someone who can handle demanding tasks effectively.  1
#> 2     I am someone who feels capable of meeting difficult goals.  2
#> 3   I am someone who sets high standards for my accomplishments.  3
#> 4 I am someone who pushes myself to achieve outstanding results.  4
#> 5        I am someone who keep working until a task is finished.  5
#> 6      I am someone who stay focused even when progress is slow.  6
table(items.gpt5.4.example$type)
#> 
#> conscientiousness          openness 
#>                90                90 
table(items.gpt5.4.example$attribute)
#> 
#>    abstract-thinking achievement-striving           aesthetics 
#>                   30                   30                   30 
#>        introspection         perseverance        self-efficacy 
#>                   30                   30                   30 
```
