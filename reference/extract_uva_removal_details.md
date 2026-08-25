# Extract item-level UVA removal evidence

Creates one row per item removed by UVA, retaining the strongest
redundant relationship as the primary diagnostic and all redundant
partners for auditability.

## Usage

``` r
extract_uva_removal_details(
  uva_object,
  removed_ids,
  remaining_ids,
  items,
  sweep,
  cut.off
)
```
