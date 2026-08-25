# Build the final GENIE/AIGENIE return object

Build the final GENIE/AIGENIE return object

## Usage

``` r
build_return(item_type_level, overall_result, run.overall, keep.org)
```

## Arguments

- item_type_level:

  Named list containing results at the item-type level.

- overall_result:

  Named list containing results at the overall level when
  `run.overall = TRUE`.

- run.overall:

  Logical. Whether an overall post-reduction fit was run.

- keep.org:

  Logical. Retained for compatibility with callers; original items are
  already handled inside each pipeline result.

## Value

A named list containing `item_type_level` and a combined
`filtering_audit`; when `run.overall = TRUE`, also includes `overall`.
