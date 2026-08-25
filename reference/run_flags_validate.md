# Check that the `run.overall` and `all.together` flags are logically consistent with the number of item types.

Check that the `run.overall` and `all.together` flags are logically
consistent with the number of item types.

## Usage

``` r
run_flags_validate(run.overall, all.together, item.attributes, silently)
```

## Arguments

- run.overall:

  If a final quality analysis should be run on the overall sample

- all.together:

  If the reduction analysis should be run on all of the items agnostic
  of item type

- item.attributes:

  A named list of attributes and item types.

- silently:

  whether the print statements should appear

## Value

a named list with the updated `all.together` and `run.overall` flags
