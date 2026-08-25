# Build GENIE item-filtering audit table

Combines the stage-specific evidence that caused item removal with
pre-reduction network-loading diagnostics. UVA rows report wTO
redundancy evidence and the retained redundant partner. bootEGA rows
report empirical item stability. Network loadings are descriptive
context and are not used as filtering thresholds.

## Usage

``` r
build_filtering_audit(
  items,
  type_name,
  uva_log,
  boot_removed,
  initial_ega,
  uva.cut.off,
  stability.cut.off = 0.75,
  selection_dropped = character(0),
  final_dropped = character(0)
)
```

## Arguments

- items:

  Data frame for one item type with `ID`, `statement`, and `attribute`.

- type_name:

  Character label for the item type.

- uva_log:

  Item-level removal log returned by
  [`reduce_redundancy_uva()`](https://laralee.github.io/AIGENIE/reference/reduce_redundancy_uva.md).

- boot_removed:

  Data frame of removals returned by
  [`iterative_stability_check()`](https://laralee.github.io/AIGENIE/reference/iterative_stability_check.md).

- initial_ega:

  Pre-reduction EGA object computed on the full dense embeddings.

- uva.cut.off:

  Numeric wTO cutoff used by UVA.

- stability.cut.off:

  Numeric item-stability cutoff used by bootEGA.

- selection_dropped:

  Character vector of items left unassigned during embedding/model
  selection.

- final_dropped:

  Character vector of items left unassigned by the final EGA.

## Value

A tidy data frame with one row per filtered item and the evidence for
its removal.
