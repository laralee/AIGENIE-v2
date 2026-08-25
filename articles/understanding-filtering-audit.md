# Understanding the AIGENIE / GENIE Filtering Audit

## Overview

The `filtering_audit` output provides an **item-level provenance
record** for the AIGENIE / GENIE reduction process.

Each row corresponds to an item removed during filtering and documents:

1.  **what item was removed**,
2.  **when it was removed**,
3.  **which formal criterion caused removal**,
4.  **which statistic triggered the decision**, and
5.  **what the item’s structural profile looked like before removal**.

Conceptually:

``` text
What item was removed?
        |
        v
At what stage was it removed?
        |
        v
What formal criterion caused removal?
        |
        v
What statistic triggered that criterion?
        |
        v
What structural evidence characterizes the item?
```

The combined audit is available from:

``` r

results$filtering_audit
```

Type-specific audits are available from:

``` r

results$item_type_level$openness$filtering_audit
results$item_type_level$conscientiousness$filtering_audit
```

## Loading the bundled worked example

AIGENIE includes a frozen GPT-5.4 item pool and matching OpenAI
`text-embedding-3-small` embeddings so that users can inspect a
realistic GENIE example without regenerating items or embeddings.

``` r

library(AIGENIE)
#> Loading required package: EGAnet
#> 
#> EGAnet (version 2.4.1) 
#> 
#> For help getting started, see <https://r-ega.net> 
#> 
#> For bugs and errors, submit an issue to <https://github.com/hfgolino/EGAnet/issues>
#> AI-GENIE loaded. Python dependencies will be configured on first use.
#> For GPU support, run: AIGENIE::install_gpu_support()
#> For local LLM support, run: AIGENIE::install_local_llm_support()

data("items.gpt5.4.example")
data("embeddings.gpt5.4.example")

dim(items.gpt5.4.example)
#> [1] 180   4
dim(embeddings.gpt5.4.example)
#> [1] 1536  180

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
```

The full GENIE analysis is intentionally not evaluated while building
the vignette because bootstrapped EGA is computationally expensive.

Users can reproduce the analysis interactively with:

``` r

gpt54 <- GENIE(
  items = items.gpt5.4.example,
  embedding.matrix = embeddings.gpt5.4.example,
  EGA.model = "glasso",
  EGA.algorithm = "walktrap",
  EGA.uni.method = "louvain",
  uva.cut.off = 0.20,
  run.overall = TRUE,
  all.together = FALSE,
  plot = FALSE,
  silently = FALSE
)

gpt54$filtering_audit
```

## Complete definition of every output element

| Output | Definition | Interpretation |
|----|----|----|
| `ID` | Unique identifier of the removed item. | Links the audit row to the original item pool. |
| `type` | Higher-level item group analyzed separately by GENIE. | For example, `conscientiousness` or `openness`. |
| `attribute` | Intended theoretical dimension or facet assigned to the item. | Examples include `achievement-striving` and `abstract-thinking`. |
| `statement` | Full text of the removed item. | Makes the audit directly interpretable without joining back to the original item table. |
| `removal_stage` | Stage of AIGENIE / GENIE that removed the item. | Typically `UVA` or `bootEGA`. |
| `reason` | Human-readable explanation of the formal removal decision. | For example, `Redundancy: wTO = 0.280 >= 0.20`. |
| `diagnostic_name` | Name of the statistic that determined removal. | `wTO` for UVA and `item_stability` for bootEGA. |
| `diagnostic_value` | Numerical value of the statistic used in the decision. | Compared against the relevant cutoff. |
| `cutoff` | Prespecified threshold used at that filtering stage. | For example, `.20` for UVA or `.75` for bootEGA. |
| `uva_sweep` | Iterative UVA sweep in which the item was removed. | `1` means the item was identified in the first redundancy pass. `NA` for non-UVA removals. |
| `redundant_with_ID` | Primary redundancy counterpart associated with a UVA removal. | Identifies the item providing highly overlapping structural information. |
| `redundant_with_statement` | Text of the primary redundancy counterpart. | Allows direct inspection of content overlap. |
| `redundant_wTO` | Weighted topological overlap between the removed item and its primary redundancy counterpart. | Higher values indicate greater local network redundancy. |
| `all_redundant_with_IDs` | All items with which the removed item met the UVA redundancy criterion during that sweep. | Useful when redundancy occurs in a cluster rather than a single pair. |
| `all_redundant_wTO` | Compact representation of all redundancy partners and their corresponding wTO values. | For example, `46=0.280`. |
| `boot_run` | Iterative bootEGA filtering run in which the unstable item was removed. | `1` means the item failed the first stability assessment. |
| `item_stability` | Empirical bootEGA item stability. | Quantifies reproducibility of the item’s community placement across bootstrap samples. |
| `stability_deficit` | Amount by which item stability fell below the required cutoff. | Computed as `cutoff - item_stability` for unstable items. |
| `pre_reduction_EGA_community` | Community assigned by EGA before filtering. | Community numbers are labels, not ordered dimensions. |
| `pre_reduction_primary_network_loading` | Standardized network loading for the item’s EGA-assigned community before reduction. | Larger absolute values indicate stronger association with the assigned dimension. |
| `pre_reduction_primary_network_loading_abs` | Absolute value of the primary network loading. | Useful for comparing loading magnitude independent of sign. |
| `pre_reduction_strongest_cross_community` | Alternative EGA community with the strongest non-primary loading in absolute magnitude. | Identifies the strongest competing dimensional association. |
| `pre_reduction_strongest_cross_loading` | Signed network loading on that strongest alternative community. | Retains direction and magnitude. |
| `pre_reduction_strongest_cross_loading_abs` | Absolute value of the strongest cross-community loading. | Facilitates comparison with the primary loading. |
| `pre_reduction_loading_gap` | Difference between the absolute primary loading and absolute strongest cross-loading. | Large positive values indicate clear dimensional assignment; values near zero indicate ambiguity. |

## Formal filtering criteria versus structural diagnostics

A critical distinction is that `filtering_audit` contains both:

- **formal filtering criteria**, which determine whether an item is
  removed, and
- **descriptive structural diagnostics**, which characterize the item’s
  psychometric profile.

### UVA removal criterion

For UVA:

``` text
diagnostic_name  = wTO
diagnostic_value = weighted topological overlap
cutoff           = UVA cutoff
```

An item is flagged for local redundancy when

``` math
wTO \geq \text{cutoff}.
```

For example, with a cutoff of $`0.20`$,

``` math
0.280 \geq 0.20
```

meets the redundancy criterion.

### bootEGA removal criterion

For bootEGA:

``` text
diagnostic_name  = item_stability
diagnostic_value = empirical item stability
cutoff           = stability cutoff
```

An item is removed when

``` math
\text{item stability} < \text{cutoff}.
```

For example,

``` math
0.414 < 0.750
```

indicates insufficient structural stability.

## Network loadings do not determine removal

The `pre_reduction_*network_loading*` fields are **diagnostic
evidence**, not filtering thresholds.

This distinction is essential.

In the bundled GPT-5.4 example, four UVA-removed items had strong
primary network loadings and very small cross-loadings:

|  ID | Primary loading | Strongest cross-loading | Loading gap |
|----:|----------------:|------------------------:|------------:|
|   4 |            .487 |                    .019 |        .468 |
|  28 |            .447 |                    .044 |        .403 |
|  58 |            .416 |                    .038 |        .378 |
| 131 |            .510 |                   -.006 |        .504 |

These items were not removed because they were weak indicators. They
were removed because they provided **locally redundant information**
relative to other items.

## Example: a UVA removal

Consider Item 4:

``` text
ID                    4
removal_stage         UVA
diagnostic_name       wTO
diagnostic_value      0.2797578
cutoff                0.20
uva_sweep             1
redundant_with_ID     46
redundant_wTO         0.2797578
```

Removed statement:

> I am someone who pushes myself to achieve outstanding results.

Redundancy counterpart:

> I am someone who push myself to deliver results that stand out.

The appropriate interpretation is:

> During the first UVA sweep, Item 4 and Item 46 showed weighted
> topological overlap of approximately .280. Because this exceeded the
> prespecified .20 redundancy threshold, Item 4 was removed.

Both items may be strong indicators of achievement striving. The issue
is that they provide highly overlapping information.

## Why `all_redundant_*` exists

Redundancy does not always occur as a single isolated pair.

An item may exceed the wTO threshold with several other items:

``` text
all_redundant_with_IDs
"15; 22; 41"

all_redundant_wTO
"15=0.231; 22=0.287; 41=0.219"
```

In that situation:

- `redundant_with_ID` identifies the primary redundancy counterpart,
- `all_redundant_with_IDs` preserves the complete redundancy set, and
- `all_redundant_wTO` preserves the corresponding numerical evidence.

## Understanding bootEGA item stability

Conceptually, bootEGA asks:

> Across bootstrap replications, does this item continue to belong to
> the same dimension?

An item with:

``` text
item_stability = 0.95
```

has highly reproducible structural placement.

An item with:

``` text
item_stability = 0.414
```

shows considerably weaker reproducibility.

With a cutoff of `.75`, the latter item fails the stability requirement.

### Stability deficit

The audit defines stability deficit as

``` math
D_s = c_s - s_i,
```

where $`c_s`$ is the stability cutoff and $`s_i`$ is item stability.

For Item 95,

``` math
D_s = 0.750 - 0.414 = 0.336.
```

A larger positive `stability_deficit` indicates that the item fell
farther below the required stability criterion.

## Example: a bootEGA removal

Item 95 in the bundled worked example had:

``` text
removal_stage       bootEGA
item_stability      0.414
cutoff              0.750
stability_deficit   0.336
```

Its pre-reduction network-loading profile was approximately:

``` text
primary loading          = 0.195
strongest cross-loading  = 0.170
loading gap              = 0.025
```

The primary and strongest cross-loading were almost identical:

``` math
0.195 - 0.170 = 0.025.
```

This is consistent with poorly differentiated dimensional placement.

The correct interpretation is:

> Item 95 was removed because its empirical bootEGA item stability was
> below the prespecified criterion. Its weakly differentiated
> network-loading profile provides convergent structural evidence for
> that instability.

The incorrect interpretation is:

> Item 95 was removed because its loading was too low.

Network loadings are not themselves the filtering rule.

## Understanding the network-loading diagnostics

### Primary network loading

`pre_reduction_primary_network_loading` is the standardized network
loading corresponding to the item’s **EGA-assigned community**.

If EGA assigns item $`i`$ to community $`k`$, then

``` math
\lambda_{\text{primary},i} = \lambda_{ik}.
```

It is not simply defined as the largest loading in the row.

### Strongest cross-loading

Among all communities other than the assigned community, AIGENIE
identifies the loading with the greatest absolute magnitude.

The audit reports:

``` text
pre_reduction_strongest_cross_community
pre_reduction_strongest_cross_loading
pre_reduction_strongest_cross_loading_abs
```

### Loading gap

The loading gap is

``` math
\Delta_{\lambda,i}
=
|\lambda_{\text{primary},i}|
-
|\lambda_{\text{cross},i}|.
```

Interpretation:

- **large positive gap**: clear dimensional assignment,
- **gap near zero**: similar association with another dimension,
- **negative gap**: an alternative loading exceeds the loading on the
  EGA-assigned community.

For Item 95:

``` math
|\lambda_p| = 0.195,
```

``` math
|\lambda_c| = 0.170,
```

and

``` math
\Delta_\lambda = 0.195 - 0.170 = 0.025.
```

The small gap is consistent with ambiguous dimensional placement.

## Why the `pre_reduction_` prefix matters

The variables deliberately use names such as:

``` text
pre_reduction_primary_network_loading
pre_reduction_strongest_cross_loading
pre_reduction_loading_gap
```

because they describe the item **before it was removed**.

This prevents a methodological misunderstanding that network loadings
constitute the filtering rule. Instead, they provide a snapshot of the
item’s structural profile before filtering.

## Type-level and pooled audit scopes

For a type-specific audit:

``` r

results$item_type_level$openness$filtering_audit
```

the network diagnostics describe the type-level pre-reduction structure.

The combined audit is available from:

``` r

results$filtering_audit
```

and provides a unified, publication-ready record across item types.

When `run.overall = TRUE`, GENIE also returns a pooled post-reduction
fit:

``` r

results$overall
```

This pooled analysis evaluates the union of the items that survived the
type-level reductions. It does **not** perform another UVA or bootEGA
reduction.

## Four questions answered by the audit

### 1. What was removed?

``` text
ID
type
attribute
statement
```

### 2. Why was it removed?

``` text
removal_stage
reason
diagnostic_name
diagnostic_value
cutoff
```

### 3. What exact filtering evidence was involved?

For redundancy:

``` text
uva_sweep
redundant_with_ID
redundant_with_statement
redundant_wTO
all_redundant_with_IDs
all_redundant_wTO
```

For instability:

``` text
boot_run
item_stability
stability_deficit
```

### 4. What did its dimensional structure look like before removal?

``` text
pre_reduction_EGA_community
pre_reduction_primary_network_loading
pre_reduction_primary_network_loading_abs
pre_reduction_strongest_cross_community
pre_reduction_strongest_cross_loading
pre_reduction_strongest_cross_loading_abs
pre_reduction_loading_gap
```

## Worked GPT-5.4 result

The bundled regression example produces:

``` text
Conscientiousness: 90 -> 87 items
Initial NMI:       1.000
Final NMI:         1.000

Openness:          90 -> 88 items
Initial NMI:       0.9555
Final NMI:         1.000

Overall:           180 -> 175 items
Initial NMI:       0.9864
Final NMI:         1.000
```

The audit identifies:

- four UVA redundancy removals, and
- one bootEGA instability removal.

The two stages therefore serve complementary purposes:

- **UVA** removes local redundancy.
- **bootEGA** removes insufficiently stable dimensional assignments.

## Reduction summaries

In addition to the item-level audit, each type-level result contains a
stage-by-stage `reduction_summary`:

``` r

results$item_type_level$openness$reduction_summary
results$item_type_level$conscientiousness$reduction_summary
```

These summaries document the trajectory from the initial item pool
through UVA and bootEGA to the final retained pool, including changes in
item count and NMI.

## Summary

The `filtering_audit` is more than a deletion log.

It provides an **item-level psychometric provenance record** for the
AIGENIE / GENIE reduction process.

For every removed item, it records:

- the item itself,
- the filtering stage,
- the formal removal criterion,
- the exact diagnostic statistic,
- the decision threshold,
- the relevant redundancy or stability evidence, and
- the item’s pre-reduction network-loading profile.

AIGENIE / GENIE therefore return not only a reduced item pool, but also
a transparent and publication-ready justification for every filtering
decision.
