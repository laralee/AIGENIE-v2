# Plot Stability Comparison (network + item stability dotplot, side by side)

Builds a 4-panel comparison: pre-reduction network + pre-reduction item
stability, next to post-reduction network + post-reduction item
stability. Mirrors the layout of the AIGENIE simulation/reference
figure.

## Usage

``` r
plot_stability_comparison(boot1, boot2, caption1, caption2, nmi1, nmi2, title)
```

## Arguments

- boot1, boot2:

  bootEGA objects (pre and post reduction).

- caption1, caption2:

  Captions under each network panel.

- nmi1, nmi2:

  NMI values pre/post.

- title:

  Overall title.

## Value

A patchwork object combining the four panels.
