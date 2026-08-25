# Compute pre-reduction item-level network-loading diagnostics

Uses
[`EGAnet::net.loads()`](https://rdrr.io/pkg/EGAnet/man/net.loads.html)
on an EGA solution and reports, for each item, the loading on its
assigned EGA community, its strongest loading on another community, and
the absolute primary-to-cross-loading gap. These statistics are
descriptive audit information; they are not item-removal criteria.

## Usage

``` r
network_loading_diagnostics(ega_object, items)
```

## Arguments

- ega_object:

  An [`EGAnet::EGA.fit`](https://rdrr.io/pkg/EGAnet/man/EGA.fit.html)
  object or a standard EGA object.

- items:

  Data frame containing at least `ID`.

## Value

A data frame with one row per item represented in the EGA network.
