# Print Results

Displays a summary of the AI-GENIE analysis results, including the EGA
model used, embedding type, starting and final number of items, and NMI
values before and after reduction. The summary includes the number of
iterations for both UVA (Unique Variable Analysis) and bootstrapped EGA
steps.

## Usage

``` r
print_results(obj, obj2, run.overall)
```

## Arguments

- obj:

  A list object containing the OVERALL analysis results returned by
  `get_results`.

- obj2:

  A list object containing the ITEM-TYPE LEVEL analysis results returned
  by `get_results`.

- run.overall:

  A flag denoting if overall results should be printed

## Value

No return value; the function prints the results to the console.
