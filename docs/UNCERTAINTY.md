# What the error bars mean

Being rewritten, along with the rest of the documentation.

The standing rules, so that nothing in the application is unexplained in the
meantime:

- **Fit uncertainties are corrected for correlated residuals.** The force is
  low-pass filtered before fitting, so consecutive samples are not independent
  measurements. The reported error is `σ·√τ/√N`, where τ is the integrated
  autocorrelation time of the fit residual.
- **τ is measured per fit, not assumed**, and travels with the results as its
  own column, so the size of an error bar can be explained rather than
  asserted. It differs between cohorts; do not carry a value from one to
  another.
- **Treat the corrected uncertainty as a lower bound.** Part of τ comes from
  filtering and part from the model not perfectly describing the data, and the
  correction absorbs both without distinguishing them.
- **`z_max` decides whether a persistence length means anything.** Below full
  extension the worm-like-chain curve is nearly straight, and a straight line
  constrains only the product of the two fitted lengths. If two populations
  disagree on persistence length, compare their `z_max` first.
- **Nothing is rejected automatically.** An unphysical-looking fit is a real
  answer to what best explains the data under the model; judging it is the
  analyst's job, and the criteria gate makes any exclusion explicit and
  recorded in the export manifest.
