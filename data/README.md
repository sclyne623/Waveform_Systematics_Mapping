# Derived results

The CSVs preserve the numeric values from the analysis exports. The release
removes filesystem paths, internal run and error metadata, redundant CSV
indices, and exploratory columns from the two Mahalanobis summary tables.
Run `python audit.py` from the repository root to verify their checksums.

| Location | Contents |
| --- | --- |
| `aligned/`, `precessing/` | Per-recovery squared Mahalanobis distances and effective sample sizes, normalized marginal biases, effective Hanford SNR and sensitive-volume mismatch for each test model, and HLV network SNR for each injection. The precessing directory also includes all clipping candidates and the rescued subset. |
| `controls/` | Long-form weighted posterior radius quantiles for same-model controls. The squared-radius column is `D_sq_posterior_quantile` and the 90th-percentile rows have `Probability = 0.90`. |
| `appendix/` | Alternative-center distances, matched-coordinate 5D exports and marginal biases, 6D/9D comparisons, and covariance eigenspectra. |
| `aligned/restricted_mode_reanalysis.csv`, `aligned/mode_content_mismatch.csv` | Paired restricted-mode recoveries and the full-grid decomposition of mismatch into shared- and missing-mode contributions. |

## Joining tables

- The aligned injection key is `spin`, inclination, and mass ratio. The
  precessing key also includes primary spin tilt. Use the model as part of the
  key for recovery-level tables. The four main mismatch files identify their
  waveform model in the filename.
- `Mass Ratio`, `mass ratio`, and `q` denote the lighter-to-heavier mass
  ratio. Inclination is `theta_JN` or `inclination`. The main tables use
  radians for angles. The restricted-mode table labels its degree column
  explicitly. The manuscript audit matches the main grids after rounding the
  injection coordinates to three decimal places.
- Build the common grid by keeping aligned recoveries with effective sample
  size at least 5,000 and precessing recoveries that reach 5,000 or pass
  clipping. Then keep only the configurations present for all three models
  in that analysis. For checks on a single aligned model, the paper also
  reports results on that model's own retained grid.
- `D_sq` is the truth-to-center squared Mahalanobis distance. Use the
  median of the retained same-model controls' within-posterior 90th
  percentiles for calibration. For the main mismatch check,
  `mismatch_snr3` and `snr_h1_effective` define the comparison with
  the threshold `n_p / (2 * snr_h1_effective**2)`.
