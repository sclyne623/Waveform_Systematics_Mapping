# Mapping Systematic Biases in Gravitational-Wave Parameter Estimation

Samuel Clyne, Michael Pürrer, Nihar Gupte, Stephen R. Green, Lorenzo Pompili, and Annalena Kofler

This repository contains the derived results and two methods scripts for the
study of aligned-spin and precessing binary black hole waveform systematics.
The inference workflow uses [Dingo](https://github.com/dingo-gw/dingo).

## Check the paper's results

Run the audit with Python 3.10 or later. It verifies the CSV checksums,
compares the precessing clipping decisions with the released pass list, and
reconstructs the common grids, empirical thresholds, mismatch comparison,
network-SNR sensitivity check, and marginal-pass/joint-fail counts.

```sh
python audit.py
```

| Analysis | Common configurations | Calibrated threshold | Mismatch flags below the joint threshold |
| --- | ---: | ---: | ---: |
| Aligned spin | 79 | 7.61 | 67 of 123 |
| Precessing | 87 | 17.48 | 42 of 92 |

See the [data guide](data/README.md) for the table inventory and join keys.
The tables support numerical checks from derived results. Recomputing the
diagnostics from the original samples requires the posterior files. Repeating
the inference requires the original Dingo checkpoints and run settings.

## Apply the methods to your data

Use Python 3.11 for the examples. Install the numerical dependencies with
`python -m pip install -r requirements.txt`.

### Squared Mahalanobis distance

`mahalanobis.py` takes a NumPy `.npz` with `samples` of shape
`(n_samples, n_parameters)`, one importance weight per sample in `weights`,
and an optional vector `truth`. It reports the weighted geometric median,
weighted sample covariance, percentile of the squared posterior radii, and,
if supplied with a truth, its squared distance.

```sh
python mahalanobis.py posterior.npz
```

For the aligned basis, use natural-log chirp mass, natural-log mass ratio,
effective aligned spin, and inclination. For the precessing basis, use
natural-log chirp mass, natural-log mass ratio, the three Cartesian components
of each spin at the 20 Hz reference frequency, and inclination. Angles are
in radians and mass ratio is the lighter mass divided by the heavier mass.
Pass the already transformed samples and injection truth in the same order.
For a control set, run the calculation at probability 0.9 for each retained
same-model posterior and pass those per-control quantiles to `calibrate`.

### Waveform mismatch

`mismatch.py` has an orientation-grid calculation and a sensitive-volume
aggregation function. It maximizes overlap over relative time and template
polarization, searches template phase, and can also search the common
in-plane spin rotation. It uses an 8 × 8 grid over signal phase and
polarization by default.

To aggregate precomputed orientations from a CSV with columns `mismatch`
and `snr`, run:

```sh
python mismatch.py aggregate orientations.csv
```

The Dingo example compares NRSur7dq4 and IMRPhenomXPHM for one precessing
configuration. It uses a 15 Hz generation start, 20 Hz reference frequency,
0.125 Hz bin spacing, and the 20–2048 Hz analysis band.

Install `dingo-gw==0.5.4` and LALSuite in the example environment. The example
was checked with Python 3.11 and LALSimulation 5.2.1. Download
`aLIGODesign.txt` from the
[Advanced LIGO design curve release](https://dcc.ligo.org/LIGO-T1800044/public).
Set `LAL_DATA_PATH` to the directory containing the surrogate file for your
LALSuite version. Use
[NRSur7dq4.h5](https://dcc-llo.ligo.org/LIGO-T1900393/public) for LALSuite
versions earlier than 7.25, and
[NRSur7dq4_v1.0.h5](https://dcc.ligo.org/LIGO-T2500012/public) for newer versions.

```sh
LAL_DATA_PATH=/path/to/lal-data python mismatch.py dingo-example aLIGODesign.txt
```

The example gives a mismatch of about 0.01090 and an effective Hanford SNR
of 24.87255 across 64 orientations in the tested environment. The archived
IMRPhenomXPHM row for the same configuration has a mismatch of 0.01087 and
the same SNR to the precision shown. The archived tables give the paper's
values for the full grids, including the SEOBNRv5 comparisons.
