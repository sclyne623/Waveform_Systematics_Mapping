"""Measure injection bias and calibrate it with same-model controls."""

import argparse
import json

import numpy as np
from geom_median.numpy import compute_geometric_median


ALIGNED = ("ln_chirp_mass", "ln_mass_ratio", "chi_eff", "theta_jn")
PRECESSING = (
    "ln_chirp_mass", "ln_mass_ratio", "spin_1x", "spin_1y", "spin_1z",
    "spin_2x", "spin_2y", "spin_2z", "theta_jn",
)


def posterior_geometry(samples, weights, truth=None, probability=0.9):
    """Return the weighted center, covariance, and squared-radius statistics.

    Supply samples in the analysis coordinates, with one sample per row and
    one importance weight per sample. The optional truth uses the same basis.
    """
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if samples.ndim != 2 or weights.shape != (len(samples),):
        raise ValueError("Expected a sample matrix and one weight per row.")
    if not 0 < probability < 1:
        raise ValueError("The probability must be strictly between zero and one.")
    valid = np.isfinite(samples).all(axis=1) & np.isfinite(weights) & (weights > 0)
    samples, weights = samples[valid], weights[valid]
    if not samples.shape[1] or len(samples) <= samples.shape[1]:
        raise ValueError("Need more valid samples than coordinates.")
    weights /= weights.max()
    weights /= weights.sum()

    center = compute_geometric_median(samples, weights=weights).median
    covariance = np.atleast_2d(np.cov(samples, rowvar=False, aweights=weights))
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("Posterior covariance must be positive definite.") from error
    delta = samples - center
    scaled = np.linalg.solve(covariance, delta.T).T
    radii = np.maximum(np.einsum("ij,ij->i", delta, scaled), 0.0)
    order = np.argsort(radii, kind="stable")
    sorted_radii = radii[order]
    cumulative = np.cumsum(weights[order])
    quantile_index = min(np.searchsorted(cumulative, probability), len(radii) - 1)
    quantile = sorted_radii[quantile_index]
    result = {
        "center": center.tolist(),
        "covariance": covariance.tolist(),
        "condition_number": float(np.linalg.cond(covariance)),
        "posterior_radius_squared_quantile": float(quantile),
        "valid_samples": len(samples),
        "effective_sample_size": float(1 / (weights @ weights)),
    }
    if truth is not None:
        truth = np.asarray(truth, dtype=float)
        if truth.shape != center.shape or not np.isfinite(truth).all():
            raise ValueError("Expected a finite truth in the analysis coordinates.")
        offset = truth - center
        result["d_m_squared"] = float(offset @ np.linalg.solve(covariance, offset))
    return result


def calibrate(control_quantiles):
    """Take the median of the per-control squared-radius quantiles."""
    quantiles = np.asarray(control_quantiles, dtype=float)
    if quantiles.ndim != 1 or not quantiles.size:
        raise ValueError("Supply one quantile per retained same-model control.")
    if not np.isfinite(quantiles).all() or (quantiles < 0).any():
        raise ValueError("Control quantiles must be finite and nonnegative.")
    return float(np.median(quantiles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("posterior", help="NPZ with samples, weights, and optional truth.")
    parser.add_argument("--probability", type=float, default=0.9)
    args = parser.parse_args()
    with np.load(args.posterior, allow_pickle=False) as posterior:
        result = posterior_geometry(
            posterior["samples"], posterior["weights"], posterior.get("truth"),
            args.probability,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
