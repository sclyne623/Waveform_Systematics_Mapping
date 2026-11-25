"""Compute orientation-averaged, single-detector waveform mismatch."""

import argparse
import json
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.optimize import minimize


def summarize_orientations(mismatches, snrs, silence_fraction=1e-5):
    """Return the sensitive-volume mismatch, effective SNR, and usable count."""
    mismatches, snrs = np.asarray(mismatches, float), np.asarray(snrs, float)
    if snrs.ndim != 1 or mismatches.shape != snrs.shape or not snrs.size:
        raise ValueError("Expected one mismatch and SNR per orientation.")
    if not np.isfinite(snrs).all() or (snrs < 0).any() or snrs.max() == 0:
        raise ValueError("SNRs must be finite and nonnegative, with a positive maximum.")
    if not 0 <= silence_fraction < 1:
        raise ValueError("Silence fraction must lie in [0, 1).")
    valid = (snrs > 0) & (snrs >= silence_fraction * snrs.max())
    match = 1 - mismatches[valid]
    if not np.isfinite(match).all() or (match < 0).any() or (match > 1).any():
        raise ValueError("Active orientation mismatches must lie in [0, 1].")
    scale = snrs[valid].max()
    rho = snrs[valid] / scale
    volume = np.sum(rho**3)
    return {
        "mismatch_snr3": float(1 - np.cbrt(np.sum(rho**3 * match**3) / volume)),
        "snr_h1_effective": float(scale * np.sqrt(volume / rho.sum())),
        "orientations_used": len(rho),
    }


def _maximum_overlap(signal, template_polarizations, sqrt_weight, n_fft, snr):
    """Maximize the real overlap over time shifts and template polarization."""
    template = np.asarray(template_polarizations) * sqrt_weight
    gram = np.real(template.conj() @ template.T)
    values, vectors = np.linalg.eigh(gram)
    active = values > values[-1] * 1e-15
    if not active.any():
        raise ValueError("The template has zero power in the analysis band.")
    basis = (vectors[:, active].T @ template) / np.sqrt(values[active, None])
    correlations = n_fft * np.fft.ifft(basis.conj() * signal, n=n_fft, axis=1).real
    return float(np.clip(np.linalg.norm(correlations, axis=0).max() / snr, 0, 1))


def waveform_mismatch(signal_at, template_at, psd, delta_f, *, precessing=False,
                      f_low=20.0, f_high=2048.0, orientations=8):
    """Compare waveform callbacks on the one-sided grid starting at zero.

    signal_at(phi) and template_at(phi, alpha, signal_phi) return complex FD plus
    and cross arrays at fixed inclination. For precessing templates, alpha
    rotates the in-plane spins defined at the signal phase signal_phi.
    The aligned search always passes alpha=0.
    """
    psd = np.asarray(psd, dtype=float)
    if psd.ndim != 1 or len(psd) < 2 or not 0 < delta_f or orientations < 1:
        raise ValueError("Expected a one-sided PSD grid and positive grid spacing.")
    freq = np.arange(len(psd)) * delta_f
    in_band = (freq >= f_low) & (freq <= f_high) & (freq > 0)
    if precessing:
        # Match the half-open frequency interval used by the PyCBC calculation.
        in_band &= freq < f_high
    if not in_band.any() or not np.isfinite(psd[in_band]).all() or (psd[in_band] <= 0).any():
        raise ValueError("The analysis band needs a finite, strictly positive PSD.")
    sqrt_weight = np.zeros_like(psd)
    sqrt_weight[in_band] = np.sqrt(4 * delta_f / psd[in_band])
    n_fft = 2 * (len(psd) - 1) if precessing else 8 * len(psd)

    angles = np.linspace(0, 2 * np.pi, orientations, endpoint=False)
    signals = [
        (np.cos(angles)[:, None] * hp + np.sin(angles)[:, None] * hc) * sqrt_weight
        for hp, hc in (signal_at(phi) for phi in angles)
    ]
    snrs = np.linalg.norm(signals, axis=2).ravel()
    if snrs.max() == 0:
        raise ValueError("Every signal orientation has zero power in the analysis band.")
    phases = np.linspace(0, 2 * np.pi, 10 if precessing else 40)
    coarse = list(product(phases, phases)) if precessing else [(p,) for p in phases]
    mismatches = np.full(len(snrs), np.nan)
    for i, strain in enumerate(np.concatenate(signals)):
        if snrs[i] < 1e-5 * snrs.max():
            continue

        def objective(point):
            phase, *rotation = point
            template = template_at(phase, rotation[0] if rotation else 0.0,
                                   angles[i // orientations])
            return 1 - _maximum_overlap(strain, template, sqrt_weight, n_fft, snrs[i])

        scores = [objective(point) for point in coarse]
        best = int(np.argmin(scores))
        refined = minimize(objective, coarse[best], method="Nelder-Mead")
        mismatches[i] = min(scores[best], refined.fun) if np.isfinite(refined.fun) else scores[best]
    return summarize_orientations(mismatches, snrs)


def dingo_example(asd_file, signal_model, orientations):
    """Compare one precessing injection against IMRPhenomXPHM with Dingo."""
    from dingo.gw.domains import build_domain
    from dingo.gw.waveform_generator import WaveformGenerator

    delta_f, f_max = 0.125, 2048.0
    domain = build_domain(
        dict(type="FrequencyDomain", f_min=20.0, f_max=f_max, delta_f=delta_f)
    )
    settings = dict(domain=domain, f_ref=20.0, f_start=15.0)
    signal = WaveformGenerator(approximant=signal_model, **settings)
    template = WaveformGenerator(approximant="IMRPhenomXPHM", **settings)
    params = dict(
        chirp_mass=35.0, mass_ratio=0.5, luminosity_distance=900.0,
        theta_jn=np.pi / 3, a_1=0.5, a_2=0.0, tilt_1=np.pi / 3, tilt_2=0.0,
        phi_jl=0.0, phi_12=0.0, phase=0.0, geocent_time=0.0,
    )
    frequency, asd = np.loadtxt(asd_file, unpack=True)
    psd = np.interp(domain.sample_frequencies, frequency, asd**2, left=np.nan, right=np.nan)

    def polarizations(waveform):
        return tuple(domain.update_data(waveform[key]) for key in ("h_plus", "h_cross"))

    def signal_at(phi):
        return polarizations(signal.generate_hplus_hcross(dict(params, phase=phi)))

    @lru_cache
    def base_arguments(signal_phi):
        return template._convert_parameters(
            dict(params, phase=signal_phi, f_ref=settings["f_ref"])
        )

    def template_at(phi, alpha, signal_phi):
        # SimInspiralFD uses Cartesian spin slots 2–7 and orbital phase slot 10.
        base = base_arguments(signal_phi)
        arguments = list(base)
        c, s = np.cos(alpha), np.sin(alpha)
        for x, y in ((2, 3), (5, 6)):
            arguments[x] = c * base[x] - s * base[y]
            arguments[y] = s * base[x] + c * base[y]
        arguments[10] = phi
        return polarizations(template.generate_FD_waveform(tuple(arguments)))

    return waveform_mismatch(
        signal_at, template_at, psd, delta_f, precessing=True, orientations=orientations,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate", help="Combine precomputed orientations.")
    aggregate.add_argument("csv", help="CSV with mismatch and snr columns.")
    example = subparsers.add_parser("dingo-example", help="Generate the example with Dingo.")
    example.add_argument("asd", help="Two-column frequency and ASD text file.")
    example.add_argument("--signal-model", choices=["NRSur7dq4", "IMRPhenomXPHM"], default="NRSur7dq4")
    example.add_argument("--orientations", type=int, default=8, help="Grid size per angle.")
    args = parser.parse_args()
    if args.command == "aggregate":
        data = np.genfromtxt(args.csv, names=True, delimiter=",", ndmin=1)
        result = summarize_orientations(data["mismatch"], data["snr"])
    else:
        result = dingo_example(args.asd, args.signal_model, args.orientations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
