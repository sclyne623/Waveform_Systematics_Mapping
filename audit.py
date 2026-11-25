"""Verify the released tables and reproduce selected manuscript results."""

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
GRID = ("spin", "theta_JN", "Mass Ratio")
P_GRID = ("spin", "tilt", "theta_JN", "Mass Ratio")
SETTINGS = {
    "aligned": (
        GRID, "maha_fingerprint_zero_noise_HM.csv", "NRHybSur3dq8",
        {"IMRPhenomXHM": "HM_mismatchIMR_H1_snr3_volume.csv",
         "SEOBNRv5HM": "HM_mismatchEOB_H1_snr3_volume.csv"},
        ("spin", "inclination", "mass ratio"), "SNRs_HM.csv",
        "bias_1D_chis_newsky_zero_noise_full.csv", 4,
    ),
    "precessing": (
        P_GRID, "maha_fingerprint_P_zero_noise_all_injections_.csv", "NRSur7dq4",
        {"IMRPhenomXPHM": "XPHM_mismatch_snr3_volume.csv",
         "SEOBNRv5PHM": "EOB_mismatch_snr3_volume.csv"},
        ("spin", "tilt", "theta_JN", "q"), "SNRs_P.csv",
        "bias_1D_chis_newsky_zero_noise_full_P.csv", 9,
    ),
}


def read(path):
    with (ROOT / "data" / path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def grid_key(row, columns):
    return tuple(round(float(row[column]), 3) for column in columns)


def check(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def clipping_rescues():
    candidates = read("precessing/precessing_joint_clipping_test.csv")
    passed = read("precessing/precessing_zero_noise_joint_clipping_passed.csv")
    check(len(candidates), 81, "Precessing clipping candidates")
    check(len(passed), 25, "Precessing clipping rescues")
    check(
        {(r["Model"], r["case"]) for r in candidates if r["joint_pass"] == "True"},
        {(r["model"], r["case"]) for r in passed},
        "Clipping decisions by recovery",
    )
    return {(r["model"], grid_key(r, ("spin", "tilt", "theta_JN", "q"))) for r in passed}


def summarize(regime, controls, rescues):
    grid, recovery_file, control, mismatch_files, mismatch_grid, snr_file, bias_file, dim = SETTINGS[regime]
    retained = defaultdict(dict)
    for row in read(f"{regime}/{recovery_file}"):
        key, model = grid_key(row, grid), row["Model"]
        if float(row["Effective Sample Size"]) >= 5000 or (model, key) in rescues:
            retained[key][model] = float(row["D_sq"])
    models = {control, *mismatch_files}
    common = {k: values for k, values in retained.items() if values.keys() == models}
    q90 = {
        grid_key(row, grid): float(row["D_sq_posterior_quantile"])
        for row in controls
        if row["Analysis"] == regime and row["Model"] == control and float(row["Probability"]) == 0.9
    }
    eta = median(q90[k] for k in common)

    network = {grid_key(r, grid): float(r["SNR"]) for r in read(f"{regime}/{snr_file}")}
    h1_counts, network_counts = Counter(), Counter()
    for model, filename in mismatch_files.items():
        mismatch = {grid_key(r, mismatch_grid): r for r in read(f"{regime}/{filename}")}
        for k, distances in common.items():
            row = mismatch[k]
            for snr, counts in ((float(row["snr_h1_effective"]), h1_counts),
                                (network[k], network_counts)):
                if float(row["mismatch_snr3"]) > dim / (2 * snr**2):
                    counts["flags"] += 1
                    counts["below_eta"] += distances[model] <= eta

    parameters = {"chirp_mass", "mass_ratio", "chi_eff", "theta_jn"}
    if regime == "precessing":
        parameters.add("chi_p")
    marginal = defaultdict(dict)
    for row in read(f"{regime}/{bias_file}"):
        if row["parameter"] in parameters:
            k = grid_key(row, grid)
            marginal[k, row["Model"]][row["parameter"]] = abs(float(row["Norm Bias"]))
    strict = 0
    for k, distances in common.items():
        for model, dm2 in distances.items():
            values = marginal[k, model]
            check(values.keys(), parameters, "Marginal parameters")
            strict += max(values.values()) < 1.644854 and dm2 > eta
    return (len(common), round(eta, 2), h1_counts["below_eta"], h1_counts["flags"],
            round(100 * network_counts["below_eta"] / network_counts["flags"], 1), strict)


def main():
    for line in (ROOT / "data/SHA256SUMS").read_text().splitlines():
        expected, path = line.split("  ", 1)
        check(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)
    controls = read("controls/mahalanobis_posterior_percentiles_controls.csv")
    rescues = clipping_rescues()
    for regime, expected in (
        ("aligned", (79, 7.61, 67, 123, 57.8, 3)),
        ("precessing", (87, 17.48, 42, 92, 56.0, 22)),
    ):
        common, eta, below, flags, network, strict = result = summarize(
            regime, controls, rescues if regime == "precessing" else set(),
        )
        check(result, expected, regime)
        print(f"{regime}: {common} configurations, eta={eta:.2f}, "
              f"{below}/{flags} H1 mismatch flags below eta, "
              f"{network:.1f}% with the network-SNR threshold, "
              f"{strict} marginal-pass/joint-fail recoveries")


if __name__ == "__main__":
    main()
