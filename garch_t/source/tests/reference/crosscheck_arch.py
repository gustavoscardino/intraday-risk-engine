"""Cross-check of the C++ GARCH(1,1) estimator against the Python `arch` package.

Usage:
    python crosscheck_arch.py <path/to/fit_series(.exe)>
    e.g. from garch_t:  python source/tests/reference/crosscheck_arch.py bin/fit_series.exe

Requires: numpy, arch (pip install arch). For each configuration a GARCH series is
simulated, fitted by both implementations, and compared:
  * log-likelihood: C++ must reach at least arch's maximum (minus 1e-6);
  * parameters: must agree within tolerances proportional to their standard errors;
  * robust standard errors: must agree within 15%.
Exit code 0 if every case passes.
"""
import json
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from arch import arch_model

warnings.filterwarnings("ignore")


def simulate(n, mu, omega, alpha, beta, nu, seed, burn=500):
    rng = np.random.default_rng(seed)
    if np.isfinite(nu):
        z = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2) / nu)
    else:
        z = rng.standard_normal(n + burn)
    h = omega / (1 - alpha - beta)
    y = np.empty(n + burn)
    for t in range(n + burn):
        e = np.sqrt(h) * z[t]
        y[t] = mu + e
        h = omega + alpha * e * e + beta * h
    return y[burn:]


CASES = [
    # name,            n,    mu,   omega, alpha, beta, nu,     dist,     mean
    ("t/zero/2000",   2000, 0.0,  0.05,  0.08,  0.90, 6.0,    "t",      "Zero"),
    ("t/zero/5000",   5000, 0.0,  0.02,  0.05,  0.93, 4.5,    "t",      "Zero"),
    ("t/const/5000",  5000, 0.05, 0.05,  0.10,  0.85, 8.0,    "t",      "Constant"),
    ("n/zero/3000",   3000, 0.0,  0.03,  0.07,  0.90, np.inf, "normal", "Zero"),
    ("n/const/3000",  3000, 0.02, 0.03,  0.07,  0.90, np.inf, "normal", "Constant"),
    ("t/zero/hi-pers", 8000, 0.0, 0.005, 0.04,  0.955, 7.0,   "t",      "Zero"),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    exe = sys.argv[1]
    failures = 0
    header = f"{'case':16s} {'param':6s} {'C++':>12s} {'arch':>12s} {'diff/SE':>8s}"
    print(header)
    print("-" * len(header))
    with tempfile.TemporaryDirectory() as tmp:
        for seed, (name, n, mu, om, a, b, nu, dist, mean) in enumerate(CASES, start=1):
            y = simulate(n, mu, om, a, b, nu, seed)  # returns in percent units
            path = os.path.join(tmp, f"case{seed}.csv")
            np.savetxt(path, y, header="r", comments="", fmt="%.17g")
            args = [exe, path, "--dist", dist, "--mean", mean.lower()]
            out = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)

            res = arch_model(y, mean=mean, vol="GARCH", p=1, q=1, dist=dist, rescale=False).fit(
                disp="off", options={"maxiter": 2000, "ftol": 1e-12})
            names = (["mu"] if mean == "Constant" else []) + ["omega", "alpha", "beta"] + \
                    (["nu"] if dist == "t" else [])
            ap = dict(zip(names, res.params.values))
            ase = dict(zip(names, res.std_err.values))

            ok = out["loglik"] >= res.loglikelihood - 1e-6
            print(f"{name:16s} {'loglik':6s} {out['loglik']:12.4f} {res.loglikelihood:12.4f} "
                  f"{out['loglik'] - res.loglikelihood:+8.1e}  {'ok' if ok else 'FAIL'}")
            for k in names:
                c, r, se = out["params"][k], ap[k], ase[k]
                z = abs(c - r) / se if se > 0 else 0.0
                rse_c = out["robust_std_err"][k]
                se_ok = rse_c is not None and abs(rse_c / se - 1) < 0.15
                good = z < 0.05 and se_ok
                ok = ok and good
                print(f"{'':16s} {k:6s} {c:12.6f} {r:12.6f} {z:8.4f}  "
                      f"se {rse_c if rse_c is not None else float('nan'):.5f} vs {se:.5f} "
                      f"{'' if good else 'FAIL'}")
            failures += 0 if ok else 1
    print("\nall cases agree with arch" if failures == 0 else f"\n{failures} case(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
