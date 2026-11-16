"""Independent constrained quantile regression oracle.

At fixed beta1, unroll the recursion into a linear model and solve exact pinball
regression by linear programming (HiGHS), independently of C++ Nelder-Mead.
Checks deterministic conditional-quantile samples at several seeds and compares
against a persistence grid and the C++ fitted persistence. SciPy is test-only.
"""
import csv
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack


def oracle(observations, initial, persistence, tau):
    n = len(observations)
    # Scale returns so the LP feasibility tolerance is meaningful.
    scale = 1000
    design = np.zeros((n, 3))
    offsets = np.zeros(n)
    state = np.zeros(3)
    q = initial * scale
    for i, (aggregate, rv, _) in enumerate(observations):
        state = persistence * state + [1, abs(aggregate)*scale, math.sqrt(rv)*scale]
        q *= persistence
        design[i], offsets[i] = state, q
    target = np.array([r[2]*scale for r in observations])
    matrix = hstack([csr_matrix(design), eye(n), -eye(n)], format='csr')
    objective = np.r_[np.zeros(3), np.full(n, tau), np.full(n, 1-tau)]
    result = linprog(objective, A_eq=matrix, b_eq=target-offsets,
                     bounds=[(None, 0)]*3+[(0, None)]*(2*n), method='highs')
    assert result.success, result.message
    return result.fun/n/scale


def main():
    executable = Path(sys.argv[1]).resolve()
    for seed in (7, 42, 123):
        rng = random.Random(seed)
        tau, q = .05, -.004
        observations = []
        for i in range(900):
            aggregate = rng.gauss(0, .0015)
            rv = (.0003 + rng.random()*.002)**2
            q = -.0003 + .75*q - .2*abs(aggregate) - .1*math.sqrt(rv)
            # Shift exponential innovations so P(target < q | features) = tau.
            target = q + .003*(rng.expovariate(1)+math.log1p(-tau))
            observations.append((aggregate, rv, target))
        with tempfile.TemporaryDirectory(prefix='rcaviar-oracle-') as directory:
            path = Path(directory)/'observations.csv'
            with open(path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['aggregate_return', 'realized_variance', 'target_return'])
                writer.writerows(observations)
            fitted = json.loads(subprocess.check_output([str(executable), str(path), str(tau)], text=True))
        loss = fitted['mean_pinball']
        reference = min(oracle(observations, fitted['initial_quantile'], p, tau)
                        for p in (0, .25, .5, .75, .9, .995, fitted['beta'][1]))
        # Numerical optimization is not a proof of global optimality. This fixture
        # permits a 2% objective gap, while independently verifying the objective.
        assert loss >= reference - 1e-10
        assert loss <= reference*1.02 + 1e-10, (seed, loss, reference, fitted)
        print(f'seed={seed}: C++ pinball={loss:.10g}; LP-grid={reference:.10g}; ratio={loss/reference:.6f}')


if __name__ == '__main__':
    main()
