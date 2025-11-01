import numpy as np

def gauss_seidel(Ybus, P, V_init=None, tol=1e-4, max_iter=100):
    """
    Gauss-Seidel power flow for PQ buses (no PV).
    - Ybus: (n, n) complex numpy array for bus admittance matrix.
    - P: (n,) complex numpy array for net complex bus power injections (negative for loads, positive for generations).
    - V_init: (n,) complex numpy array, initial bus voltages.
    - tol: convergence (default 1e-4).
    - max_iter: max iterations.

    Assumes bus 0 is the slack/reference bus: its voltage is fixed!
    """

    n = len(P)
    # Initial bus voltages: default 1.0+0j (flat start)
    if V_init is None:
        V = np.ones(n, dtype=complex)
    else:
        V = V_init.copy()

    for _ in range(max_iter):
        V_prev = V.copy()
        for i in range(n):
            if i == 0:
                # Slack bus stays fixed
                continue
            sigma = sum(Ybus[i, j] * V[j] for j in range(n) if j != i)
            # Gauss-Seidel update for bus i (PQ type)
            V[i] = (1 / Ybus[i, i]) * ((P[i] / np.conj(V[i])) - sigma)
        # Convergence check
        if np.max(np.abs(V - V_prev)) < tol:
            break
    return V
