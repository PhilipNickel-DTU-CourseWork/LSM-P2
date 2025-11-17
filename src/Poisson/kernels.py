"""Jacobi iteration kernels."""

import numpy as np
from numba import njit, prange


def jacobi_step_numpy(uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float, omega: float) -> float:
    c = 1.0 / 6.0
    h2 = h * h
    N = u.shape[0] - 2

    u[1:-1, 1:-1, 1:-1] = (
        omega * c * (
            uold[0:-2, 1:-1, 1:-1]
            + uold[2:, 1:-1, 1:-1]
            + uold[1:-1, 0:-2, 1:-1]
            + uold[1:-1, 2:, 1:-1]
            + uold[1:-1, 1:-1, 0:-2]
            + uold[1:-1, 1:-1, 2:]
            + h2 * f[1:-1, 1:-1, 1:-1]
        )
        + (1.0 - omega) * uold[1:-1, 1:-1, 1:-1]
    )

    return np.sqrt(np.sum((u - uold) ** 2)) / N**3


@njit(parallel=True)
def jacobi_step_numba(uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float, omega: float) -> float:
    c = 1.0 / 6.0
    h2 = h * h
    N = u.shape[0] - 2

    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            for k in range(1, u.shape[2] - 1):
                u[i, j, k] = (
                    omega * c * (
                        uold[i - 1, j, k]
                        + uold[i + 1, j, k]
                        + uold[i, j - 1, k]
                        + uold[i, j + 1, k]
                        + uold[i, j, k - 1]
                        + uold[i, j, k + 1]
                        + h2 * f[i, j, k]
                    )
                    + (1.0 - omega) * uold[i, j, k]
                )

    diff_sum = 0.0
    for i in prange(u.shape[0]):
        for j in range(u.shape[1]):
            for k in range(u.shape[2]):
                diff = u[i, j, k] - uold[i, j, k]
                diff_sum += diff * diff

    return np.sqrt(diff_sum) / N**3
