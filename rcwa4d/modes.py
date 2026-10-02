"""Eigenmodes of homogeneous and patterned layers."""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve

__all__ = ["homogeneous_modes", "pq_matrices", "eigen_modes"]


def homogeneous_modes(Kx, Ky, e_r=1, m_r=1):
    """Analytic eigenmodes of a homogeneous layer.

    Returns
    -------
    W : ndarray
        Electric-field mode matrix (identity).
    V : ndarray
        Magnetic-field mode matrix.
    Kz : ndarray
        Diagonal matrix of normalized longitudinal wavevectors. The branch is chosen so that
        propagating waves have ``Kz > 0`` and evanescent waves have ``Im(Kz) < 0``.
    """
    n = len(Kx)
    identity = np.identity(n)
    P = np.block([[Kx * Ky, e_r * m_r * identity - Kx**2], [Ky**2 - m_r * e_r * identity, -Ky * Kx]]) / e_r
    Q = (e_r / m_r) * P

    kz2 = (m_r * e_r - np.diag(Kx) ** 2 - np.diag(Ky) ** 2).astype(complex)
    kz = np.where(kz2.real < 0, -1j * np.sqrt(-kz2), np.sqrt(kz2))

    eigenvalues = 1j * np.concatenate([kz, kz])
    V = Q / eigenvalues[None, :]
    return np.identity(2 * n), V, np.diag(kz)


def pq_matrices(Kx, Ky, e_conv, mu_conv):
    """The ``P`` and ``Q`` matrices of the RCWA eigenproblem for a patterned layer."""
    e_kx, e_ky = solve(e_conv, Kx), solve(e_conv, Ky)
    mu_kx, mu_ky = solve(mu_conv, Kx), solve(mu_conv, Ky)
    P = np.block([[Kx @ e_ky, mu_conv - Kx @ e_kx], [Ky @ e_ky - mu_conv, -Ky @ e_kx]])
    Q = np.block([[Kx @ mu_ky, e_conv - Kx @ mu_kx], [Ky @ mu_ky - e_conv, -Ky @ mu_kx]])
    return P, Q


def eigen_modes(P, Q):
    """Solve the layer eigenproblem.

    Returns the electric mode matrix ``W``, the diagonal eigenvalue matrix ``Lambda``
    (``Lambda**2`` are the eigenvalues of ``P @ Q``), and the magnetic mode matrix ``V``.
    """
    lambda_squared, W = np.linalg.eig(P @ Q)
    lam = np.sqrt(lambda_squared.astype(complex))
    V = (Q @ W) / lam[None, :]
    return W, np.diag(lam), V
