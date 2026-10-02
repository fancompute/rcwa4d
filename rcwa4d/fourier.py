"""Fourier-space bookkeeping: convolution matrices and in-plane wavevector matrices.

All wavevectors are normalized by the free-space wavenumber ``k0``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["convmat2D", "k_matrix_2d", "k_matrix_given_mn", "k_expanded"]


def convmat2D(A: np.ndarray, Q: int, P: int) -> np.ndarray:
    """Build the 2D Fourier convolution (Toeplitz) matrix of a real-space pattern.

    Parameters
    ----------
    A : ndarray, shape (Ny, Nx)
        Real-space sampling of the material (e.g. permittivity) over one unit cell.
    Q : int
        Maximum diffraction order along x; orders run from ``-Q`` to ``Q``.
    P : int
        Maximum diffraction order along y; orders run from ``-P`` to ``P``.

    Returns
    -------
    ndarray, shape ((2P+1)(2Q+1), (2P+1)(2Q+1))
        Complex convolution matrix, with row/column index ``p * (2Q+1) + q``.
    """
    shape = A.shape
    Af = np.fft.fftshift(np.fft.fft2(A)) / np.prod(shape)
    p0, q0 = shape[0] // 2, shape[1] // 2  # location of the (0, 0) order in Af

    pp, qq = np.meshgrid(np.arange(-P, P + 1), np.arange(-Q, Q + 1), indexing="ij")
    pp, qq = pp.ravel(), qq.ravel()
    dp = pp[:, None] - pp[None, :]
    dq = qq[:, None] - qq[None, :]
    return Af[p0 + dp, q0 + dq].astype(complex)


def k_matrix_2d(beta_x, beta_y, k0, a_x, a_y, N_p, N_q):
    """Diagonal ``Kx``, ``Ky`` matrices for a rectangular lattice.

    ``beta_x``, ``beta_y`` are the incident in-plane wavevector components divided by ``k0``.
    Orders are flattened row-major over a ``(2N_q+1, 2N_p+1)`` grid.
    """
    k_x = beta_x - 2 * np.pi * np.arange(-N_p, N_p + 1) / (k0 * a_x)
    k_y = beta_y - 2 * np.pi * np.arange(-N_q, N_q + 1) / (k0 * a_y)
    kx, ky = np.meshgrid(k_x, k_y)
    return np.diag(kx.ravel()), np.diag(ky.ravel())


def k_matrix_given_mn(beta_x, beta_y, k0, a_x, a_y, N_x, N_y, n, m, angle, layer=1):
    """``Kx``, ``Ky`` for one block of the extended (4D) plane-wave basis.

    The block is labelled by the order ``(n, m)`` of the *other* lattice; ``layer`` selects
    whether the expansion is done in the frame of lattice 1 (unrotated) or lattice 2 (rotated
    by ``angle``).
    """
    if layer == 1:
        dkx = n * np.cos(angle) + m * np.sin(angle)
        dky = -n * np.sin(angle) + m * np.cos(angle)
        k_x = beta_x - 2 * np.pi * (np.arange(-N_x, N_x + 1) + dkx) / (k0 * a_x)
        k_y = beta_y - 2 * np.pi * (np.arange(-N_y, N_y + 1) + dky) / (k0 * a_y)
        kx, ky = np.meshgrid(k_x, k_y)
    elif layer == 2:
        k_x = -2 * np.pi * np.arange(-N_x, N_x + 1) / (k0 * a_x)
        k_y = -2 * np.pi * np.arange(-N_y, N_y + 1) / (k0 * a_y)
        kx, ky = np.meshgrid(k_x, k_y)
        kx, ky = kx * np.cos(angle) + ky * np.sin(angle), ky * np.cos(angle) - kx * np.sin(angle)
        kx += -n * 2 * np.pi / (k0 * a_x) + beta_x
        ky += -m * 2 * np.pi / (k0 * a_y) + beta_y
    else:
        raise ValueError(f"layer must be 1 or 2, got {layer!r}")
    return np.diag(kx.ravel()), np.diag(ky.ravel())


def k_expanded(beta_x, beta_y, k0, a_x, a_y, N, M, angle, e_r, e_t, m_r=1, m_t=1):
    """Full ``Kx``, ``Ky`` and the reflection/transmission ``Kz`` in the extended basis.

    Returns diagonal matrices of size ``NM**2`` with ``NM = (2N+1)(2M+1)``.
    Note: the incident ``beta`` is not rotated for the second lattice, which is only
    approximate away from normal incidence.
    """
    k_x = beta_x - 2 * np.pi * np.arange(-N, N + 1) / (k0 * a_x)
    k_y = beta_y - 2 * np.pi * np.arange(-M, M + 1) / (k0 * a_y)
    dk_x = -2 * np.pi * np.arange(-N, N + 1) / (k0 * a_x)
    dk_y = -2 * np.pi * np.arange(-M, M + 1) / (k0 * a_y)

    kx, ky = (g.ravel() for g in np.meshgrid(k_x, k_y))
    dkx, dky = np.meshgrid(dk_x, dk_y)
    dkx, dky = dkx * np.cos(angle) + dky * np.sin(angle), -dkx * np.sin(angle) + dky * np.cos(angle)
    dkx, dky = dkx.ravel(), dky.ravel()

    NM = (2 * N + 1) * (2 * M + 1)
    kx = np.tile(dkx, NM) + np.repeat(kx, NM)
    ky = np.tile(dky, NM) + np.repeat(ky, NM)

    kz_r = np.conj(np.sqrt((m_r * e_r - kx**2 - ky**2).astype(complex)))
    kz_t = np.conj(np.sqrt((m_t * e_t - kx**2 - ky**2).astype(complex)))
    return np.diag(kx), np.diag(ky), np.diag(kz_r), np.diag(kz_t)
