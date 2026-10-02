"""Scattering matrices, the Redheffer star product, and extended-basis bookkeeping.

Scattering matrices are stored as dicts with keys ``"S11"``, ``"S12"``, ``"S21"``, ``"S22"``.
Sign convention follows EMLab (``exp(-i k.r)``).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, solve

__all__ = [
    "ab_matrices",
    "s_layer",
    "s_reflection",
    "s_transmission",
    "redheffer_star",
    "expanded_indices",
    "expand_matrices",
    "expand_vectors",
    "expand_smatrices",
]


def ab_matrices(W_layer, Wg, V_layer, Vg):
    """Interface matrices ``A = W_layer^-1 Wg + V_layer^-1 Vg`` and ``B`` (with a minus sign)."""
    w = solve(W_layer, Wg)
    v = solve(V_layer, Vg)
    return w + v, w - v


def s_layer(A, B, thickness, k0, Lambda):
    """Symmetric scattering matrix of a layer of given thickness, embedded in gap medium."""
    X = np.diag(np.exp(-np.diag(Lambda) * thickness * k0))
    A_inv_X = solve(A, X)
    term = A - X @ B @ A_inv_X @ B
    S11 = solve(term, X @ B @ A_inv_X @ A - B)
    S12 = solve(term, X @ (A - B @ solve(A, B)))
    return {"S11": S11, "S12": S12, "S21": S12, "S22": S11}


def s_reflection(Ar, Br):
    """Scattering matrix connecting the reflection half-space to the gap medium."""
    Ar_inv = inv(Ar)
    Ar_inv_Br = solve(Ar, Br)
    return {"S11": -Ar_inv_Br, "S12": 2 * Ar_inv, "S21": 0.5 * (Ar - Br @ Ar_inv_Br), "S22": Br @ Ar_inv}


def s_transmission(At, Bt):
    """Scattering matrix connecting the gap medium to the transmission half-space."""
    At_inv = inv(At)
    At_inv_Bt = solve(At, Bt)
    return {"S11": Bt @ At_inv, "S12": 0.5 * (At - Bt @ At_inv_Bt), "S21": 2 * At_inv, "S22": -At_inv_Bt}


def redheffer_star(SA, SB):
    """Redheffer star product ``SA * SB`` of two scattering matrices."""
    identity = np.identity(len(SA["S11"]))
    D = identity - SB["S11"] @ SA["S22"]
    F = identity - SA["S22"] @ SB["S11"]
    return {
        "S11": SA["S11"] + SA["S12"] @ solve(D, SB["S11"]) @ SA["S21"],
        "S12": SA["S12"] @ solve(D, SB["S12"]),
        "S21": SB["S21"] @ solve(F, SA["S21"]),
        "S22": SB["S22"] + SB["S21"] @ solve(F, SA["S22"]) @ SB["S12"],
    }


def expanded_indices(NM, orientation):
    """Map per-block indices into the extended (4D) basis.

    The extended basis is the tensor product of the plane waves of both lattices. Each block
    ``b`` (an order of the *other* lattice) contributes ``2 * NM`` unknowns (x and y
    polarizations). Returns ``idx`` of shape ``(NM, 2 * NM)`` such that local index ``o`` of
    block ``b`` sits at ``idx[b, o]`` in the extended basis.

    For ``orientation == 1`` blocks are interleaved; for ``orientation == 2`` they are
    contiguous within each polarization.
    """
    block = np.arange(NM)[:, None]
    local = np.arange(2 * NM)[None, :]
    if orientation == 1:
        return local * NM + block
    if orientation == 2:
        return (local // NM) * NM * NM + block * NM + local % NM
    raise ValueError(f"orientation must be 1 or 2, got {orientation!r}")


def expand_matrices(mats, NM, orientation):
    """Scatter a list of ``NM`` per-block ``(2NM, 2NM)`` matrices into one block-sparse matrix."""
    idx = expanded_indices(NM, orientation)
    out = np.zeros((2 * NM * NM, 2 * NM * NM), dtype=complex)
    for b, mat in enumerate(mats):
        out[np.ix_(idx[b], idx[b])] = mat
    return out


def expand_vectors(vecs, NM, orientation):
    """Scatter a list of ``NM`` per-block length-``2NM`` vectors into one extended vector."""
    idx = expanded_indices(NM, orientation)
    out = np.zeros(2 * NM * NM, dtype=complex)
    for b, vec in enumerate(vecs):
        out[idx[b]] = np.ravel(vec)
    return out


def expand_smatrices(s_list, NM, orientation):
    """Expand a list of per-block scattering matrices into the extended basis."""
    return {key: expand_matrices([s[key] for s in s_list], NM, orientation) for key in ("S11", "S12", "S21", "S22")}
