"""Superpositions of plane-wave simulations, e.g. for finite beams."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from tqdm import tqdm

__all__ = ["SummedRCWA", "pk_to_pte_ptm", "get_real_space_bases", "field_fourier_to_real"]


def pk_to_pte_ptm(px, py, k_inc):
    """Convert an in-plane polarization ``(px, py)`` to TE/TM amplitudes for incidence ``k_inc``."""
    kx, ky = k_inc
    kz = np.sqrt(1 - kx**2 - ky**2)
    k_hat = np.array([kx, ky, kz]) / np.linalg.norm([kx, ky, kz])
    te_vector = np.cross(k_hat, [0, 0, 1])
    tm_vector = np.cross(te_vector, k_hat)
    p_vector = np.array([px, py, 0])
    return np.dot(p_vector, te_vector), np.dot(p_vector, tm_vector)


def get_real_space_bases(k0, gxs, gys, real_space_x_grid, real_space_y_grid):
    """Plane-wave basis ``exp(-i k0 (gx x + gy y))`` sampled on a grid; shape ``[nG, nX, nY]``."""
    phase = (
        -1j * k0 * (gxs.reshape(-1, 1, 1) * real_space_x_grid[None] + gys.reshape(-1, 1, 1) * real_space_y_grid[None])
    )
    return np.exp(phase)


def field_fourier_to_real(coefs, real_space_bases):
    """Synthesize real-space fields from Fourier coefficients ``[..., nG]``."""
    return np.tensordot(coefs, real_space_bases, axes=([-1], [0]))


class SummedRCWA:
    """Coherent sum of RCWA simulations over several incident wavevectors.

    Parameters
    ----------
    obj_ref : RCWA
        Template simulation; it is copied for each incident wavevector.
    freq : float
        Frequency in units of ``c / a``.
    k_incs : array_like, shape (nk, 2)
        In-plane incident wavevectors, normalized by ``k0``.
    amps : array_like, shape (nk,)
        Complex amplitude of each plane-wave component.
    px, py : float
        In-plane polarization of the incident field.
    x_min, x_max, y_min, y_max, num_pts : float, float, float, float, int
        Real-space window and sampling used by :meth:`get_field`.
    """

    def __init__(self, obj_ref, freq, k_incs, amps, px=1, py=0, x_min=-10, x_max=10, y_min=-10, y_max=10, num_pts=100):
        xs = np.linspace(x_min, x_max, num_pts)
        ys = np.linspace(y_min, y_max, num_pts)
        self.real_space_x_grid, self.real_space_y_grid = np.meshgrid(xs, ys)
        self.freq = freq
        self.k_incs = np.array(k_incs)
        self.amps = np.array(amps).ravel()
        if self.k_incs.shape[0] != self.amps.shape[0]:
            raise ValueError("k_incs and amps should have the same length")
        self.px, self.py = px, py
        self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max

        self.obj_ref = obj_ref
        obj_ref.set_freq_k(freq, (0, 0))
        self.gxs = np.diag(obj_ref.Kx)
        self.gys = np.diag(obj_ref.Ky)

        self.objs = []
        self.kzs = []
        for k_inc in self.k_incs:
            obj = deepcopy(obj_ref)
            obj.set_freq_k(freq, kxy_inc=k_inc)
            self.objs.append(obj)
            self.kzs.append(np.sqrt(1 - np.diag(obj.Kx) ** 2 - np.diag(obj.Ky) ** 2 - 1e-9j))

    def total_RT(self):
        """Solve every component; returns ``(R, T)`` of the last component."""
        for k_inc, obj in zip(self.k_incs, self.objs):
            pte, ptm = pk_to_pte_ptm(self.px, self.py, k_inc)
            R, T = obj.get_RT(pte, ptm, storing_intermediate_Smats=True)
        return R, T

    def get_field(self, which_layer=0, z_offset=0, real_space=True):
        """Summed field ``[6, ...]`` (Ex, Ey, Ez, Hx, Hy, Hz). Call :meth:`total_RT` first.

        ``which_layer=0`` gives the reflected field, ``-1`` (or the number of layers) the
        transmitted field, and any other value the field inside that layer.
        """
        num_layers = len(self.objs[0].layer_thicknesses)
        fields = []
        for obj, kz in tqdm(zip(self.objs, self.kzs), total=len(self.objs)):
            if which_layer == 0:
                field = obj.get_RT_field()[0].reshape(6, -1)
            elif which_layer in (-1, num_layers):
                field = obj.get_RT_field()[1].reshape(6, -1)
            else:
                field = np.array(obj.get_internal_field([which_layer], [z_offset])).reshape(6, -1)

            if real_space:
                bases = get_real_space_bases(
                    obj.k0, np.diag(obj.Kx), np.diag(obj.Ky), self.real_space_x_grid, self.real_space_y_grid
                )
                field = field * np.exp(-1j * obj.k0 * kz * z_offset)
                field = field_fourier_to_real(field, bases)
            fields.append(field)
        return np.tensordot(self.amps, np.array(fields), axes=([-1], [0]))
