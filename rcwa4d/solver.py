"""The RCWA solver for (possibly twisted) stacks of periodic layers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .fourier import convmat2D, k_expanded, k_matrix_2d, k_matrix_given_mn
from .modes import eigen_modes, homogeneous_modes, pq_matrices
from .smatrix import (
    ab_matrices,
    expand_matrices,
    expand_smatrices,
    expand_vectors,
    expanded_indices,
    redheffer_star,
    s_layer,
    s_reflection,
    s_transmission,
)

__all__ = ["RCWA", "rcwa"]


class RCWA:
    """Rigorous coupled-wave analysis of a layered stack, with optional twist between layers.

    Each layer is periodic in one of two lattices: orientation ``1`` (unrotated) or
    orientation ``2`` (rotated by ``twist``). When ``twist == 0`` the standard RCWA is used;
    otherwise the fields are expanded in the product basis of both lattices (the "4D" basis),
    which has ``NM**2`` plane waves with ``NM = (2N+1)(2M+1)``.

    Lengths are in units of ``a`` and frequencies in units of ``c / a``. Light is incident
    from the reflection half-space (``e_r``, ``m_r``) on the first layer and exits into the
    transmission half-space (``e_t``, ``m_t``).

    Parameters
    ----------
    epsr_list : sequence of ndarray or None
        Real-space permittivity of each layer, sampled over one unit cell. ``None`` marks a
        vacuum gap layer.
    thickness_list : sequence of float
        Thickness of each layer.
    orientation_list : sequence of {1, 2}, optional
        Lattice each layer belongs to. Ignored (may be ``None``) when ``twist == 0``.
    mu_list : sequence of ndarray or None, optional
        Real-space permeability of each layer. Defaults to non-magnetic layers.
    twist : float
        Rotation angle of lattice 2 relative to lattice 1, in radians.
    gap_layer_indices : sequence of int, optional
        Indices of vacuum layers, which are computed analytically. Layers with
        ``epsr_list[i] is None`` are always treated as gap layers.
    N, M : int
        Truncation of the Fourier expansion: orders ``-N..N`` along x and ``-M..M`` along y.
    a : float
        Length unit used to normalize frequency.
    ax, ay : float, optional
        Lattice constants along x and y (default to ``a``).
    e_r, e_t, m_r, m_t : float
        Permittivity / permeability of the reflection and transmission half-spaces.
    verbose : bool
        Print progress messages.

    Examples
    --------
    >>> sim = RCWA([eps, eps], [0.2, 0.2], [1, 2], twist=np.deg2rad(5), N=1, M=1)
    >>> sim.set_freq_k(0.75, theta_phi=(0, 0))
    >>> (R, T), (r0, t0) = sim.get_RT(pte=0, ptm=1)
    """

    def __init__(
        self,
        epsr_list: Sequence[np.ndarray | None],
        thickness_list: Sequence[float],
        orientation_list: Sequence[int] | None = None,
        mu_list: Sequence[np.ndarray | None] | None = None,
        twist: float = 0,
        gap_layer_indices: Sequence[int] | None = None,
        N: int = 1,
        M: int = 1,
        a: float = 1.0,
        ax: float | None = None,
        ay: float | None = None,
        e_r: complex = 1,
        e_t: complex = 1,
        m_r: complex = 1,
        m_t: complex = 1,
        verbose: bool = False,
    ):
        if len(epsr_list) != len(thickness_list):
            raise ValueError("epsr_list and thickness_list must have the same length")
        self.N, self.M, self.NM = N, M, (2 * N + 1) * (2 * M + 1)
        self.ER = [convmat2D(e, N, M) if e is not None else None for e in epsr_list]
        if mu_list is None:
            self.UR = [np.eye(self.NM)] * len(self.ER)
        else:
            self.UR = [convmat2D(u, N, M) if u is not None else None for u in mu_list]
        self.layer_thicknesses = list(thickness_list)
        self.orientations = list(orientation_list) if orientation_list is not None else [1] * len(self.ER)
        if twist != 0 and any(o not in (1, 2) for o in self.orientations):
            raise ValueError("orientation_list entries must be 1 or 2")
        self.twist = twist
        self.gap_layer_indices = set(gap_layer_indices or ())
        self.gap_layer_indices.update(i for i, e in enumerate(self.ER) if e is None)
        self.a = a
        self.ax = a if ax is None else ax
        self.ay = a if ay is None else ay
        self.e_r, self.e_t, self.m_r, self.m_t = e_r, e_t, m_r, m_t
        self.verbose = verbose
        self.Sg = None  # global scattering matrix; depends on frequency and k
        self._reset_intermediates()

    @property
    def is_twisted(self) -> bool:
        return self.twist != 0

    @property
    def num_orders(self) -> int:
        """Number of plane waves (per polarization) in the expansion."""
        return self.NM**2 if self.is_twisted else self.NM

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _reset_intermediates(self):
        # quantities saved per layer for reconstructing internal fields
        self.internal_Smats = []
        self.internal_lambdas = []
        self.internal_Ws = []
        self.internal_Vs = []
        self._has_intermediates = False

    # ------------------------------------------------------------------ setup

    def set_freq_k(self, freq, theta_phi=None, kxy_inc=None):
        """Set the frequency and incident direction.

        Parameters
        ----------
        freq : float
            Frequency in units of ``c / a``.
        theta_phi : (float, float), optional
            Polar and azimuthal angle of incidence, in radians.
        kxy_inc : (float, float), optional
            In-plane incident wavevector normalized by ``k0``; used if ``theta_phi`` is
            not given.
        """
        if theta_phi is None and kxy_inc is None:
            raise ValueError("provide either theta_phi or kxy_inc")
        self._log("setting freq k...")
        self.Sg = None
        self._reset_intermediates()
        self.k0 = 2 * np.pi * freq / self.a

        n_i = np.sqrt(self.e_r * self.m_r)
        if theta_phi is not None:
            theta, phi = theta_phi
            self.kx_inc = n_i * np.sin(theta) * np.cos(phi)
            self.ky_inc = n_i * np.sin(theta) * np.sin(phi)
            self.kz_inc = np.sqrt(n_i**2 - self.kx_inc**2 - self.ky_inc**2)
            self.theta, self.phi = theta, phi
        else:
            self.kx_inc, self.ky_inc = kxy_inc
            self.kz_inc = np.sqrt(n_i**2 - self.kx_inc**2 - self.ky_inc**2)
            self.theta, self.phi = np.arccos(self.kz_inc / n_i), np.arctan2(self.ky_inc, self.kx_inc)

        if self.is_twisted:
            # Precompute per-block K matrices and gap modes for both lattice orientations.
            self._gap_blocks = {1: [], 2: []}
            for m in range(-self.M, self.M + 1):
                for n in range(-self.N, self.N + 1):
                    for orientation in (1, 2):
                        Kx, Ky = k_matrix_given_mn(
                            self.kx_inc,
                            self.ky_inc,
                            self.k0,
                            self.ax,
                            self.ay,
                            self.N,
                            self.M,
                            n,
                            m,
                            angle=self.twist,
                            layer=orientation,
                        )
                        _, Vg, Kzg = homogeneous_modes(Kx, Ky)
                        self._gap_blocks[orientation].append((Kx, Ky, Vg, Kzg))
            self.Kx, self.Ky, self.kzr, self.kzt = k_expanded(
                self.kx_inc,
                self.ky_inc,
                self.k0,
                self.ax,
                self.ay,
                self.N,
                self.M,
                self.twist,
                self.e_r,
                self.e_t,
                self.m_r,
                self.m_t,
            )
        else:
            self.Kx, self.Ky = k_matrix_2d(self.kx_inc, self.ky_inc, self.k0, self.ax, self.ay, self.N, self.M)
            _, Vg, Kzg = homogeneous_modes(self.Kx, self.Ky)
            self._gap_blocks = {1: [(self.Kx, self.Ky, Vg, Kzg)]}
            self.kzr = homogeneous_modes(self.Kx, self.Ky, self.e_r, self.m_r)[2]
            self.kzt = homogeneous_modes(self.Kx, self.Ky, self.e_t, self.m_t)[2]

    # ------------------------------------------------------- scattering matrix

    def _blocks(self, orientation):
        return self._gap_blocks[orientation if self.is_twisted else 1]

    def _expand_s(self, s_list, orientation):
        return expand_smatrices(s_list, self.NM, orientation) if self.is_twisted else s_list[0]

    def solve_smatrix(self, store_intermediate=False):
        """Compute the global scattering matrix ``self.Sg`` of the whole stack.

        If ``store_intermediate`` is set, also store the partial scattering matrices and the
        layer eigenmodes needed by :meth:`get_internal_field`.
        """
        if not hasattr(self, "k0"):
            raise RuntimeError("call set_freq_k before solving")
        self._log(f"solving Smat {'4D' if self.is_twisted else '2D'}...")
        self._reset_intermediates()
        Wg = np.eye(2 * self.NM)

        def half_space_smatrix(orientation, e, m, s_func):
            s_list = []
            for Kx, Ky, Vg, _ in self._blocks(orientation):
                V = Vg if (e == 1 and m == 1) else homogeneous_modes(Kx, Ky, e, m)[1]
                s_list.append(s_func(*ab_matrices(Wg, Wg, Vg, V)))
            return self._expand_s(s_list, orientation)

        S_ref = half_space_smatrix(1, self.e_r, self.m_r, s_reflection)
        S_trans = half_space_smatrix(2, self.e_t, self.m_t, s_transmission)

        if store_intermediate:
            self.internal_Wg, self.internal_Vg, _ = homogeneous_modes(self.Kx, self.Ky)

        Sg = S_ref
        for i, e_conv in enumerate(self.ER):
            orientation = self.orientations[i]
            if store_intermediate:
                self.internal_Smats.append(Sg)  # S-matrix of everything before layer i
            s_list, Ws, Vs, lambdas = [], [], [], []
            for Kx, Ky, Vg, Kzg in self._blocks(orientation):
                if i in self.gap_layer_indices:
                    W_i, V_i = Wg, Vg
                    Lambda = 1j * np.block([[Kzg, np.zeros_like(Kzg)], [np.zeros_like(Kzg), Kzg]])
                else:
                    P, Q = pq_matrices(Kx, Ky, e_conv, self.UR[i])
                    W_i, Lambda, V_i = eigen_modes(P, Q)
                A, B = ab_matrices(W_i, Wg, V_i, Vg)
                s_list.append(s_layer(A, B, self.layer_thicknesses[i], self.k0, Lambda))
                Ws.append(W_i)
                Vs.append(V_i)
                lambdas.append(np.diag(Lambda))
            Sg = redheffer_star(Sg, self._expand_s(s_list, orientation))

            if store_intermediate:
                if self.is_twisted:
                    self.internal_lambdas.append(expand_vectors(lambdas, self.NM, orientation))
                    self.internal_Ws.append(expand_matrices(Ws, self.NM, orientation))
                    self.internal_Vs.append(expand_matrices(Vs, self.NM, orientation))
                else:
                    self.internal_lambdas.append(lambdas[0])
                    self.internal_Ws.append(Ws[0])
                    self.internal_Vs.append(Vs[0])

        self.Sg = redheffer_star(Sg, S_trans)
        self._has_intermediates = store_intermediate

    # Backward-compatible aliases
    def solve_Smat_2D(self):
        self.solve_smatrix()

    def solve_Smat_4D(self):
        self.solve_smatrix()

    def solve_Smat_2D_store_intermediate(self):
        self.solve_smatrix(store_intermediate=True)

    def solve_Smat_4D_store_intermediate(self):
        self.solve_smatrix(store_intermediate=True)

    # -------------------------------------------------------------- outputs

    def _incident_coefficients(self, pte, ptm):
        """Mode amplitudes (Ex, Ey) of a unit plane wave in the 0th diffraction order."""
        normal = np.array([0, 0, -1])  # +z points into the stack
        k_inc = np.array([self.kx_inc, self.ky_inc, self.kz_inc])
        if self.theta != 0:
            a_te = np.cross(k_inc, normal)
            a_te = a_te / np.linalg.norm(a_te)
        else:
            a_te = np.array([0, 1, 0])
        a_tm = np.cross(a_te, k_inc)
        a_tm = a_tm / np.linalg.norm(a_tm)
        pol = pte * a_te + ptm * a_tm

        cinc = np.zeros(2 * self.NM, dtype=complex)
        cinc[self.NM // 2] = pol[0]
        cinc[self.NM + self.NM // 2] = pol[1]
        if self.is_twisted:
            # place the source in the 0th order of the other lattice as well
            cinc_expanded = np.zeros(2 * self.NM**2, dtype=complex)
            cinc_expanded[expanded_indices(self.NM, 1)[self.NM // 2]] = cinc
            cinc = cinc_expanded
        return cinc.reshape(-1, 1)

    def get_RT(self, pte, ptm, storing_all_orders=True, storing_intermediate_Smats=False, cinc_overwrite=None):
        """Reflection and transmission for a plane wave incident in the 0th order.

        Parameters
        ----------
        pte, ptm : complex
            TE and TM amplitudes of the incident wave.
        storing_all_orders : bool
            Keep the complex amplitudes of all diffraction orders in ``self.reflected``,
            ``self.transmitted``, ``self.rz`` and ``self.tz`` (needed for field reconstruction).
        storing_intermediate_Smats : bool
            Store per-layer data needed by :meth:`get_internal_field`.
        cinc_overwrite : ndarray, optional
            Custom incident mode amplitudes, of length ``2 * num_orders`` (Ex then Ey).

        Returns
        -------
        (R, T) : (float, float)
            Total reflected and transmitted power, normalized to the incident power.
        (r0, t0) : (ndarray, ndarray)
            Complex ``(Ex, Ey, Ez)`` amplitudes of the 0th-order reflected and transmitted waves.
        """
        if self.Sg is None or (storing_intermediate_Smats and not self._has_intermediates):
            self.solve_smatrix(store_intermediate=storing_intermediate_Smats)

        n = self.num_orders
        if cinc_overwrite is None:
            cinc = self._incident_coefficients(pte, ptm)
        else:
            cinc = np.asarray(cinc_overwrite).reshape(2 * n, 1)
        self.cinc = cinc

        reflected = self.Sg["S11"] @ cinc
        transmitted = self.Sg["S21"] @ cinc
        rx, ry = reflected[:n], reflected[n:]
        tx, ty = transmitted[:n], transmitted[n:]
        # longitudinal components from div E = 0
        rz = np.linalg.solve(self.kzr, self.Kx @ rx + self.Ky @ ry)
        tz = np.linalg.solve(self.kzt, self.Kx @ tx + self.Ky @ ty)
        r_sq = np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2
        t_sq = np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2

        # diffraction efficiency of each order
        self.R_orders = (np.real(self.kzr) @ r_sq).ravel() / np.real(self.kz_inc)
        self.T_orders = (np.real(self.kzt) @ t_sq).ravel() / np.real(self.kz_inc)
        R, T = np.sum(self.R_orders), np.sum(self.T_orders)
        self._log(f"got R, T, R+T: {R}, {T}, {R + T}")

        if storing_all_orders:
            self.reflected, self.transmitted = reflected, transmitted
            self.rz, self.tz = rz, tz

        c = n // 2
        r0 = np.array([rx[c, 0], ry[c, 0], rz[c, 0]])
        t0 = np.array([tx[c, 0], ty[c, 0], tz[c, 0]])
        return (R, T), (r0, t0)

    def _require_intermediates(self):
        if not self._has_intermediates or not hasattr(self, "reflected"):
            raise RuntimeError("call get_RT(..., storing_intermediate_Smats=True) first")

    def get_internal_field(self, which_layers=None, offsets=None):
        """Fourier coefficients of the fields inside the stack.

        Must be called after ``get_RT(..., storing_intermediate_Smats=True)``.

        Parameters
        ----------
        which_layers : sequence of int, optional
            0-based layer indices. Defaults to all layers.
        offsets : sequence of float, optional
            Depth inside each layer, measured from its top interface. Defaults to 0.

        Returns
        -------
        list of ndarray
            For each requested position, the concatenated coefficients
            ``[Ex, Ey, Ez, Hx, Hy, Hz]`` (H is scaled by the vacuum impedance).
        """
        self._require_intermediates()
        if which_layers is None:
            which_layers = list(range(len(self.layer_thicknesses)))
        if offsets is None:
            offsets = [0] * len(which_layers)

        Wg, Vg = self.internal_Wg, self.internal_Vg
        to_fourier_gap = np.block([[Wg, Wg], [-Vg, Vg]])
        inv_cache = {}
        fields = []
        for layer, z in zip(which_layers, offsets):
            self._log(f"getting internal field for layer {layer} offset {z}")
            S = self.internal_Smats[layer]
            # mode amplitudes in the gap medium just before this layer
            c_minus = np.linalg.solve(S["S12"], self.reflected - S["S11"] @ self.cinc)
            c_plus = S["S21"] @ self.cinc + S["S22"] @ c_minus

            W, V, lam = self.internal_Ws[layer], self.internal_Vs[layer], self.internal_lambdas[layer]
            to_fourier = np.block([[W, W], [-V, V]])
            mode_coeff = np.linalg.solve(to_fourier, to_fourier_gap @ np.concatenate([c_plus, c_minus]))
            phase = np.exp(np.concatenate([-lam * self.k0 * z, lam * self.k0 * z]))
            sx, sy, ux, uy = (to_fourier @ (phase[:, None] * mode_coeff)).reshape(4, -1)

            if layer not in inv_cache:
                inv_cache[layer] = (
                    self._expanded_inverse(self.ER[layer], layer),
                    self._expanded_inverse(self.UR[layer], layer),
                )
            econv_inv, mconv_inv = inv_cache[layer]
            sz = -1j * econv_inv @ (self.Kx @ uy - self.Ky @ ux)
            uz = -1j * mconv_inv @ (self.Kx @ sy - self.Ky @ sx)
            fields.append(np.concatenate([sx, sy, sz, ux, uy, uz]))
        return fields

    def _expanded_inverse(self, conv, layer):
        conv_inv = np.eye(self.NM) if conv is None else np.linalg.inv(conv)
        if not self.is_twisted:
            return conv_inv
        if self.orientations[layer] == 1:
            return np.kron(conv_inv, np.eye(self.NM))
        return np.kron(np.eye(self.NM), conv_inv)

    def get_RT_field(self):
        """Fourier coefficients ``[Ex, Ey, Ez, Hx, Hy, Hz]`` of the reflected and transmitted fields.

        Must be called after ``get_RT(..., storing_intermediate_Smats=True)``.
        """
        self._require_intermediates()
        W, V = self.internal_Wg, self.internal_Vg
        cinc = np.linalg.solve(W, self.cinc)
        to_fourier = np.vstack([W, V])
        erx, ery, hrx, hry = (to_fourier @ self.Sg["S11"] @ cinc).reshape(4, -1)
        etx, ety, htx, hty = (to_fourier @ self.Sg["S21"] @ cinc).reshape(4, -1)
        erz = np.linalg.solve(self.kzr, self.Kx @ erx + self.Ky @ ery)
        hrz = np.linalg.solve(self.kzr, self.Kx @ hrx + self.Ky @ hry)
        etz = np.linalg.solve(self.kzt, self.Kx @ etx + self.Ky @ ety)
        htz = np.linalg.solve(self.kzt, self.Kx @ htx + self.Ky @ hty)
        return np.concatenate([erx, ery, erz, hrx, hry, hrz]), np.concatenate([etx, ety, etz, htx, hty, htz])

    def get_Stress_tensor(self, which_layers=None, offsets=None):
        """Plane-averaged Maxwell stress components ``[Tx, Ty, Tz]`` at the given positions.

        Currently only valid inside vacuum (gap) layers.
        """
        tensors = []
        for field in self.get_internal_field(which_layers, offsets):
            sx, sy, sz, ux, uy, uz = field.reshape(6, -1)
            Tx = np.real(sx * np.conj(sz) + ux * np.conj(uz))
            Ty = np.real(sy * np.conj(sz) + uy * np.conj(uz))
            Tz = 0.5 * np.real(
                sz * np.conj(sz)
                + uz * np.conj(uz)
                - np.abs(sx) ** 2
                - np.abs(sy) ** 2
                - np.abs(ux) ** 2
                - np.abs(uy) ** 2
            )
            tensors.append([np.sum(Tx), np.sum(Ty), np.sum(Tz)])
        return tensors


#: Backward-compatible lowercase alias.
rcwa = RCWA
