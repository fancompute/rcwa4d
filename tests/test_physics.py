import numpy as np
import pytest

from rcwa4d import RCWA, convmat2D, rcwa
from rcwa4d.smatrix import expanded_indices

DEG = np.pi / 180


def slab_reflectance(n_slab, n_sub, thickness, freq):
    """Analytic normal-incidence reflectance of a slab on a substrate, light from vacuum."""
    r12 = (1 - n_slab) / (1 + n_slab)
    r23 = (n_slab - n_sub) / (n_slab + n_sub)
    phase = np.exp(2j * n_slab * 2 * np.pi * freq * thickness)
    return abs((r12 + r23 * phase) / (1 + r12 * r23 * phase)) ** 2


@pytest.mark.parametrize("twist", [0, 10 * DEG])
@pytest.mark.parametrize("e_t", [1, 2.25])
def test_uniform_slab_matches_fresnel(twist, e_t):
    eps = np.full((16, 16), 4.0)
    sim = RCWA([eps, eps], [0.15, 0.1], [1, 2], twist=twist, N=1, M=1, e_t=e_t)
    sim.set_freq_k(0.6, (0, 0))
    (R, T), _ = sim.get_RT(pte=1, ptm=0)
    R_exact = slab_reflectance(2.0, np.sqrt(e_t), 0.25, 0.6)
    assert R == pytest.approx(R_exact, abs=1e-10)
    assert R + T == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize("twist", [0, 10 * DEG])
@pytest.mark.parametrize("e_t", [1, 2.25])
def test_energy_conservation(eps_a, eps_b, twist, e_t):
    sim = RCWA([eps_a, None, eps_b], [0.2, 0.1, 0.2], [1, 1, 2], twist=twist, N=1, M=1, e_t=e_t)
    for freq in (0.65, 0.8):
        sim.set_freq_k(freq, (6 * DEG, 15 * DEG))
        (R, T), _ = sim.get_RT(pte=0.6, ptm=0.8)
        assert R + T == pytest.approx(1, abs=1e-8)
        assert np.all(sim.R_orders >= -1e-12) and np.all(sim.T_orders >= -1e-12)


@pytest.mark.parametrize("twist", [0, 10 * DEG])
def test_internal_field_tangential_continuity(eps_a, eps_b, twist):
    """Ex, Ey, Hx, Hy must be continuous across every interface inside the stack."""
    thicknesses = [0.2, 0.1, 0.2]
    sim = RCWA([eps_a, None, eps_b], thicknesses, [1, 1, 2], twist=twist, N=1, M=1)
    sim.set_freq_k(0.75, (5 * DEG, 0))
    sim.get_RT(1, 0, storing_intermediate_Smats=True)
    bottoms = sim.get_internal_field([0, 1], thicknesses[:2])
    tops = sim.get_internal_field([1, 2], [0, 0])
    for bottom, top in zip(bottoms, tops):
        ex, ey, _, hx, hy, _ = bottom.reshape(6, -1)
        ex2, ey2, _, hx2, hy2, _ = top.reshape(6, -1)
        np.testing.assert_allclose(np.concatenate([ex, ey, hx, hy]), np.concatenate([ex2, ey2, hx2, hy2]), atol=1e-8)


def test_internal_field_requires_intermediates(eps_a):
    sim = RCWA([eps_a], [0.2])
    sim.set_freq_k(0.7, (0, 0))
    sim.get_RT(0, 1)
    with pytest.raises(RuntimeError):
        sim.get_internal_field()


def test_resolving_after_frequency_change_resets_intermediates(eps_a):
    sim = RCWA([eps_a, None], [0.2, 0.1])
    for freq in (0.6, 0.7):
        sim.set_freq_k(freq, (0, 0))
        sim.get_RT(0, 1, storing_intermediate_Smats=True)
        assert len(sim.internal_Smats) == 2


def test_kxy_and_theta_phi_agree(eps_a):
    sim = RCWA([eps_a], [0.2])
    theta, phi = 12 * DEG, 30 * DEG
    sim.set_freq_k(0.7, theta_phi=(theta, phi))
    rt1, _ = sim.get_RT(1, 0)
    sim.set_freq_k(0.7, kxy_inc=(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)))
    rt2, _ = sim.get_RT(1, 0)
    np.testing.assert_allclose(rt1, rt2, atol=1e-12)


def test_convmat_of_uniform_material_is_scaled_identity():
    C = convmat2D(np.full((32, 32), 3.5), 2, 1)
    np.testing.assert_allclose(C, 3.5 * np.eye(15), atol=1e-12)


@pytest.mark.parametrize("orientation", [1, 2])
def test_expanded_indices_is_a_permutation(orientation):
    NM = 9
    idx = expanded_indices(NM, orientation)
    assert sorted(idx.ravel()) == list(range(2 * NM * NM))


def test_lowercase_alias():
    assert rcwa is RCWA
