"""Results must match the values produced by the original (pre-refactor) implementation."""

import numpy as np
import pytest

from rcwa4d import RCWA

DEG = np.pi / 180

# name: (R, T, r0, t0)
REFERENCE = {
    "2d_single": (
        0.5627227396022193,
        0.43727726039778136,
        [(-0.683221456209 + 0.309727592213j), 0j, 0j],
        [(-0.273030466073 - 0.60227205231j), (-0 + 0j), 0j],
    ),
    "2d_oblique": (
        0.32964697911475715,
        0.6703530208852438,
        [(-0.191242567276 + 0.030940234902j), (-0.532055954755 + 0.05608905705j), (-0.076111347073 + 0.009669693677j)],
        [(0.038920290183 + 0.333371745501j), (0.07227700973 + 0.734135773585j), (0.012315463613 + 0.11563105276j)],
    ),
    "2d_gap": (
        0.3576827793905616,
        0.6423172206094205,
        [(-0 + 0j), (-0.357354047021 - 0.479563201745j), (-0 + 0j)],
        [-0j, (0.01982874449 + 0.801201623501j), -0j],
    ),
    "4d_bilayer": (
        0.07356619985858759,
        0.9264338001414117,
        [(-0.006446784499 - 0.087321957055j), (-0 + 0j), 0j],
        [(-0.917694840976 + 0.215168794026j), (-0.010588729924 + 0.039367066041j), 0j],
    ),
    "4d_gap_oblique": (
        0.6183020830218098,
        0.3816979169781896,
        [(-0.368303860269 - 0.185690008763j), (-0.605955004527 - 0.263003284714j), (-0.077767039351 - 0.037165173763j)],
        [(-0.084955230849 + 0.330335214609j), (-0.104216234461 + 0.497623270653j), (-0.016229073046 + 0.067545435396j)],
    ),
}


def make_case(name, eps_a, eps_b):
    if name == "2d_single":
        return RCWA([eps_a], [0.2], N=1, M=1), 0.75, (0, 0), (0, 1)
    if name == "2d_oblique":
        return RCWA([eps_a, eps_b], [0.2, 0.3], [1, 1], N=2, M=1), 0.7, (10 * DEG, 30 * DEG), (0.6, 0.8)
    if name == "2d_gap":
        sim = RCWA([eps_a, None, eps_b], [0.2, 0.1, 0.2], [1, 1, 1], gap_layer_indices=[1], N=1, M=2)
        return sim, 0.8, (5 * DEG, 0), (1, 0)
    if name == "4d_bilayer":
        return RCWA([eps_a, eps_a], [0.2, 0.2], [1, 2], twist=10 * DEG, N=1, M=1), 0.75, (0, 0), (0, 1)
    if name == "4d_gap_oblique":
        sim = RCWA([eps_a, None, eps_b], [0.2, 0.1, 0.2], [1, 1, 2], gap_layer_indices=[1], twist=7 * DEG, N=1, M=1)
        return sim, 0.72, (8 * DEG, 20 * DEG), (0.6, 0.8)
    raise KeyError(name)


@pytest.mark.parametrize("name", REFERENCE)
def test_matches_reference(name, eps_a, eps_b):
    sim, freq, angles, (pte, ptm) = make_case(name, eps_a, eps_b)
    sim.set_freq_k(freq, angles)
    (R, T), (r0, t0) = sim.get_RT(pte, ptm)
    R_ref, T_ref, r0_ref, t0_ref = REFERENCE[name]
    np.testing.assert_allclose([R, T], [R_ref, T_ref], atol=1e-9)
    np.testing.assert_allclose(r0, r0_ref, atol=1e-9)
    np.testing.assert_allclose(t0, t0_ref, atol=1e-9)
