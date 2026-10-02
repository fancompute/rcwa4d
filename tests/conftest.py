import numpy as np
import pytest


def hole_pattern(eps_slab, radius, n=64, center=(0.0, 0.0), eps_hole=1.0):
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, n), np.linspace(-0.5, 0.5, n))
    eps = np.full((n, n), float(eps_slab))
    eps[(xs - center[0]) ** 2 + (ys - center[1]) ** 2 < radius**2] = eps_hole
    return eps


@pytest.fixture
def eps_a():
    return hole_pattern(4, 0.25)


@pytest.fixture
def eps_b():
    return hole_pattern(6, 0.2, center=(0.1, 0.0), eps_hole=1.5)
