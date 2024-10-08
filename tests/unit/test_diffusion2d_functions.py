"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import numpy as np


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    w = 5.
    h = 5.
    dx = 0.05
    dy = 0.05
    solver = SolveDiffusion2D()
    solver.initialize_domain(w, h, dx, dy)
    assert solver.nx == w/dx, "nx should be 100"
    assert solver.ny == h/dy, "ny should be 100"


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    d = 5.
    T_cold = 500.
    T_hot = 1000.
    solver.dx = 100.
    solver.dy = 100.
    solver.initialize_physical_parameters(d, T_cold, T_hot)

    assert solver.dt == 500, "dt should be 500"


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    T_cold = 500.
    T_hot = 1000.
    nx = 100
    ny = 100
    w = 5.
    h = 5.
    dx = 100.
    dy = 100.

    solver = SolveDiffusion2D()
    solver.T_cold = T_cold
    solver.T_hot = T_hot
    solver.nx = nx
    solver.ny = ny
    solver.w = w
    solver.h = h
    solver.dx = dx
    solver.dy = dy

    u = T_cold * np.ones((nx, ny))

    # Initial conditions - circle of radius r centred at (cx,cy) (mm)
    r = min(h, w) / 4.0
    cx = w / 2.0
    cy = h / 2.0
    r2 = r ** 2
    for i in range(nx):
        for j in range(ny):
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                u[i, j] = T_hot

    assert np.array_equal(solver.set_initial_condition(), u)
