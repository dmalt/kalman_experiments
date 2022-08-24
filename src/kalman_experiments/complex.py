import random

import numpy as np
import numpy.typing as npt

Vec2 = npt.NDArray  # array of shape(2,)
Mat2 = npt.NDArray  # array of shape(2, 2)


def complex_randn() -> complex:
    """Generate random complex number with real and imag parts sampled from N(0, 1)"""
    return random.gauss(0, 1) + 1j * random.gauss(0, 1)


def complex2vec(z: complex) -> Vec2:
    """Convert complex number to 2d vector"""
    return np.array([z.real, z.imag])


def vec2complex(v: Vec2) -> complex:
    """Convert 2d vector to a complex number"""
    return v[0] + 1j * v[1]


def complex2mat(z: complex) -> Mat2:
    """Convert complex number to 2x2 antisymmetrical matrix"""
    return np.array([[z.real, -z.imag], [z.imag, z.real]])


def mat2complex(M: Mat2) -> complex:
    """Convert complex number to 2x2 antisymmetrical matrix"""
    return M[0, 0] + 1j * M[1, 0]
