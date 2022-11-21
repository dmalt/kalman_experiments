from typing import Any, NewType, TypeGuard

import numpy as np
import numpy.typing as npt

Vec1D = npt.NDArray[np.floating[Any]]  # one-dimensional vector of shape(n,)
Cov = npt.NDArray[np.floating[Any]]  # covariance matrix of shape(n, n)
Vec = npt.NDArray[np.floating[Any]]  # column vector of shape(n, 1)
Mat = npt.NDArray[np.floating[Any]]  # matrix of shape(n, m)

Vec2 = npt.NDArray[np.floating[Any]]  # 2-dimensional vector of shape(2, 1)
Mat2 = npt.NDArray[np.floating[Any]]  # 2-dimensional vector of shape(2, 2)

Timeseries = npt.NDArray[np.number]  # one-dimensional timeseries vector of shape(n_samp,)


NonNegativeInt = NewType("NonNegativeInt", int)
NonNegativeFloat = NewType("NonNegativeFloat", float)
PositiveInt = NewType("PositiveInt", NonNegativeInt)
PositiveFloat = NewType("PositiveFloat", NonNegativeFloat)
Alpha = NewType("Alpha", float)


def is_positive_int(num: int) -> TypeGuard[PositiveInt]:
    return num > 0


def is_positive_float(num: float) -> TypeGuard[PositiveFloat]:
    return num > 0


def is_nonneg_int(num: int) -> TypeGuard[NonNegativeInt]:
    return num >= 0


def is_nonneg_float(num: float) -> TypeGuard[NonNegativeFloat]:
    return num >= 0


def check_positive_int(num: int):
    if is_positive_int(num):
        return num
    raise ValueError(f"Number must be positive; got {num}")


def check_positive_float(num: float):
    if is_positive_float(num):
        return num
    raise ValueError(f"Number must be positive; got {num}")


def is_nonnegative_int(num: int) -> TypeGuard[NonNegativeInt]:
    return num >= 0


def is_nonnegative_float(num: float) -> TypeGuard[NonNegativeFloat]:
    return num >= 0


def check_nonnegative_int(num: int) -> NonNegativeInt:
    if is_nonnegative_int(num):
        return NonNegativeInt(num)
    raise ValueError(f"Number must be nonnegative; got {num}")


def check_nonnegative_float(num: float) -> NonNegativeFloat:
    if is_nonnegative_float(num):
        return NonNegativeFloat(num)
    raise ValueError(f"Number must be nonnegative; got {num}")


def is_in_alpha_range(alpha: float) -> TypeGuard[Alpha]:
    return -2 <= alpha <= 2


def check_in_alpha_range(num: float) -> Alpha:
    if is_in_alpha_range(num):
        return num
    raise ValueError(f"Alpha must be in [-2, 2] range; got {num}")
