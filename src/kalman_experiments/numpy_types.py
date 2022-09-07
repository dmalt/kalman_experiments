from typing import Any

import numpy as np
import numpy.typing as npt

Vec1D = npt.NDArray[np.floating[Any]]  # of shape(n,)
Cov = npt.NDArray[np.floating[Any]]  # of shape(n, n)
Vec = npt.NDArray[np.floating[Any]]  # of shape(n, 1)
Mat = npt.NDArray[np.floating[Any]]  # of shape(n, m)

Vec2 = npt.NDArray[np.floating[Any]]  # of shape(2, 1)
Mat2 = npt.NDArray[np.floating[Any]]  # of shape(2, 2)

ComplexTimeseries = npt.NDArray[np.complex_]  # of shape(n_samp,)
RealTimeseries = npt.NDArray[np.floating]  # of shape(n_samp,)
