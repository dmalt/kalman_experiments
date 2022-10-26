from typing import Any

import numpy as np
import numpy.typing as npt

Vec1D = npt.NDArray[np.floating[Any]]  # one-dimensional vector of shape(n,)
Cov = npt.NDArray[np.floating[Any]]  # covariance matrix of shape(n, n)
Vec = npt.NDArray[np.floating[Any]]  # column vector of shape(n, 1)
Mat = npt.NDArray[np.floating[Any]]  # matrix of shape(n, m)

Vec2 = npt.NDArray[np.floating[Any]]  # 2-dimensional vector of shape(2, 1)
Mat2 = npt.NDArray[np.floating[Any]]  # 2-dimensional vector of shape(2, 2)

Timeseries = npt.NDArray[np.number]  # one-dimensional timeseries vector of shape(n_samp,)
