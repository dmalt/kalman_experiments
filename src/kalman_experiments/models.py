from __future__ import annotations

from cmath import exp
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .complex import complex_randn


class NoiseGenerator(Protocol):
    def step(self) -> float:
        """Generate single noise sample"""
        ...


@dataclass
class SingleRhythmModel:
    freq: float
    A: float
    sigma: float
    sr: float
    meas_noise: NoiseGenerator
    x: complex = 0

    def __post_init__(self):
        self.Phi = self.A * exp(2 * np.pi * self.freq / self.sr * 1j)

    def step(self) -> float:
        """Update model state and generate measurement"""
        self.x = self.Phi * self.x + complex_randn() * self.sigma
        return self.x.real + self.meas_noise.step()
