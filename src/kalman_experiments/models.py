from __future__ import annotations

from cmath import exp
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from kalman_experiments.numpy_types import ComplexTimeseries, RealTimeseries

from .complex import complex_randn


class SignalGenerator(Protocol):
    def step(self) -> float:
        """Generate single noise sample"""
        ...


@dataclass
class SingleRhythmModel:
    freq: float
    A: float
    s: float
    sr: float
    x: complex = 0

    def __post_init__(self):
        self.Phi = self.A * exp(2 * np.pi * self.freq / self.sr * 1j)

    def step(self) -> float:
        """Update model state and generate measurement"""
        self.x = self.Phi * self.x + complex_randn() * self.s
        return self.x.real


class StatefulSignalGenerator(SignalGenerator, Protocol):
    @property
    def x(self) -> complex:
        ...


class ModelAdapter:
    def __init__(self, model: StatefulSignalGenerator):
        self.model = model

    def collect_states_and_meas(self, n_samp: int) -> tuple[ComplexTimeseries, RealTimeseries]:
        states = np.zeros(n_samp, dtype=complex)
        meas = np.zeros(n_samp)
        for i in range(n_samp):
            meas[i] = self.model.step()
            states[i] = self.model.x
        return states, meas
