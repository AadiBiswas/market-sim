import math
from dataclasses import dataclass

import numpy as np

NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True, slots=True)
class FundamentalSnapshot:
    timestamp: int
    value: float


class FundamentalProcess:
    """Latent value as arithmetic Brownian motion: V += sigma * sqrt(dt) * Z.

    Value is in (fractional) ticks, sigma in ticks / sqrt(second), and time in
    integer nanoseconds like everything else. ABM can wander below zero and we
    don't clip it, so pick initial_value and sigma so that it practically won't.

    One draw per advance, so a given seed only reproduces a path on the same
    update schedule. Experiments comparing strategies need to hold that fixed.
    """

    def __init__(self, initial_value: float, sigma: float, seed: int) -> None:
        if not math.isfinite(initial_value):
            raise ValueError(f"initial_value must be finite, got {initial_value}")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError(f"sigma must be finite and non-negative, got {sigma}")
        self._value = float(initial_value)
        self._sigma = float(sigma)
        self._rng = np.random.default_rng(seed)
        self._timestamp = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def timestamp(self) -> int:
        return self._timestamp

    def advance_to(self, timestamp: int) -> float:
        if timestamp < self._timestamp:
            raise ValueError(f"can't advance back to t={timestamp}, process is at t={self._timestamp}")
        if timestamp > self._timestamp:
            dt_seconds = (timestamp - self._timestamp) / NS_PER_SECOND
            self._value += self._sigma * math.sqrt(dt_seconds) * float(self._rng.standard_normal())
            self._timestamp = timestamp
        return self._value
