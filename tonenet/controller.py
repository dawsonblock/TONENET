"""
Classical PI controller for adaptive parameter adjustment.

No quantum mysticism - just honest signal processing.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class ClassicalController:
    """
    Standard PI controller for adaptive parameter adjustment.
    Replaces 'consciousness field coupling' with honest signal processing.
    """

    def __init__(
        self,
        k_p: float = 0.1,
        k_i: float = 0.01,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Args:
            k_p: Proportional gain
            k_i: Integral gain
            bounds: Optional parameter bounds dict, e.g. {'f0': (50, 8000)}
        """
        self.k_p = k_p
        self.k_i = k_i
        self.bounds = bounds or {}
        self.integral: Dict[str, float] = {}

    def adapt(
        self,
        theta: Dict[str, float],
        error: Dict[str, float],
        dt: float = 1.0
    ) -> Dict[str, float]:
        """
        Adapt parameters based on error signal.

        Args:
            theta: current parameters (e.g., {'f0': 440.0, 'amp': 0.8})
            error: error signals (e.g., {'f0': 10.0, 'amp': -0.1})
            dt: time step

        Returns:
            theta_new: adapted parameters
        """
        theta_new = {}

        for key, current in theta.items():
            # Initialize integral
            if key not in self.integral:
                self.integral[key] = 0.0

            # PI control
            self.integral[key] += error.get(key, 0.0) * dt
            delta = self.k_p * error.get(key, 0.0) + self.k_i * self.integral[key]

            # Update with bounds
            new_val = current + delta
            if key in self.bounds:
                low, high = self.bounds[key]
                new_val = np.clip(new_val, low, high)
                # Anti-windup
                if new_val == low or new_val == high:
                    self.integral[key] *= 0.9

            theta_new[key] = float(new_val)

        return theta_new

    def reset(self):
        """Reset integral accumulators."""
        self.integral = {}
