import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PIDGains:
    kp: float = 0.07
    ki: float = 0.0008
    kd: float = 0.12
    heading: float = 0.5
    deadband: float = 0.02


class PIDController:
    """
    Classical PID baseline respecting the same discrete action space:
    0 = steer left, 1 = steer right, 2 = hold.
    """

    def __init__(self, gains: PIDGains = PIDGains(), integral_limit: float = 120.0):
        self.gains = gains
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def act(self, offset, heading_error, turn_limit, dt=1 / 60.0):
        # PID on lateral offset plus heading stabilizer.
        self.integral += offset * dt
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        derivative = (offset - self.prev_error) / dt
        self.prev_error = offset

        steer = (
            self.gains.kp * offset
            + self.gains.kd * derivative
            + self.gains.ki * self.integral
            + self.gains.heading * heading_error
        )

        steer = float(np.clip(steer, -turn_limit, turn_limit))
        if steer > self.gains.deadband:
            return 1  # steer right
        if steer < -self.gains.deadband:
            return 0  # steer left
        return 2  # hold
