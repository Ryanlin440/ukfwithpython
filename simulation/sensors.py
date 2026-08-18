import math

import numpy as np

from simulation.physics import SENSOR_DIRECTIONS, normalize_angle, ray_box_intersection, rotation_matrix

# Value reported by a distance sensor whose target is beyond its usable range.
OUT_OF_RANGE = -1.0


class Sensor:
    def __init__(self, noise_std=0.0, bias=0.0):
        self.noise_std = noise_std
        self.bias = bias

    def add_noise(self, value, noise_std=None):
        std = self.noise_std if noise_std is None else noise_std
        if std <= 0.0:
            return value + self.bias
        return value + self.bias + np.random.normal(0.0, std)


class OdometrySensor(Sensor):
    """Two-axis dead-wheel odometry plus a heading source.

    One unpowered pod rolls along the robot's X axis and one along its Y axis,
    so both forward and lateral displacement are measured directly rather than
    inferred from the drive wheels.

    The lateral pod cannot sit exactly on the centre of rotation, and that
    matters: a pod mounted `lateral_offset` inches ahead of the centre is swept
    sideways whenever the robot turns, reading `lateral_offset * delta_theta`
    even when the chassis has not translated sideways at all. Whoever consumes
    this reading has to subtract that term back out to recover the chassis
    motion.
    """

    def __init__(self, noise_std=0.4, angular_noise_std=0.002, lateral_offset=5.0, bias=0.0):
        super().__init__(noise_std=noise_std, bias=bias)
        self.angular_noise_std = angular_noise_std
        self.lateral_offset = lateral_offset

    def sample(self, forward_delta, lateral_delta, delta_theta):
        true_forward = float(forward_delta)
        true_theta = float(normalize_angle(delta_theta))
        true_lateral = float(lateral_delta) + self.lateral_offset * true_theta
        return {
            "true_forward": true_forward,
            "true_lateral": true_lateral,
            "true_theta": true_theta,
            "noisy_forward": self.add_noise(true_forward),
            "noisy_lateral": self.add_noise(true_lateral),
            "noisy_theta": self.add_noise(true_theta, self.angular_noise_std),
        }


class IMUSensor(Sensor):
    """Absolute heading, as from a magnetometer or a fused IMU."""

    def __init__(self, noise_std=0.5 * math.pi / 180.0, bias=0.0):
        super().__init__(noise_std=noise_std, bias=bias)

    def sample(self, theta_true):
        return {
            "true": normalize_angle(theta_true),
            "noisy": normalize_angle(self.add_noise(theta_true)),
        }


class DistanceSensor(Sensor):
    """Forward-facing rangefinder with a finite usable range.

    Past `max_range` the sensor has nothing to report, so it returns
    OUT_OF_RANGE rather than a number. Callers must drop that channel from the
    measurement instead of feeding the sentinel to the filter.
    """

    def __init__(self, tag, noise_std=0.8, bias=0.0):
        super().__init__(noise_std=noise_std, bias=bias)
        self.tag = tag
        self.direction = SENSOR_DIRECTIONS[tag]

    def origin(self, robot_state, mount_offset):
        pose = np.asarray([robot_state["x"], robot_state["y"]], dtype=float)
        return pose + rotation_matrix(robot_state["theta"]) @ (self.direction * mount_offset)

    def world_direction(self, theta):
        world_dir = rotation_matrix(theta) @ self.direction
        return world_dir / max(np.linalg.norm(world_dir), 1e-12)

    def sample(self, robot_state, arena_size, mount_offset, max_range):
        origin = self.origin(robot_state, mount_offset)
        world_dir = self.world_direction(robot_state["theta"])
        distance = ray_box_intersection(origin, world_dir, arena_size)

        # The hardware only ever sees the noisy value, so that is what decides
        # whether the reading falls inside the usable range.
        noisy = float(max(0.0, self.add_noise(distance)))
        # Plain bool, not numpy's: np.bool_ is not JSON serializable and this
        # flag travels straight out through the API.
        in_range = bool(noisy <= max_range)

        return {
            "true": float(distance),
            "noisy": noisy if in_range else OUT_OF_RANGE,
            "in_range": in_range,
            "origin": origin,
            "direction": world_dir,
        }
