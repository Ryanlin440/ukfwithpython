import math

import numpy as np

from simulation.physics import (
    DISTANCE_SENSOR_KEYS,
    SENSOR_DIRECTIONS,
    normalize_angle,
    ray_box_intersection,
    rotation_matrix,
)

# Channels whose residuals must be wrapped into [-pi, pi] instead of subtracted.
ANGULAR_CHANNELS = frozenset({'imu'})


class UKF:
    """Unscented Kalman filter over the pose state [x, y, theta].

    The filter is fed noisy odometry as its control input and corrects with
    whichever of the IMU / distance sensors are currently enabled. It never
    sees ground truth.
    """

    def __init__(self, state=None, covariance=1.0, heading_covariance=None, alpha=1.0,
                 beta=2.0, kappa=0.0, mount_offset=0.0, position_bound=None):
        self.state = np.zeros(3, dtype=float) if state is None else np.asarray(state, dtype=float)
        # Position and heading need separate initial variances: they are in
        # square inches and square radians respectively, and one scalar covering
        # both makes whichever unit it was not chosen for wildly wrong. An
        # over-large heading variance is especially damaging, because the sigma
        # points then span headings far apart and the averaged displacement they
        # predict comes out short.
        self.P = np.eye(3, dtype=float) * covariance
        if heading_covariance is not None:
            self.P[2, 2] = float(heading_covariance)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.mount_offset = mount_offset
        # Half-extent of the region the robot is known to occupy. This is map
        # knowledge, not ground truth, and it matters: outside the arena every
        # distance reading saturates, so the measurement model goes flat and a
        # large excursion would never be pulled back.
        self.position_bound = position_bound
        self.last_details = ""

    def set_state(self, x, y, theta):
        self.state = np.array([x, y, normalize_angle(theta)], dtype=float)
        self._project()

    def _project(self):
        if self.position_bound is not None:
            bound = float(self.position_bound)
            self.state[0] = min(bound, max(-bound, self.state[0]))
            self.state[1] = min(bound, max(-bound, self.state[1]))

    def _ensure_positive_definite(self, matrix, eps=1e-9):
        matrix = 0.5 * (matrix + matrix.T)
        min_eig = float(np.min(np.linalg.eigvalsh(matrix)))
        if min_eig <= eps:
            matrix = matrix + np.eye(matrix.shape[0], dtype=float) * (eps - min_eig)
            matrix = 0.5 * (matrix + matrix.T)
        return matrix

    def scaling_lambda(self, n):
        """Scaled-UT lambda, floored so that (n + lambda) never approaches zero."""
        lambda_val = self.alpha ** 2 * (n + self.kappa) - n
        if n + lambda_val < 1e-3:
            lambda_val = 1e-3 - n
        return lambda_val

    def sigma_points(self, mean, covariance):
        n = len(mean)
        lambda_val = self.scaling_lambda(n)
        scaled = self._ensure_positive_definite((n + lambda_val) * np.asarray(covariance, dtype=float))
        root = np.linalg.cholesky(scaled)

        points = np.zeros((2 * n + 1, n), dtype=float)
        points[0] = mean
        for i in range(n):
            points[i + 1] = mean + root[:, i]
            points[i + 1 + n] = mean - root[:, i]

        weights_mean = np.full(2 * n + 1, 1.0 / (2.0 * (n + lambda_val)), dtype=float)
        weights_cov = weights_mean.copy()
        weights_mean[0] = lambda_val / (n + lambda_val)
        weights_cov[0] = lambda_val / (n + lambda_val) + (1.0 - self.alpha ** 2 + self.beta)
        return points, weights_mean, weights_cov

    def propagate_state(self, sigma_state, control, dt):
        """Advance sigma points by a body-frame velocity control (vx, vy, omega).

        The body velocity is rotated by the heading at the *start* of the step,
        because that is the frame the odometry sensor resolved its measured
        displacement into. Using any other heading here (a midpoint, say)
        introduces a systematic rotation error whenever the robot is turning.
        """
        vx, vy, omega = control
        propagated = np.empty_like(sigma_state)
        for i, sample in enumerate(sigma_state):
            x, y, theta = sample
            theta = normalize_angle(theta)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            propagated[i, 0] = x + (cos_t * vx - sin_t * vy) * dt
            propagated[i, 1] = y + (sin_t * vx + cos_t * vy) * dt
            propagated[i, 2] = normalize_angle(theta + omega * dt)
        return propagated

    def circular_mean(self, angles, weights):
        s = float(np.sum(weights * np.sin(angles)))
        c = float(np.sum(weights * np.cos(angles)))
        if abs(s) < 1e-15 and abs(c) < 1e-15:
            return float(angles[0])
        return math.atan2(s, c)

    def predict(self, control, dt, process_noise):
        sigma_points, wm, wc = self.sigma_points(self.state, self.P)
        propagated = self.propagate_state(sigma_points, control, dt)

        predicted_mean = np.array([
            float(np.sum(wm * propagated[:, 0])),
            float(np.sum(wm * propagated[:, 1])),
            self.circular_mean(propagated[:, 2], wm),
        ], dtype=float)

        predicted_cov = np.zeros((3, 3), dtype=float)
        for i in range(len(propagated)):
            diff = propagated[i] - predicted_mean
            diff[2] = normalize_angle(diff[2])
            predicted_cov += wc[i] * np.outer(diff, diff)
        predicted_cov = self._ensure_positive_definite(predicted_cov + process_noise)

        self.state = predicted_mean
        self.P = predicted_cov
        self._project()
        return self.state.copy(), self.P.copy()

    def distance_sensor_measurement(self, state, sensor_dir, arena_size):
        """Geometric range from a sensor at `state` to the wall it faces.

        Deliberately not capped at the sensor's usable range. A channel only
        appears in the measurement vector when the real sensor got a reading, and
        clamping the prediction would flatten the gradient exactly where a sigma
        point disagrees with that reading.
        """
        direction = np.asarray(sensor_dir, dtype=float)
        R = rotation_matrix(state[2])
        world_dir = R @ direction
        world_dir = world_dir / max(np.linalg.norm(world_dir), 1e-12)
        origin = np.array([state[0], state[1]], dtype=float) + R @ (direction * self.mount_offset)
        return ray_box_intersection(origin, world_dir, arena_size)

    def measurement_layout(self, enabled_sensors, available=None):
        """Ordered channel names the measurement vector is expected to contain.

        The simulation must build its measurement vector in exactly this order;
        keeping both sides derived from one function is what prevents the vector
        and the model from silently disagreeing on length. `available` drops the
        distance channels that returned nothing this step, so a sensor going out
        of range shrinks the measurement rather than poisoning it.
        """
        layout = []
        if enabled_sensors.get('imu', False):
            layout.append('imu')
        for key in DISTANCE_SENSOR_KEYS:
            if enabled_sensors.get(key, False) and (available is None or available.get(key, True)):
                layout.append(key)
        return layout

    def measurement_model(self, state, enabled_sensors, arena_size, available=None):
        values = []
        for channel in self.measurement_layout(enabled_sensors, available):
            if channel == 'imu':
                values.append(normalize_angle(state[2]))
            else:
                values.append(self.distance_sensor_measurement(state, SENSOR_DIRECTIONS[channel], arena_size))
        return np.asarray(values, dtype=float)

    def update(self, measured_values, sensors_enabled, arena_size, measurement_noise, available=None):
        layout = self.measurement_layout(sensors_enabled, available)
        measured_values = np.asarray(measured_values, dtype=float)
        if not layout:
            return self.state.copy(), self.P.copy()
        if measured_values.shape != (len(layout),):
            raise ValueError(
                f"measurement vector has shape {measured_values.shape}, expected ({len(layout)},) for {layout}"
            )

        measurement_noise = np.asarray(measurement_noise, dtype=float)
        if measurement_noise.shape != (len(layout), len(layout)):
            raise ValueError(
                f"measurement noise has shape {measurement_noise.shape}, expected {(len(layout), len(layout))}"
            )

        angular = [i for i, channel in enumerate(layout) if channel in ANGULAR_CHANNELS]

        sigma_points, wm, wc = self.sigma_points(self.state, self.P)
        predicted_measurements = np.asarray(
            [self.measurement_model(p, sensors_enabled, arena_size, available) for p in sigma_points],
            dtype=float,
        )

        z_mean = np.zeros(len(layout), dtype=float)
        for i in range(len(layout)):
            column = predicted_measurements[:, i]
            z_mean[i] = self.circular_mean(column, wm) if i in angular else float(np.sum(wm * column))

        Pzz = np.zeros((len(layout), len(layout)), dtype=float)
        Pxz = np.zeros((3, len(layout)), dtype=float)
        for i in range(len(sigma_points)):
            delta_x = sigma_points[i] - self.state
            delta_x[2] = normalize_angle(delta_x[2])
            delta_z = predicted_measurements[i] - z_mean
            for idx in angular:
                delta_z[idx] = normalize_angle(delta_z[idx])
            Pzz += wc[i] * np.outer(delta_z, delta_z)
            Pxz += wc[i] * np.outer(delta_x, delta_z)

        Pzz = self._ensure_positive_definite(Pzz + measurement_noise)

        innovation = measured_values - z_mean
        for idx in angular:
            innovation[idx] = normalize_angle(innovation[idx])

        # Solve K @ Pzz = Pxz rather than inverting Pzz explicitly.
        try:
            K = np.linalg.solve(Pzz.T, Pxz.T).T
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        self.state = self.state + K @ innovation
        self.state[2] = normalize_angle(self.state[2])
        self._project()
        self.P = self._ensure_positive_definite(self.P - K @ Pzz @ K.T)
        return self.state.copy(), self.P.copy()
