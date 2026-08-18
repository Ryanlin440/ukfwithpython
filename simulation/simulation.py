import math
import threading
import time

import numpy as np

from simulation.physics import (
    DISTANCE_SENSOR_KEYS,
    compute_covariance_ellipse,
    normalize_angle,
    rotation_matrix,
)
from simulation.robot import Robot
from simulation.sensors import OUT_OF_RANGE, DistanceSensor, IMUSensor, OdometrySensor
from simulation.ukf import UKF

# Every length in this simulation is in inches.
ARENA_HALF = 72.0          # 12 ft x 12 ft field, measured from the centre
ROBOT_SIZE = 14.0          # square chassis, 14 in on a side
SENSOR_MAX_RANGE = 75.0    # rangefinders report nothing past this

# Fixed integration step. Real elapsed time is accumulated and consumed in whole
# steps of this size so the physics stay deterministic regardless of how often
# the browser polls.
FIXED_DT = 0.02
MAX_SUBSTEPS = 5

# Distance sensors sit flush on the robot's faces. This must not exceed 0.5, or
# a sensor would protrude past the wall the robot's own body is clamped against.
SENSOR_MOUNT_FRACTION = 0.5


class Simulation:
    def __init__(self):
        self.lock = threading.RLock()
        self.dt = FIXED_DT
        self.speed = 1.0
        self.arena_size = ARENA_HALF
        self.max_sensor_range = SENSOR_MAX_RANGE
        self.running = True
        self.debug = False
        self.ukf_enabled = True

        self.noise_config = {
            'odometry': 0.4,                      # inches per step
            'odometry_angular': 0.002,            # radians per step
            'imu': 0.5 * math.pi / 180.0,         # radians
            'distance': 0.8,                      # inches
        }

        self.ukf_config = {
            'alpha': 1.0,
            'beta': 2.0,
            'kappa': 0.0,
            'initial_covariance': 4.0,      # square inches
            'initial_heading_deg': 15.0,    # standard deviation, degrees
        }

        self.sensor_enabled = {
            'odom': True,
            'imu': True,
            'front': True,
            'back': True,
            'left': True,
            'right': True,
        }

        self.odom_sensor = OdometrySensor(
            noise_std=self.noise_config['odometry'],
            angular_noise_std=self.noise_config['odometry_angular'],
        )
        self.imu_sensor = IMUSensor(noise_std=self.noise_config['imu'])
        self.distance_sensors = {
            key: DistanceSensor(key, noise_std=self.noise_config['distance'])
            for key in DISTANCE_SENSOR_KEYS
        }

        self.reset()

    # ------------------------------------------------------------------ setup

    @property
    def mount_offset(self):
        return self.robot.width * SENSOR_MOUNT_FRACTION

    def build_ukf(self, x=0.0, y=0.0, theta=0.0):
        ukf = UKF(
            covariance=self.ukf_config['initial_covariance'],
            heading_covariance=math.radians(self.ukf_config['initial_heading_deg']) ** 2,
            alpha=self.ukf_config['alpha'],
            beta=self.ukf_config['beta'],
            kappa=self.ukf_config['kappa'],
            mount_offset=self.mount_offset,
            position_bound=self.arena_size - self.robot.width / 2.0,
        )
        ukf.set_state(x, y, theta)
        return ukf

    def reset(self):
        with self.lock:
            self.robot = Robot(x=0.0, y=0.0, theta=0.0, width=ROBOT_SIZE, arena_size=self.arena_size)
            self.ukf = self.build_ukf()
            self.key_states = {key: False for key in ('W', 'S', 'ArrowLeft', 'ArrowRight')}
            self.history = {
                'position_error': [],
                'heading_error': [],
                'x_track': [],
                'y_track': [],
                'est_x_track': [],
                'est_y_track': [],
            }
            self.accumulator = 0.0
            self.last_tick = time.monotonic()
            pose = self.robot.get_state()
            self.sensor_cache = self.generate_sensor_data(pose, pose, 0.0)
            self.update_ukf_detail_text()

    def reset_ukf(self):
        with self.lock:
            self.ukf = self.build_ukf(self.robot.x, self.robot.y, self.robot.theta)
            self.update_ukf_detail_text()

    # --------------------------------------------------------------- controls

    def set_debug(self, enabled):
        with self.lock:
            self.debug = bool(enabled)

    def toggle_ukf(self):
        with self.lock:
            self.ukf_enabled = not self.ukf_enabled

    def set_key_state(self, key, pressed):
        with self.lock:
            if key in self.key_states:
                self.key_states[key] = bool(pressed)

    def pause(self):
        with self.lock:
            self.running = False

    def resume(self):
        with self.lock:
            self.running = True
            # Discard the time spent paused so the sim does not fast-forward.
            self.last_tick = time.monotonic()
            self.accumulator = 0.0

    def step(self):
        """Advance by however much real time has elapsed, in fixed substeps."""
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_tick
            self.last_tick = now
            if not self.running:
                return
            self.accumulator += max(0.0, elapsed) * self.speed
            substeps = 0
            while self.accumulator >= self.dt and substeps < MAX_SUBSTEPS:
                self.update_simulation(self.dt)
                self.accumulator -= self.dt
                substeps += 1
            if substeps >= MAX_SUBSTEPS:
                # Fell too far behind (tab was backgrounded); drop the backlog.
                self.accumulator = 0.0

    def single_step(self):
        """Advance exactly one fixed step, even while paused."""
        with self.lock:
            self.update_simulation(self.dt, force=True)

    # ------------------------------------------------------------ config APIs

    def apply_config(self, payload):
        with self.lock:
            if 'ukf' in payload:
                updated = dict(self.ukf_config)
                values = payload['ukf'] or {}
                if 'alpha' in values:
                    updated['alpha'] = min(2.0, max(0.01, float(values['alpha'])))
                if 'beta' in values:
                    updated['beta'] = max(0.0, float(values['beta']))
                if 'kappa' in values:
                    updated['kappa'] = max(-2.9, float(values['kappa']))
                if 'initial_covariance' in values:
                    updated['initial_covariance'] = max(1e-6, float(values['initial_covariance']))
                if 'initial_heading_deg' in values:
                    updated['initial_heading_deg'] = min(180.0, max(0.01, float(values['initial_heading_deg'])))
                # Only rebuild the filter when something actually changed, so
                # editing an unrelated field does not silently reset it.
                if updated != self.ukf_config:
                    self.ukf_config = updated
                    self.ukf = self.build_ukf(self.ukf.state[0], self.ukf.state[1], self.ukf.state[2])

            if 'sensor_noise' in payload:
                values = payload['sensor_noise'] or {}
                for key in ('odometry', 'odometry_angular', 'imu', 'distance'):
                    if key in values:
                        self.noise_config[key] = max(0.0, float(values[key]))
                self.sync_sensor_noise()

            if 'simulation' in payload and 'speed' in payload['simulation']:
                self.speed = min(4.0, max(0.05, float(payload['simulation']['speed'])))

            if 'sensors' in payload:
                for key, enabled in (payload['sensors'] or {}).items():
                    if key in self.sensor_enabled:
                        self.sensor_enabled[key] = bool(enabled)

    def sync_sensor_noise(self):
        self.odom_sensor.noise_std = self.noise_config['odometry']
        self.odom_sensor.angular_noise_std = self.noise_config['odometry_angular']
        self.imu_sensor.noise_std = self.noise_config['imu']
        for sensor in self.distance_sensors.values():
            sensor.noise_std = self.noise_config['distance']

    def apply_experiment(self, name):
        deg = math.pi / 180.0
        presets = {
            'perfect_sensors': {'odometry': 0.0, 'odometry_angular': 0.0, 'imu': 0.0, 'distance': 0.0},
            'noisy_odometry': {'odometry': 3.0, 'odometry_angular': 0.02, 'imu': 0.5 * deg, 'distance': 0.8},
            'noisy_imu': {'odometry': 0.4, 'odometry_angular': 0.002, 'imu': 15.0 * deg, 'distance': 0.8},
            'no_odometry': {'odometry': 0.0, 'odometry_angular': 0.0, 'imu': 0.5 * deg, 'distance': 0.8},
            'odometry_only': {'odometry': 1.2, 'odometry_angular': 0.006, 'imu': 0.0, 'distance': 0.0},
            'all_sensors': {'odometry': 0.4, 'odometry_angular': 0.002, 'imu': 0.5 * deg, 'distance': 0.8},
            'high_noise': {'odometry': 5.0, 'odometry_angular': 0.03, 'imu': 5.0 * deg, 'distance': 3.0},
        }

        with self.lock:
            config = presets.get(name, presets['all_sensors'])
            self.noise_config.update(config)
            self.sync_sensor_noise()

            # 'no_odometry' keeps the IMU on deliberately: four orthogonal rays
            # in a square arena are invariant under 90-degree rotations, so with
            # no heading reference at all the pose is genuinely unobservable and
            # the filter can settle into a rotated alias of the true pose.
            self.sensor_enabled = {
                'odom': name != 'no_odometry',
                'imu': name != 'odometry_only',
                'front': name != 'odometry_only',
                'back': name != 'odometry_only',
                'left': name != 'odometry_only',
                'right': name != 'odometry_only',
            }

    def get_config(self):
        with self.lock:
            return {
                'ukf': dict(self.ukf_config),
                'sensor_noise': dict(self.noise_config),
                'sensors': dict(self.sensor_enabled),
                'simulation': {'speed': self.speed, 'debug': self.debug},
            }

    # ------------------------------------------------------------------ model

    def build_process_noise(self, dt):
        """Prediction uncertainty added each step, in world coordinates.

        The body-frame noise is anisotropic, so it has to be rotated into world
        coordinates rather than applied as a circle. The lateral axis is the
        noisier one: recovering chassis motion from the lateral pod means
        subtracting `lateral_offset * omega`, which folds the heading sensor's
        noise into the lateral estimate on top of that pod's own.
        """
        if self.sensor_enabled.get('odom', False):
            offset = self.odom_sensor.lateral_offset
            forward_var = self.noise_config['odometry'] ** 2
            lateral_var = (
                self.noise_config['odometry'] ** 2
                + (offset * self.noise_config['odometry_angular']) ** 2
            )
            angular_var = self.noise_config['odometry_angular'] ** 2
        else:
            # No odometry: the filter has no motion information at all, so the
            # noise has to cover every displacement one step could produce.
            forward_var = lateral_var = (self.robot.max_linear_velocity * dt) ** 2
            angular_var = (self.robot.max_angular_velocity * dt) ** 2

        body = np.diag([max(forward_var, 1e-9), max(lateral_var, 1e-9)])
        R = rotation_matrix(self.ukf.state[2])
        noise = np.zeros((3, 3), dtype=float)
        noise[:2, :2] = R @ body @ R.T
        noise[2, 2] = max(angular_var, 1e-9)
        return noise

    def build_measurement_noise(self, layout):
        variances = []
        for channel in layout:
            std = self.noise_config['imu'] if channel == 'imu' else self.noise_config['distance']
            # Floor the variance: a zero-noise sensor makes the innovation
            # covariance singular.
            floor = 1e-6 if channel == 'imu' else 1e-4
            variances.append(max(std ** 2, floor))
        return np.diag(variances)

    def build_control(self, sensor_data, dt):
        """Body-frame velocity handed to the prediction step, from odometry.

        The lateral pod's reading is not the chassis's lateral motion: the pod
        sits ahead of the centre of rotation, so turning sweeps it sideways.
        Subtracting `lateral_offset * omega` removes that swing and leaves the
        actual sideways displacement, which for an unslipping differential drive
        is close to zero -- but the pod is what establishes that, rather than it
        being assumed.
        """
        if dt <= 0.0 or not self.sensor_enabled.get('odom', False):
            return np.zeros(3, dtype=float)

        omega = sensor_data['odom_theta_noisy'] / dt
        lateral_rate = sensor_data['odom_lateral_noisy'] / dt
        return np.array([
            sensor_data['odom_forward_noisy'] / dt,
            lateral_rate - self.odom_sensor.lateral_offset * omega,
            omega,
        ], dtype=float)

    def available_channels(self, sensor_data):
        """Distance channels that produced a usable reading this step."""
        return {key: bool(sensor_data[f'{key}_in_range']) for key in DISTANCE_SENSOR_KEYS}

    def update_simulation(self, dt, force=False):
        with self.lock:
            if not self.running and not force:
                return

            self.robot.set_control(
                forward=1.0 if self.key_states['W'] else 0.0,
                backward=1.0 if self.key_states['S'] else 0.0,
                rotate_left=1.0 if self.key_states['ArrowLeft'] else 0.0,
                rotate_right=1.0 if self.key_states['ArrowRight'] else 0.0,
            )

            previous_pose = self.robot.get_state()
            self.robot.update(dt)
            current_pose = self.robot.get_state()

            sensor_data = self.generate_sensor_data(previous_pose, current_pose, dt)
            self.sensor_cache = sensor_data

            if self.ukf_enabled:
                self.ukf.predict(self.build_control(sensor_data, dt), dt, self.build_process_noise(dt))

                available = self.available_channels(sensor_data)
                layout = self.ukf.measurement_layout(self.sensor_enabled, available)
                if layout:
                    measurement = np.array(
                        [sensor_data['imu_noisy'] if c == 'imu' else sensor_data[f'{c}_noisy'] for c in layout],
                        dtype=float,
                    )
                    self.ukf.update(
                        measurement,
                        self.sensor_enabled,
                        self.arena_size,
                        self.build_measurement_noise(layout),
                        available,
                    )
            else:
                self.ukf.set_state(self.robot.x, self.robot.y, self.robot.theta)

            self.record_history()
            self.update_ukf_detail_text()

    def generate_sensor_data(self, previous_pose, current_pose, dt):
        # The pods resolve displacement in the frame the robot started the step
        # in, which is the same frame the filter's motion model rotates out of.
        heading = previous_pose['theta']
        dx = current_pose['x'] - previous_pose['x']
        dy = current_pose['y'] - previous_pose['y']
        forward_delta = dx * math.cos(heading) + dy * math.sin(heading)
        lateral_delta = -dx * math.sin(heading) + dy * math.cos(heading)
        delta_theta = normalize_angle(current_pose['theta'] - previous_pose['theta'])

        odom = self.odom_sensor.sample(forward_delta, lateral_delta, delta_theta)
        imu = self.imu_sensor.sample(current_pose['theta'])

        values = {
            'odom_forward_true': odom['true_forward'],
            'odom_forward_noisy': odom['noisy_forward'],
            'odom_lateral_true': odom['true_lateral'],
            'odom_lateral_noisy': odom['noisy_lateral'],
            'odom_theta_true': odom['true_theta'],
            'odom_theta_noisy': odom['noisy_theta'],
            'imu_true': imu['true'],
            'imu_noisy': imu['noisy'],
        }

        rays = []
        for key in DISTANCE_SENSOR_KEYS:
            reading = self.distance_sensors[key].sample(
                current_pose, self.arena_size, self.mount_offset, self.max_sensor_range
            )
            origin = reading['origin']
            end = origin + reading['direction'] * reading['true']
            rays.append({
                'key': key,
                'start_x': float(origin[0]), 'start_y': float(origin[1]),
                'end_x': float(end[0]), 'end_y': float(end[1]),
                'in_range': reading['in_range'],
                'enabled': self.sensor_enabled.get(key, False),
            })
            values[f'{key}_true'] = reading['true']
            values[f'{key}_noisy'] = reading['noisy']
            values[f'{key}_in_range'] = reading['in_range']

        values['sensor_rays'] = rays
        return values

    # ------------------------------------------------------------- reporting

    def record_history(self):
        true_pose = self.robot.get_state()
        est = self.ukf.state

        self.history['position_error'].append(math.hypot(true_pose['x'] - est[0], true_pose['y'] - est[1]))
        self.history['heading_error'].append(normalize_angle(true_pose['theta'] - est[2]))
        self.history['x_track'].append(true_pose['x'])
        self.history['y_track'].append(true_pose['y'])
        self.history['est_x_track'].append(float(est[0]))
        self.history['est_y_track'].append(float(est[1]))

        max_len = 300
        for key, series in self.history.items():
            if len(series) > max_len:
                self.history[key] = series[-max_len:]

    def update_ukf_detail_text(self):
        est = self.ukf.state
        available = self.available_channels(self.sensor_cache)
        layout = self.ukf.measurement_layout(self.sensor_enabled, available)
        dropped = [
            key for key in DISTANCE_SENSOR_KEYS
            if self.sensor_enabled.get(key, False) and not available[key]
        ]
        with np.printoptions(precision=4, suppress=True):
            self.ukf.last_details = (
                f"State [x, y, theta] = [{est[0]:.3f} in, {est[1]:.3f} in, "
                f"{math.degrees(normalize_angle(est[2])):.2f} deg]\n"
                f"Covariance P (in^2, rad^2) =\n{self.ukf.P}\n"
                f"Sigma-point scaling: alpha={self.ukf.alpha:g} beta={self.ukf.beta:g} "
                f"kappa={self.ukf.kappa:g}\n"
                f"Measurements used: {', '.join(layout) or 'none'}\n"
                f"Out of range: {', '.join(dropped) or 'none'}"
            )

    def get_state(self):
        with self.lock:
            true_pose = self.robot.get_state()
            est = self.ukf.state
            ellipse = compute_covariance_ellipse(self.ukf.P[:2, :2])
            sensor_data = self.sensor_cache

            sigma_points = []
            if self.debug:
                points, _, _ = self.ukf.sigma_points(self.ukf.state, self.ukf.P)
                sigma_points = [{'x': float(p[0]), 'y': float(p[1])} for p in points]

            sensor_keys = ['odom_forward_true', 'odom_forward_noisy',
                           'odom_lateral_true', 'odom_lateral_noisy',
                           'odom_theta_true', 'odom_theta_noisy',
                           'imu_true', 'imu_noisy']
            for key in DISTANCE_SENSOR_KEYS:
                sensor_keys += [f'{key}_true', f'{key}_noisy', f'{key}_in_range']

            return {
                'true_pose': {'x': true_pose['x'], 'y': true_pose['y'], 'theta': true_pose['theta']},
                'ukf_estimate': {'x': float(est[0]), 'y': float(est[1]), 'theta': float(est[2])},
                'position_error': math.hypot(true_pose['x'] - est[0], true_pose['y'] - est[1]),
                'heading_error': normalize_angle(true_pose['theta'] - est[2]),
                'sigma_x': math.sqrt(max(float(self.ukf.P[0, 0]), 0.0)),
                'sigma_y': math.sqrt(max(float(self.ukf.P[1, 1]), 0.0)),
                'sigma_theta': math.sqrt(max(float(self.ukf.P[2, 2]), 0.0)),
                'covariance_ellipse': {
                    'cx': float(est[0]),
                    'cy': float(est[1]),
                    'rx': float(ellipse['rx']),
                    'ry': float(ellipse['ry']),
                    'angle': float(ellipse['angle']),
                },
                'robot': {'width': self.robot.width, 'track_width': self.robot.track_width},
                'wheels': {
                    'left': self.robot.v_left,
                    'right': self.robot.v_right,
                    'max': self.robot.max_wheel_speed,
                    'linear': self.robot.linear_velocity,
                    'angular': self.robot.angular_velocity,
                },
                'world': {'size': self.arena_size, 'sensor_max_range': self.max_sensor_range,
                          'out_of_range': OUT_OF_RANGE},
                'sensors': {key: sensor_data.get(key, 0.0) for key in sensor_keys},
                'sensor_enabled': dict(self.sensor_enabled),
                'sensor_rays': sensor_data.get('sensor_rays', []),
                'graphs': {
                    'position_error': list(self.history['position_error']),
                    'heading_error': [math.degrees(v) for v in self.history['heading_error']],
                    'x_track': list(self.history['x_track']),
                    'y_track': list(self.history['y_track']),
                    'est_x_track': list(self.history['est_x_track']),
                    'est_y_track': list(self.history['est_y_track']),
                },
                'debug': self.debug,
                'running': self.running,
                'ukf_details': self.ukf.last_details,
                'sigma_points': sigma_points,
                'ukf_enabled': self.ukf_enabled,
            }
