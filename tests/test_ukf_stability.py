import math
import unittest

import numpy as np

from simulation.physics import ray_box_intersection
from simulation.sensors import OUT_OF_RANGE
from simulation.simulation import Simulation
from simulation.ukf import UKF


def drive(sim, keys, steps, dt=0.02):
    for key in keys:
        sim.set_key_state(key, True)
    errors = []
    for _ in range(steps):
        sim.update_simulation(dt)
        errors.append(sim.history['position_error'][-1])
    return errors


class TestRayCasting(unittest.TestCase):
    def test_distance_is_measured_to_the_exit_wall(self):
        # A sensor inside the arena sees the wall it is pointing at, not the
        # slab entry point behind it.
        self.assertAlmostEqual(ray_box_intersection([0.0, 0.0], [1.0, 0.0], 5.0), 5.0)
        self.assertAlmostEqual(ray_box_intersection([3.0, 0.0], [1.0, 0.0], 5.0), 2.0)
        self.assertAlmostEqual(ray_box_intersection([3.0, 0.0], [-1.0, 0.0], 5.0), 8.0)
        self.assertAlmostEqual(ray_box_intersection([0.0, 0.0], [0.0, -1.0], 5.0), 5.0)

    def test_diagonal_ray_hits_the_corner(self):
        root = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(ray_box_intersection([0.0, 0.0], [root, root], 5.0), 5.0 * math.sqrt(2.0))

    def test_sensor_flush_against_a_wall_reads_zero(self):
        self.assertAlmostEqual(ray_box_intersection([0.0, 5.0], [0.0, 1.0], 5.0), 0.0)

    def test_origin_outside_the_arena_is_projected_back_in(self):
        # Never returns a non-finite value, whatever the caller hands it.
        for origin in ([9.0, 0.0], [-40.0, 12.0], [5.0000001, 5.0000001]):
            value = ray_box_intersection(origin, [1.0, 0.0], 5.0)
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)

    def test_zero_direction_returns_max_range(self):
        self.assertEqual(ray_box_intersection([0.0, 0.0], [0.0, 0.0], 5.0, max_range=13.0), 13.0)


class TestUKFNumerics(unittest.TestCase):
    def test_sigma_points_handles_indefinite_covariance(self):
        ukf = UKF(covariance=1.0)
        ukf.P = np.diag([1.0, -100.0, 1.0]).astype(float)

        points, weights_mean, weights_cov = ukf.sigma_points(ukf.state, ukf.P)

        self.assertEqual(points.shape, (7, 3))
        self.assertEqual(weights_mean.shape, (7,))
        self.assertEqual(weights_cov.shape, (7,))
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertTrue(np.all(np.isfinite(weights_mean)))
        self.assertTrue(np.all(np.isfinite(weights_cov)))
        self.assertTrue(np.all(np.linalg.eigvalsh(ukf._ensure_positive_definite(ukf.P)) > 0.0))

    def test_sigma_point_weights_sum_to_one(self):
        for alpha, kappa in ((1.0, 0.0), (0.5, 0.0), (0.9, 1.0), (0.01, -2.0)):
            ukf = UKF(alpha=alpha, kappa=kappa)
            _, wm, _ = ukf.sigma_points(np.zeros(3), np.eye(3))
            self.assertAlmostEqual(float(np.sum(wm)), 1.0, places=9, msg=f"alpha={alpha} kappa={kappa}")

    def test_degenerate_scaling_does_not_divide_by_zero(self):
        # alpha^2 * (n + kappa) - n == -n makes (n + lambda) vanish unless floored.
        ukf = UKF(alpha=0.01, kappa=-2.99)
        points, wm, wc = ukf.sigma_points(np.zeros(3), np.eye(3))
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertTrue(np.all(np.isfinite(wm)))
        self.assertTrue(np.all(np.isfinite(wc)))

    def test_circular_mean_wraps_across_pi(self):
        ukf = UKF()
        angles = np.array([math.pi - 0.1, -math.pi + 0.1])
        weights = np.array([0.5, 0.5])
        mean = ukf.circular_mean(angles, weights)
        self.assertAlmostEqual(abs(mean), math.pi, places=6)

    def test_covariance_stays_positive_definite_over_a_long_run(self):
        sim = Simulation()
        drive(sim, ['W', 'ArrowLeft'], 400)
        eigenvalues = np.linalg.eigvalsh(sim.ukf.P)
        self.assertTrue(np.all(eigenvalues > 0.0), f"P is not positive definite: {eigenvalues}")
        self.assertTrue(np.all(np.isfinite(sim.ukf.state)))


class TestMeasurementWiring(unittest.TestCase):
    def test_layout_and_model_agree_for_every_sensor_subset(self):
        """The vector the simulation builds and the model's prediction must have
        the same length for all 64 on/off combinations, not just the default."""
        sim = Simulation()
        keys = ['odom', 'imu', 'front', 'back', 'left', 'right']
        for mask in range(1 << len(keys)):
            enabled = {key: bool(mask & (1 << i)) for i, key in enumerate(keys)}
            layout = sim.ukf.measurement_layout(enabled)
            predicted = sim.ukf.measurement_model(sim.ukf.state, enabled, sim.arena_size)
            self.assertEqual(predicted.shape, (len(layout),), f"mismatch for {enabled}")

    def test_every_sensor_subset_runs_without_error(self):
        keys = ['odom', 'imu', 'front', 'back', 'left', 'right']
        for mask in range(1 << len(keys)):
            sim = Simulation()
            sim.sensor_enabled = {key: bool(mask & (1 << i)) for i, key in enumerate(keys)}
            drive(sim, ['W', 'ArrowLeft'], 20)
            self.assertTrue(np.all(np.isfinite(sim.ukf.state)), f"diverged for {sim.sensor_enabled}")

    def test_every_experiment_preset_runs_without_error(self):
        presets = [
            'perfect_sensors', 'noisy_odometry', 'noisy_imu', 'no_odometry',
            'odometry_only', 'all_sensors', 'high_noise', 'unknown_preset_falls_back',
        ]
        for name in presets:
            sim = Simulation()
            sim.apply_experiment(name)
            drive(sim, ['W', 'ArrowLeft'], 50)
            self.assertTrue(np.all(np.isfinite(sim.ukf.state)), f"diverged for preset {name}")

    def test_wrong_sized_measurement_is_rejected(self):
        sim = Simulation()
        with self.assertRaises(ValueError):
            sim.ukf.update(np.zeros(2), sim.sensor_enabled, sim.arena_size, np.eye(5))

    def test_out_of_range_channels_are_dropped_from_the_layout(self):
        sim = Simulation()
        enabled = {key: True for key in sim.sensor_enabled}
        available = {'front': True, 'back': False, 'left': True, 'right': False}
        layout = sim.ukf.measurement_layout(enabled, available)
        self.assertEqual(layout, ['imu', 'front', 'left'])
        predicted = sim.ukf.measurement_model(sim.ukf.state, enabled, sim.arena_size, available)
        self.assertEqual(predicted.shape, (3,))

    def test_layout_and_model_agree_for_every_availability_pattern(self):
        sim = Simulation()
        keys = ['front', 'back', 'left', 'right']
        enabled = {key: True for key in sim.sensor_enabled}
        for mask in range(1 << len(keys)):
            available = {key: bool(mask & (1 << i)) for i, key in enumerate(keys)}
            layout = sim.ukf.measurement_layout(enabled, available)
            predicted = sim.ukf.measurement_model(sim.ukf.state, enabled, sim.arena_size, available)
            self.assertEqual(predicted.shape, (len(layout),), f"mismatch for {available}")

    def test_reading_past_max_range_is_reported_as_out_of_range(self):
        sim = Simulation()
        sim.noise_config['distance'] = 0.0
        sim.sync_sensor_noise()
        # Backed against the +X wall, the rear sensor faces the far wall, which
        # is well past its usable range.
        sim.robot.x = sim.arena_size - sim.robot.width / 2.0
        data = sim.generate_sensor_data(sim.robot.get_state(), sim.robot.get_state(), 0.0)

        self.assertGreater(data['back_true'], sim.max_sensor_range)
        self.assertFalse(data['back_in_range'])
        self.assertEqual(data['back_noisy'], OUT_OF_RANGE)

        self.assertLessEqual(data['front_true'], sim.max_sensor_range)
        self.assertTrue(data['front_in_range'])
        self.assertNotEqual(data['front_noisy'], OUT_OF_RANGE)

    def test_the_sentinel_never_reaches_the_filter(self):
        """An out-of-range reading must shrink the measurement, not enter it as
        a -1 that the filter would treat as a real distance."""
        sim = Simulation()
        sim.robot.x = sim.arena_size - sim.robot.width / 2.0
        drive(sim, ['W'], 5)

        available = sim.available_channels(sim.sensor_cache)
        layout = sim.ukf.measurement_layout(sim.sensor_enabled, available)
        self.assertNotIn('back', layout)
        measurement = [sim.sensor_cache[f'{c}_noisy'] for c in layout if c != 'imu']
        self.assertTrue(all(value >= 0.0 for value in measurement), measurement)

    def test_opposing_sensors_cannot_both_be_out_of_range(self):
        """Sanity check on the geometry: the field is narrower than twice the
        sensor range, so the filter always keeps at least one ray per axis."""
        sim = Simulation()
        span = 2 * sim.arena_size - sim.robot.width
        self.assertLess(span, 2 * sim.max_sensor_range)

    def test_out_of_range_rays_are_not_drawn(self):
        sim = Simulation()
        sim.robot.x = sim.arena_size - sim.robot.width / 2.0
        data = sim.generate_sensor_data(sim.robot.get_state(), sim.robot.get_state(), 0.0)
        rays = {ray['key']: ray for ray in data['sensor_rays']}
        self.assertFalse(rays['back']['in_range'])
        self.assertTrue(rays['front']['in_range'])

    def test_no_enabled_sensors_leaves_the_state_untouched(self):
        sim = Simulation()
        disabled = {key: False for key in sim.sensor_enabled}
        before = sim.ukf.state.copy()
        sim.ukf.update(np.zeros(0), disabled, sim.arena_size, np.zeros((0, 0)))
        np.testing.assert_allclose(sim.ukf.state, before)


class TestOdometryPods(unittest.TestCase):
    """Two-axis dead-wheel odometry: an X pod and a Y pod, plus a heading source."""

    @staticmethod
    def noiseless(sim):
        sim.noise_config.update({'odometry': 0.0, 'odometry_angular': 0.0})
        sim.sync_sensor_noise()
        return sim

    def test_y_pod_measures_lateral_motion_the_x_pod_cannot_see(self):
        sim = self.noiseless(Simulation())
        before = sim.robot.get_state()
        sim.robot.y += 3.0  # sideways shove at theta = 0
        data = sim.generate_sensor_data(before, sim.robot.get_state(), 0.02)
        self.assertAlmostEqual(data['odom_forward_true'], 0.0, places=9)
        self.assertAlmostEqual(data['odom_lateral_true'], 3.0, places=9)

    def test_pods_resolve_into_the_body_frame(self):
        sim = self.noiseless(Simulation())
        sim.robot.theta = math.pi / 2.0
        before = sim.robot.get_state()
        sim.robot.y += 3.0  # world +Y is *forward* when facing +Y
        data = sim.generate_sensor_data(before, sim.robot.get_state(), 0.02)
        self.assertAlmostEqual(data['odom_forward_true'], 3.0, places=9)
        self.assertAlmostEqual(data['odom_lateral_true'], 0.0, places=9)

    def test_turning_sweeps_the_offset_y_pod_sideways(self):
        """The pod sits ahead of the centre of rotation, so a pure spin reads
        offset * delta_theta even though the chassis never translates."""
        sim = self.noiseless(Simulation())
        offset = sim.odom_sensor.lateral_offset
        self.assertGreater(offset, 0.0, "test is vacuous with a centred pod")

        drive(sim, ['ArrowLeft'], 20)
        data = sim.sensor_cache
        self.assertAlmostEqual(sim.robot.x, 0.0, places=9)
        self.assertAlmostEqual(sim.robot.y, 0.0, places=9)
        self.assertAlmostEqual(data['odom_lateral_true'], offset * data['odom_theta_true'], places=9)
        self.assertGreater(abs(data['odom_lateral_true']), 1e-3)

    def test_control_removes_the_pod_swing(self):
        """Left in, the swing term would make the filter believe the robot
        strafes every time it turns."""
        sim = self.noiseless(Simulation())
        drive(sim, ['ArrowLeft'], 20)
        control = sim.build_control(sim.sensor_cache, 0.02)
        self.assertAlmostEqual(control[0], 0.0, places=9)   # no forward motion
        self.assertAlmostEqual(control[1], 0.0, places=9)   # and no lateral motion
        self.assertGreater(control[2], 0.0)                 # only yaw

    def test_uncorrected_swing_would_bias_the_estimate(self):
        """Guards the correction itself: without it the filter drifts sideways
        while spinning in place."""
        sim = self.noiseless(Simulation())
        drive(sim, ['ArrowLeft'], 20)
        raw = sim.sensor_cache['odom_lateral_noisy'] / 0.02
        corrected = sim.build_control(sim.sensor_cache, 0.02)[1]
        self.assertGreater(abs(raw), 1.0)
        self.assertLess(abs(corrected), 1e-6)

    @staticmethod
    def dead_reckon(initial_heading_deg, steps=60):
        """Perfect pods, no corrections: prediction alone, from a given amount
        of initial heading uncertainty."""
        sim = Simulation()
        sim.ukf_config['initial_heading_deg'] = initial_heading_deg
        sim.ukf = sim.build_ukf()
        TestOdometryPods.noiseless(sim)
        sim.sensor_enabled = {key: False for key in sim.sensor_enabled}
        sim.sensor_enabled['odom'] = True
        drive(sim, ['W'], steps)
        return sim

    def test_pods_drive_the_prediction_step(self):
        # A thousandth of an inch: not exact, because even 0.1 degrees of heading
        # uncertainty shortens the prediction slightly (see the test below).
        sim = self.dead_reckon(0.1)
        self.assertAlmostEqual(sim.ukf.state[0], sim.robot.x, delta=1e-3)
        self.assertAlmostEqual(sim.ukf.state[1], sim.robot.y, delta=1e-3)

    def test_heading_uncertainty_shortens_the_predicted_travel(self):
        """Not a defect: the sigma points span a range of headings and each
        carries the motion off in a different direction, so their weighted mean
        travels less far than any single one did. This is exactly why the initial
        heading variance is configured separately from the position variance,
        in its own units."""
        confident = self.dead_reckon(1.0)
        uncertain = self.dead_reckon(60.0)
        self.assertLess(uncertain.ukf.state[0], confident.ukf.state[0])
        self.assertAlmostEqual(confident.ukf.state[0], confident.robot.x, delta=0.05)
        self.assertLess(uncertain.ukf.state[0], 0.85 * uncertain.robot.x)

    def test_position_and_heading_covariance_are_configured_separately(self):
        sim = Simulation()
        self.assertAlmostEqual(sim.ukf.P[0, 0], sim.ukf_config['initial_covariance'])
        self.assertAlmostEqual(sim.ukf.P[1, 1], sim.ukf_config['initial_covariance'])
        self.assertAlmostEqual(
            sim.ukf.P[2, 2],
            math.radians(sim.ukf_config['initial_heading_deg']) ** 2,
        )
        # A scalar covering both would put radians on the same scale as inches.
        self.assertNotAlmostEqual(sim.ukf.P[2, 2], sim.ukf.P[0, 0])

    def test_initial_heading_is_clamped_to_a_usable_range(self):
        sim = Simulation()
        sim.apply_config({'ukf': {'initial_heading_deg': 0.0}})
        self.assertGreater(sim.ukf_config['initial_heading_deg'], 0.0)
        sim.apply_config({'ukf': {'initial_heading_deg': 1e6}})
        self.assertLessEqual(sim.ukf_config['initial_heading_deg'], 180.0)

    def test_lateral_process_noise_accounts_for_the_offset(self):
        sim = Simulation()
        noise = sim.build_process_noise(0.02)
        # At theta = 0 the body axes line up with world, so read the diagonal.
        self.assertAlmostEqual(sim.ukf.state[2], 0.0, places=9)
        offset = sim.odom_sensor.lateral_offset
        expected = (sim.noise_config['odometry'] ** 2
                    + (offset * sim.noise_config['odometry_angular']) ** 2)
        self.assertAlmostEqual(noise[1, 1], expected, places=12)
        self.assertGreater(noise[1, 1], noise[0, 0], "lateral must be the noisier axis")

    def test_process_noise_rotates_with_heading(self):
        sim = Simulation()
        sim.ukf.set_state(0.0, 0.0, math.pi / 2.0)
        noise = sim.build_process_noise(0.02)
        forward_var = sim.noise_config['odometry'] ** 2
        # Facing +Y, the forward axis is world Y, so the variances swap places.
        self.assertAlmostEqual(noise[1, 1], forward_var, places=12)
        self.assertGreater(noise[0, 0], noise[1, 1])


class TestFilterConvergence(unittest.TestCase):
    """The filter must actually track the robot. Nothing previously asserted
    this, which is how a measurement model that pinned the estimate to the
    origin went unnoticed."""

    def test_perfect_sensors_track_almost_exactly(self):
        sim = Simulation()
        sim.apply_experiment('perfect_sensors')
        errors = drive(sim, ['W'], 200)
        self.assertLess(max(errors[100:]), 0.05)

    def test_all_sensors_converge_while_driving_and_turning(self):
        for seed in range(3):
            np.random.seed(seed)
            sim = Simulation()
            sim.apply_experiment('all_sensors')
            errors = drive(sim, ['W', 'ArrowLeft'], 300)
            settled = errors[150:]
            self.assertLess(sum(settled) / len(settled), 3.0, f"seed {seed}")

    def test_filter_recovers_from_a_bad_initial_estimate(self):
        np.random.seed(7)
        sim = Simulation()
        sim.ukf.set_state(-50.0, -50.0, math.pi)  # far from the true pose at the origin
        errors = drive(sim, ['W'], 300)
        self.assertGreater(errors[0], 50.0)
        self.assertLess(errors[-1], 5.0, "filter never recovered from a bad initialisation")

    def test_high_noise_stays_bounded(self):
        np.random.seed(3)
        sim = Simulation()
        sim.apply_experiment('high_noise')
        errors = drive(sim, ['W'], 400)
        self.assertLess(max(errors[200:]), 20.0)

    def test_estimate_never_leaves_the_arena(self):
        np.random.seed(11)
        sim = Simulation()
        sim.apply_experiment('high_noise')
        bound = sim.arena_size - sim.robot.width / 2.0
        for _ in range(400):
            sim.update_simulation(0.02)
            sim.set_key_state('W', True)
            sim.set_key_state('ArrowRight', True)
            self.assertLessEqual(abs(sim.ukf.state[0]), bound + 1e-9)
            self.assertLessEqual(abs(sim.ukf.state[1]), bound + 1e-9)

    def test_odometry_only_drifts_because_it_has_no_absolute_reference(self):
        # Dead reckoning is *supposed* to accumulate error; a filter that held a
        # perfect estimate here would be reading something it should not see.
        np.random.seed(5)
        sim = Simulation()
        sim.apply_experiment('odometry_only')
        errors = drive(sim, ['W'], 400)
        self.assertGreater(errors[-1], errors[10])

    def test_filter_never_sees_ground_truth(self):
        """Teleporting the robot must not move the estimate on its own: the only
        paths into the filter are odometry and the enabled sensors."""
        sim = Simulation()
        sim.sensor_enabled = {key: False for key in sim.sensor_enabled}
        before = sim.ukf.state.copy()
        sim.robot.x, sim.robot.y, sim.robot.theta = 3.0, -2.0, 1.2
        sim.update_simulation(0.02)
        np.testing.assert_allclose(sim.ukf.state, before, atol=1e-12)


class TestDifferentialDrive(unittest.TestCase):
    def test_w_and_s_drive_both_sides_together(self):
        for key, expect_forward in (('W', True), ('S', False)):
            sim = Simulation()
            drive(sim, [key], 10)
            self.assertAlmostEqual(sim.robot.v_left, sim.robot.v_right, places=9,
                                   msg=f"{key} must drive both sides equally")
            self.assertAlmostEqual(sim.robot.angular_velocity, 0.0, places=9)
            if expect_forward:
                self.assertGreater(sim.robot.v_left, 0.0)
            else:
                self.assertLess(sim.robot.v_left, 0.0)

    def test_driving_is_body_relative_not_screen_relative(self):
        """Unlike the old omnidirectional robot, W drives along the heading."""
        sim = Simulation()
        drive(sim, ['W'], 20)
        self.assertGreater(sim.robot.x, 0.0)
        self.assertAlmostEqual(sim.robot.y, 0.0, places=9)

        sim = Simulation()
        sim.robot.theta = math.pi / 2.0
        drive(sim, ['W'], 20)
        self.assertGreater(sim.robot.y, 0.0)
        self.assertAlmostEqual(sim.robot.x, 0.0, places=9)

    def test_turning_spins_the_sides_in_opposite_directions(self):
        for key, expect_ccw in (('ArrowLeft', True), ('ArrowRight', False)):
            sim = Simulation()
            drive(sim, [key], 10)
            self.assertLess(sim.robot.v_left * sim.robot.v_right, 0.0,
                            f"{key} must drive the sides in opposite directions")
            self.assertAlmostEqual(sim.robot.v_left, -sim.robot.v_right, places=9)
            if expect_ccw:
                self.assertGreater(sim.robot.angular_velocity, 0.0)
            else:
                self.assertLess(sim.robot.angular_velocity, 0.0)

    def test_turning_in_place_does_not_translate(self):
        sim = Simulation()
        drive(sim, ['ArrowLeft'], 60)
        self.assertAlmostEqual(sim.robot.x, 0.0, places=9)
        self.assertAlmostEqual(sim.robot.y, 0.0, places=9)
        self.assertNotAlmostEqual(sim.robot.theta, 0.0, places=3)

    def test_there_is_no_strafe_control(self):
        sim = Simulation()
        for key in ('A', 'D'):
            sim.set_key_state(key, True)
        self.assertNotIn('A', sim.key_states)
        self.assertNotIn('D', sim.key_states)
        drive(sim, [], 20)
        self.assertAlmostEqual(sim.robot.x, 0.0, places=9)
        self.assertAlmostEqual(sim.robot.y, 0.0, places=9)

    def test_drive_and_turn_together_produces_an_arc(self):
        sim = Simulation()
        drive(sim, ['W', 'ArrowLeft'], 40)
        self.assertNotAlmostEqual(sim.robot.v_left, sim.robot.v_right, places=3)
        self.assertGreater(sim.robot.linear_velocity, 0.0)
        self.assertGreater(sim.robot.angular_velocity, 0.0)

    def test_saturated_mix_keeps_wheels_within_limits(self):
        sim = Simulation()
        drive(sim, ['W', 'ArrowLeft'], 200)
        self.assertLessEqual(abs(sim.robot.v_left), sim.robot.max_wheel_speed + 1e-9)
        self.assertLessEqual(abs(sim.robot.v_right), sim.robot.max_wheel_speed + 1e-9)

    def test_acceleration_is_per_second_not_per_step(self):
        """A step of dt must change wheel speed by acceleration * dt. The ramp
        used to apply the full rate every step, making it instantaneous."""
        sim = Simulation()
        sim.set_key_state('W', True)
        sim.update_simulation(0.02)
        self.assertAlmostEqual(sim.robot.v_left, sim.robot.acceleration * 0.02, places=9)
        self.assertLess(sim.robot.v_left, sim.robot.max_wheel_speed)

    def test_wheel_speed_saturates_at_the_maximum(self):
        sim = Simulation()
        drive(sim, ['W'], 200)
        self.assertAlmostEqual(sim.robot.v_left, sim.robot.max_wheel_speed, places=9)

    def test_robot_stays_inside_the_arena(self):
        sim = Simulation()
        drive(sim, ['W'], 500)
        half = sim.robot.width / 2.0
        self.assertLessEqual(abs(sim.robot.x), sim.arena_size - half + 1e-9)
        self.assertLessEqual(abs(sim.robot.y), sim.arena_size - half + 1e-9)

    def test_field_and_chassis_are_the_expected_size(self):
        sim = Simulation()
        self.assertEqual(2 * sim.arena_size, 144.0)  # 12 ft square, in inches
        self.assertEqual(sim.robot.width, 14.0)


class TestSimulationControls(unittest.TestCase):
    def test_pause_stops_the_robot(self):
        sim = Simulation()
        sim.set_key_state('W', True)
        sim.pause()
        before = sim.robot.get_state()
        sim.update_simulation(0.02)
        self.assertEqual(sim.robot.x, before['x'])

    def test_single_step_advances_while_paused(self):
        sim = Simulation()
        sim.set_key_state('W', True)
        sim.pause()
        before = sim.robot.get_state()
        sim.single_step()
        self.assertGreater(sim.robot.x, before['x'])

    def test_releasing_a_key_stops_the_robot(self):
        sim = Simulation()
        drive(sim, ['ArrowLeft'], 30)
        self.assertNotAlmostEqual(sim.robot.angular_velocity, 0.0, places=3)

        sim.set_key_state('ArrowLeft', False)
        for _ in range(60):
            sim.update_simulation(0.02)
        self.assertAlmostEqual(sim.robot.v_left, 0.0, places=9)
        self.assertAlmostEqual(sim.robot.v_right, 0.0, places=9)
        self.assertAlmostEqual(sim.robot.angular_velocity, 0.0, places=9)

    def test_a_latched_key_keeps_driving_until_released(self):
        """Documents why the client restates its whole key set on load: the
        server only hears transitions, so a key it was told was pressed stays
        pressed until something says otherwise."""
        sim = Simulation()
        sim.set_key_state('ArrowLeft', True)
        for _ in range(50):
            sim.update_simulation(0.02)
        self.assertGreater(abs(sim.robot.theta), 0.5)

        # Restating the full set, as the page does on load, stops it.
        for key in sim.key_states:
            sim.set_key_state(key, False)
        for _ in range(60):
            sim.update_simulation(0.02)
        self.assertAlmostEqual(sim.robot.angular_velocity, 0.0, places=9)

    def test_reset_clears_pose_history_and_filter(self):
        sim = Simulation()
        drive(sim, ['W'], 50)
        sim.reset()
        self.assertEqual((sim.robot.x, sim.robot.y, sim.robot.theta), (0.0, 0.0, 0.0))
        self.assertEqual(sim.history['position_error'], [])
        np.testing.assert_allclose(sim.ukf.state, np.zeros(3))
        self.assertFalse(any(sim.key_states.values()))

    def test_history_is_bounded(self):
        sim = Simulation()
        drive(sim, ['W'], 600)
        for key, series in sim.history.items():
            self.assertLessEqual(len(series), 300, key)

    def test_editing_sensor_noise_does_not_reset_the_filter(self):
        sim = Simulation()
        drive(sim, ['W'], 60)
        before = sim.ukf.state.copy()
        sim.apply_config({'sensor_noise': {'distance': 0.05}})
        np.testing.assert_allclose(sim.ukf.state, before)
        self.assertEqual(sim.noise_config['distance'], 0.05)

    def test_resubmitting_identical_ukf_config_does_not_reset_the_filter(self):
        sim = Simulation()
        drive(sim, ['W'], 60)
        before = sim.ukf.state.copy()
        sim.apply_config({'ukf': dict(sim.ukf_config)})
        np.testing.assert_allclose(sim.ukf.state, before)

    def test_changing_ukf_config_rebuilds_the_filter_in_place(self):
        sim = Simulation()
        drive(sim, ['W'], 60)
        before = sim.ukf.state.copy()
        sim.apply_config({'ukf': {'alpha': 0.7}})
        self.assertEqual(sim.ukf_config['alpha'], 0.7)
        # Rebuilt, but around the current estimate rather than back at the origin.
        np.testing.assert_allclose(sim.ukf.state, before, atol=1e-9)

    def test_config_values_are_clamped_to_a_usable_range(self):
        sim = Simulation()
        sim.apply_config({'ukf': {'alpha': 0.0, 'initial_covariance': -3.0},
                          'sensor_noise': {'distance': -1.0},
                          'simulation': {'speed': 500.0}})
        self.assertGreater(sim.ukf_config['alpha'], 0.0)
        self.assertGreater(sim.ukf_config['initial_covariance'], 0.0)
        self.assertGreaterEqual(sim.noise_config['distance'], 0.0)
        self.assertLessEqual(sim.speed, 4.0)

    def test_disabling_the_ukf_pins_the_estimate_to_truth(self):
        sim = Simulation()
        sim.toggle_ukf()
        drive(sim, ['W'], 30)
        self.assertAlmostEqual(sim.ukf.state[0], sim.robot.x)
        self.assertAlmostEqual(sim.ukf.state[1], sim.robot.y)

    def test_get_state_is_json_serialisable(self):
        import json

        sim = Simulation()
        sim.set_debug(True)
        drive(sim, ['W', 'ArrowLeft'], 20)
        payload = json.dumps(sim.get_state())
        self.assertNotIn('Infinity', payload)
        self.assertNotIn('NaN', payload)
        self.assertEqual(len(json.loads(payload)['sigma_points']), 7)

    def test_sigma_points_are_only_computed_in_debug_mode(self):
        sim = Simulation()
        self.assertEqual(sim.get_state()['sigma_points'], [])
        sim.set_debug(True)
        self.assertEqual(len(sim.get_state()['sigma_points']), 7)


if __name__ == '__main__':
    unittest.main()
