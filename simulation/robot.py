import math

from simulation.physics import clamp, normalize_angle


class Robot:
    """Ground-truth differential-drive robot. All lengths are in inches.

    The two drive trains are independent: each wheel has its own commanded
    speed, and the chassis motion falls out of the pair. Forward speed is their
    average and yaw rate is their difference over the track width, so spinning
    the wheels in opposite directions turns the robot in place. There is no
    lateral degree of freedom -- this robot cannot strafe.
    """

    def __init__(
        self,
        x=0.0,
        y=0.0,
        theta=0.0,
        width=14.0,
        arena_size=72.0,
        max_wheel_speed=48.0,
        track_width=14.0,
        acceleration=150.0,
        deceleration=200.0,
        turn_command_scale=0.5,
    ):
        self.x = x
        self.y = y
        self.theta = theta
        self.width = width
        self.arena_size = arena_size
        self.max_wheel_speed = max_wheel_speed
        self.track_width = track_width
        self.acceleration = acceleration
        self.deceleration = deceleration
        # Turning at full wheel authority spins a 14 in robot faster than is
        # comfortable to steer, so the turn command only asks for part of it.
        self.turn_command_scale = turn_command_scale

        self.v_left = 0.0
        self.v_right = 0.0
        self.target_left = 0.0
        self.target_right = 0.0

    @property
    def linear_velocity(self):
        return 0.5 * (self.v_left + self.v_right)

    @property
    def angular_velocity(self):
        return (self.v_right - self.v_left) / self.track_width

    @property
    def max_linear_velocity(self):
        return self.max_wheel_speed

    @property
    def max_angular_velocity(self):
        return 2.0 * self.max_wheel_speed * self.turn_command_scale / self.track_width

    def set_control(self, forward, backward, rotate_left, rotate_right):
        drive = forward - backward
        turn = (rotate_left - rotate_right) * self.turn_command_scale

        # Arcade mixing: turning subtracts from one side and adds to the other.
        left = drive - turn
        right = drive + turn
        # Preserve the ratio between the sides when the mix saturates, so a
        # turn requested while driving is not silently flattened.
        scale = max(1.0, abs(left), abs(right))

        self.target_left = (left / scale) * self.max_wheel_speed
        self.target_right = (right / scale) * self.max_wheel_speed

    def _move_toward(self, current, target, dt):
        """Ramp a wheel speed toward its target at a rate in inches/second^2."""
        delta = target - current
        if abs(delta) < 1e-12:
            return target
        # Slowing down, or reversing through zero, uses the braking rate.
        speeding_up = abs(target) > abs(current) and target * current >= 0.0
        step = (self.acceleration if speeding_up else self.deceleration) * dt
        if delta > 0.0:
            return min(target, current + step)
        return max(target, current - step)

    def update(self, dt):
        self.v_left = self._move_toward(self.v_left, self.target_left, dt)
        self.v_right = self._move_toward(self.v_right, self.target_right, dt)

        v = self.linear_velocity
        omega = self.angular_velocity

        # Translate along the heading held at the start of the step, then rotate.
        # Odometry resolves its displacement in that same starting frame, so the
        # filter's motion model and the truth agree on which heading was used.
        theta_start = self.theta
        self.x += v * math.cos(theta_start) * dt
        self.y += v * math.sin(theta_start) * dt
        self.theta = normalize_angle(theta_start + omega * dt)

        half = self.width / 2.0
        self.x = clamp(self.x, -self.arena_size + half, self.arena_size - half)
        self.y = clamp(self.y, -self.arena_size + half, self.arena_size - half)

    def get_state(self):
        return {"x": self.x, "y": self.y, "theta": self.theta, "width": self.width}
