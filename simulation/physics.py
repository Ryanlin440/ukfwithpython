import math

import numpy as np

# Unit direction of each distance sensor in the robot body frame.
# Robot +X = forward, robot +Y = left.
SENSOR_DIRECTIONS = {
    'front': np.array([1.0, 0.0]),
    'back': np.array([-1.0, 0.0]),
    'left': np.array([0.0, 1.0]),
    'right': np.array([0.0, -1.0]),
}

DISTANCE_SENSOR_KEYS = ('front', 'back', 'left', 'right')


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    """Wrap an angle into [-pi, pi] in constant time."""
    return math.remainder(float(angle), 2.0 * math.pi)


def rotation_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def compute_covariance_ellipse(Pxy):
    cov = np.array([[Pxy[0, 0], Pxy[0, 1]], [Pxy[1, 0], Pxy[1, 1]]], dtype=float)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.atan2(vecs[1, 0], vecs[0, 0])
    scale = 2.4477  # 95% confidence for a 2-DOF Gaussian
    rx = math.sqrt(max(vals[0], 1e-12)) * scale
    ry = math.sqrt(max(vals[1], 1e-12)) * scale
    return {"angle": angle, "rx": rx, "ry": ry}


def ray_box_intersection(origin, direction, arena_size, max_range=None):
    """Distance from a sensor inside the arena, along `direction`, to the wall.

    The arena spans [-arena_size, arena_size] on both axes. This is the slab
    test, but note it returns the ray's *exit* parameter: the origin is a sensor
    mounted on a robot that lives inside the arena, so the wall it sees is where
    the ray leaves the box, not where it would enter.
    """
    if max_range is None:
        max_range = 4.0 * arena_size

    direction = np.asarray(direction, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if np.linalg.norm(direction) < 1e-9:
        return max_range

    # A sensor flush against a wall can land a hair outside it through rounding,
    # and an estimated pose can wander outside the arena entirely. Projecting the
    # origin back in keeps the cast well defined and the measurement function
    # continuous, which matters because the filter differentiates through it.
    origin = np.clip(origin, -arena_size, arena_size)

    t_exit = float('inf')
    for axis in range(2):
        if abs(direction[axis]) < 1e-12:
            continue  # parallel to this pair of walls; they cannot bound the ray
        inv = 1.0 / direction[axis]
        t_exit = min(t_exit, max((-arena_size - origin[axis]) * inv, (arena_size - origin[axis]) * inv))

    if not math.isfinite(t_exit):
        return max_range
    return clamp(t_exit, 0.0, max_range)
