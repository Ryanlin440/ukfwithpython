# UKF Robot Localization Sandbox

A local Python/Flask website that drives a differential-drive robot around a
12 ft x 12 ft field and visualizes noisy odometry, IMU and rangefinder readings
alongside a live Unscented Kalman Filter estimate.

**Every length in this project is in inches.** The field is 144 x 144 in, the
robot is a 14 x 14 in square, and the distance sensors are good to 75 in.

## Features

- Differential drive with two independently commanded sides
- Noisy odometry, IMU heading and four rangefinders, each individually toggleable
- Rangefinders with a finite 75 in range that drop out of the filter when exceeded
- Real UKF implementation using sigma points and an unscented transform
- Ground-truth pose, estimated pose, and the trail of each drawn together
- Live sensor rays, so you can see what the filter is actually being told
- An "i" button on every tunable parameter explaining what changing it does
- Debug mode showing sigma points and the covariance ellipse
- Experiment presets for common sensor configurations

## Setup

1. Open a terminal in this project folder.
2. Create a virtual environment (optional but recommended):

       python3 -m venv .venv
       source .venv/bin/activate

3. Install dependencies:

       pip install -r requirements.txt

## Run the app

       python app.py

Then open http://localhost:5001. Set `HOST` or `PORT` to override the defaults.

The default is port 5001 rather than 5000 because macOS gives port 5000 to the
AirPlay Receiver, which answers every request with a 403.

Run the tests with:

       python -m unittest discover -s tests

## Controls

- `W` / `S`: drive both sides forward / backward
- `Left Arrow` / `Right Arrow`: spin the sides in opposite directions to turn

There is no strafe. A differential drive has no lateral degree of freedom, so
`W` drives along the robot's current heading rather than up the screen. Holding
a drive key and a turn key together mixes the two, which produces an arc; the
mix is rescaled if it would exceed a side's maximum speed, so the ratio between
the sides survives saturation.

## Coordinate convention

- World +X = right, world +Y = up
- Robot +X = forward, robot +Y = left
- `theta` is the heading, measured counterclockwise from world +X

## How the drivetrain is modelled

Each side ramps toward its commanded speed at a finite acceleration. The chassis
motion follows from the pair: forward speed is their average, and yaw rate is
their difference divided by the track width. Equal speeds drive straight,
opposite speeds spin in place, and anything between traces an arc.

## How the filter is wired

The state is the pose `[x, y, theta]`. The UKF never receives ground truth; the
only two paths into it are:

1. **Prediction** consumes *odometry* — the noisy distance the wheels report
   rolling and the heading change they report turning. Odometry is a motion
   input, not a position measurement, so it is integrated by the prediction step
   rather than compared against the state. Its lateral term is exactly zero:
   encoders cannot observe sideways motion, and this chassis cannot produce any.
2. **Correction** consumes whichever of the *IMU heading* and *rangefinders* are
   both enabled and in range. The measurement vector's contents and ordering come
   from `UKF.measurement_layout`, and both the simulation and the measurement
   model derive their layout from that single function so the two cannot disagree.

Process noise is anisotropic in the body frame — the robot is uncertain about how
far it rolled but barely moves sideways — and is rotated into world coordinates
rather than applied as a circle. With odometry off it widens to cover every
displacement one step could produce.

The estimate is constrained to the field, which is map knowledge rather than
ground truth, and matters because rangefinder readings saturate outside the walls
and would leave the filter with no gradient to find its way back.

## Sensor range

A rangefinder reports a distance only when it measures 75 in or less. Past that
it reports `-1`, its ray is not drawn, and **the channel is dropped from the
measurement vector for that step** rather than being fed to the filter as a
literal -1. The measurement therefore changes length as the robot moves, which is
why the layout is computed fresh each step.

The predicted measurement inside the filter is deliberately *not* capped. A
channel only appears when the real sensor got a reading, and clamping the
prediction would flatten the gradient exactly where a sigma point disagrees with
that reading.

The field is 144 in across and the robot is 14 in wide, so opposing sensors span
130 in — less than two sensor ranges. At least one ray per axis is therefore
always in range, whatever the pose.

## Experiment presets

| Preset | What it shows |
| --- | --- |
| Perfect Sensors | Noise-free baseline; the estimate should sit exactly on the robot |
| Noisy Odometry | Bad dead reckoning corrected by good absolute sensors |
| Noisy IMU | Heading noise absorbed by the rangefinders |
| No Odometry | Localization from absolute sensors alone, with no motion model |
| Odometry Only | Pure dead reckoning; error grows without bound, as it should |
| All Sensors | The default, mildly noisy configuration |
| High Noise | Everything noisy at once; the estimate stays bounded but wanders |

Two notes on what you will see:

- **Odometry Only is supposed to drift.** With no absolute reference, position
  error follows a random walk and heading error rotates the whole accumulated
  path. A filter that tracked perfectly here would be reading something it should
  not have access to.
- **Heading needs an absolute reference.** Four orthogonal rays in a square field
  are invariant under 90-degree rotations, so with neither IMU nor odometry the
  pose is genuinely unobservable and the filter can settle into a rotated alias of
  the true pose. This is why the "No Odometry" preset keeps the IMU on. You can
  still turn everything off by hand to watch it happen.

## Timing

The browser polls `/api/tick` every 40 ms, but the simulation does not advance by
whatever time that call happens to represent. It accumulates real elapsed time and
consumes it in fixed 20 ms steps, so the physics stay deterministic and the speed
slider is a genuine multiplier on simulated time.
