const state = {
  debug: false,
  ukfEnabled: true,
  current: null,
  ukfDetailsVisible: false,
  inFlight: false,
  openInfo: null,
};

// Only forward/back and turn: a differential drive has no lateral degree of
// freedom, so there is nothing for A and D to command.
const controls = {
  W: false,
  S: false,
  ArrowLeft: false,
  ArrowRight: false,
};

/** What each tunable parameter actually does, shown by its "i" button. */
const INFO_TEXT = {
  speed: 'Multiplier on simulated time. The physics step stays fixed at 20 ms, so this only changes how many steps run per real second — it does not change the motion, the noise, or the filter. Turn it down to watch a single correction in detail.',
  alpha: 'How far the sigma points are spread from the mean before being propagated. Small values keep them close, which samples only the local behaviour of the motion and sensor models; large values spread them wide and capture more curvature. Below about 0.7 the centre point takes a large negative weight, and the covariance can briefly go non-positive-definite (the filter repairs it, but the estimate gets rough).',
  beta: 'Folds prior knowledge of the state distribution into the covariance weight of the centre sigma point. 2 is optimal for a Gaussian. It affects only the covariance — the mean estimate is unchanged — so raising it widens the covariance ellipse without moving the estimate.',
  kappa: 'Secondary scaling, normally 0. Positive values push the sigma points further out, acting much like a larger alpha. Strongly negative values (below about -2.9 for a 3-state filter) make the scaling degenerate; the input is clamped to keep the weights finite.',
  initialCovariance: 'How uncertain the filter says its *position* is at reset, in square inches. Large means it distrusts its starting guess and snaps hard onto the first few measurements. Small means it trusts the guess and converges slowly — set it small with a wrong starting pose and the filter can take a long time to find the robot. Use "Reset UKF" to see the effect.',
  initialHeading: 'How uncertain the filter says its *heading* is at reset, as a standard deviation in degrees. This is kept separate from the position covariance because the two are in different units, and heading punishes a bad guess much harder: the sigma points are spread across this range and each one carries the robot\u2019s motion off in a different direction, so a large value makes the averaged prediction come out short. Set it to 90° and watch dead reckoning under-travel badly until the sensors pull the heading in.',
  odometry: 'Noise on the distance the wheels report rolling each step, in inches. Odometry is the prediction input, so this doubles as how much the filter distrusts its own motion model: raise it and the filter leans on the IMU and distance sensors instead.',
  odometryAngular: 'Noise on the heading change the wheels report each step, in degrees. This is what makes dead reckoning curve away from the truth. With no IMU it is the only heading information the filter has, so raising it makes the estimate slowly rotate off course.',
  imu: 'Noise on the absolute heading reading, in degrees. Unlike odometry this does not accumulate — it is a direct measurement of orientation. Raise it and the filter has to infer heading from how the distance readings change instead.',
  distance: 'Noise on each rangefinder reading, in inches. These are the only measurements that tie the estimate to a fixed position, so raising it weakens every correction and lets the estimate drift with the odometry.',
  toggles: 'Turning a sensor off removes it from the filter entirely, not just from the display. Odometry is the prediction input, so switching it off leaves the filter with no motion model — it must localise from the absolute sensors alone. Note that four distance sensors with no heading reference are ambiguous: a square arena looks identical under 90-degree rotations, so turning off both odometry and the IMU can leave the filter locked onto a rotated pose.',
};

const configInputs = {
  alpha: document.getElementById('alphaInput'),
  beta: document.getElementById('betaInput'),
  kappa: document.getElementById('kappaInput'),
  initialCov: document.getElementById('initialCovInput'),
  initialHeading: document.getElementById('initialHeadingInput'),
  odomNoise: document.getElementById('odomNoiseInput'),
  odomAngularNoise: document.getElementById('odomAngularNoiseInput'),
  imuNoise: document.getElementById('imuNoiseInput'),
  distanceNoise: document.getElementById('distanceNoiseInput'),
  simSpeed: document.getElementById('simSpeed'),
  debugToggle: document.getElementById('debugToggle'),
};

const sensorTable = document.getElementById('sensorTable');
const ukfStateText = document.getElementById('ukfStateText');
const ukfDetails = document.getElementById('ukfDetails');
const canvas = document.getElementById('arenaCanvas');
const ctx = canvas.getContext('2d');
const graphCanvas = document.getElementById('graphCanvas');
const graphCtx = graphCanvas.getContext('2d');
const leftWheelBar = document.getElementById('leftWheelBar');
const rightWheelBar = document.getElementById('rightWheelBar');
const leftWheelValue = document.getElementById('leftWheelValue');
const rightWheelValue = document.getElementById('rightWheelValue');

const COLORS = {
  truth: '#ff6b6b',
  estimate: '#61dafb',
  ellipse: 'rgba(124,227,139,0.9)',
  sigma: 'rgba(97,218,251,0.85)',
  ray: 'rgba(249,199,79,0.75)',
  rayOff: 'rgba(255,255,255,0.14)',
  grid: '#16283c',
  gridFoot: '#22405e',
  axis: '#4a648a',
  text: '#dfeaf7',
  heading: '#f9c74f',
};

const GRID_STEP = 12; // inches (one foot)
const LABEL_STEP = 24;

function postJSON(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function sendAction(action, extra = {}) {
  postJSON('/api/action', { action, ...extra }).catch((err) => console.error('Action error', err));
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
  return Number(value).toFixed(digits);
}

function degrees(radians) {
  return (radians * 180) / Math.PI;
}

async function syncConfigFromServer() {
  try {
    const response = await fetch('/api/config');
    if (!response.ok) return;
    const config = await response.json();
    configInputs.alpha.value = config.ukf.alpha;
    configInputs.beta.value = config.ukf.beta;
    configInputs.kappa.value = config.ukf.kappa;
    configInputs.initialCov.value = config.ukf.initial_covariance;
    configInputs.initialHeading.value = config.ukf.initial_heading_deg;
    configInputs.odomNoise.value = config.sensor_noise.odometry.toFixed(2);
    configInputs.odomAngularNoise.value = degrees(config.sensor_noise.odometry_angular).toFixed(3);
    configInputs.imuNoise.value = degrees(config.sensor_noise.imu).toFixed(2);
    configInputs.distanceNoise.value = config.sensor_noise.distance.toFixed(2);
    configInputs.simSpeed.value = config.simulation.speed;
    state.debug = !!config.simulation.debug;
    configInputs.debugToggle.checked = state.debug;
    document.querySelectorAll('.sensor-toggle').forEach((item) => {
      item.checked = !!config.sensors[item.dataset.key];
    });
  } catch (err) {
    console.error('Config sync error', err);
  }
}

function applyConfig() {
  postJSON('/api/config', {
    ukf: {
      alpha: Number(configInputs.alpha.value),
      beta: Number(configInputs.beta.value),
      kappa: Number(configInputs.kappa.value),
      initial_covariance: Number(configInputs.initialCov.value),
      initial_heading_deg: Number(configInputs.initialHeading.value),
    },
    sensor_noise: {
      odometry: Number(configInputs.odomNoise.value),
      odometry_angular: (Number(configInputs.odomAngularNoise.value) * Math.PI) / 180.0,
      imu: (Number(configInputs.imuNoise.value) * Math.PI) / 180.0,
      distance: Number(configInputs.distanceNoise.value),
    },
  }).catch((err) => console.error('Config error', err));
}

function renderSensorTable() {
  const sensors = state.current.sensors || {};
  const enabled = state.current.sensor_enabled || {};
  const outOfRange = (state.current.world || {}).out_of_range ?? -1;

  const rangeRow = (key, name) => {
    const reading = sensors[`${key}_noisy`];
    const shown = sensors[`${key}_in_range`]
      ? `${formatNumber(sensors[`${key}_true`])} / ${formatNumber(reading)}`
      : `${formatNumber(sensors[`${key}_true`])} / <span class="out-of-range">${outOfRange} (out of range)</span>`;
    return [key, name, shown];
  };

  const rows = [
    ['odom', 'Odometry X pod', `${formatNumber(sensors.odom_forward_true, 3)} / ${formatNumber(sensors.odom_forward_noisy, 3)} in`],
    ['odom', 'Odometry Y pod', `${formatNumber(sensors.odom_lateral_true, 3)} / ${formatNumber(sensors.odom_lateral_noisy, 3)} in`],
    ['odom', 'Odometry turn', `${formatNumber(degrees(sensors.odom_theta_true), 3)}° / ${formatNumber(degrees(sensors.odom_theta_noisy), 3)}°`],
    ['imu', 'IMU heading', `${formatNumber(degrees(sensors.imu_true))}° / ${formatNumber(degrees(sensors.imu_noisy))}°`],
    rangeRow('front', 'Front'),
    rangeRow('back', 'Back'),
    rangeRow('left', 'Left'),
    rangeRow('right', 'Right'),
  ];

  sensorTable.innerHTML = rows
    .map(([key, name, value]) => {
      const off = enabled[key] ? '' : ' class="sensor-off"';
      return `<tr${off}><td>${name}</td><td>${value}</td></tr>`;
    })
    .join('');
}

function renderWheels() {
  const wheels = state.current.wheels || { left: 0, right: 0, max: 1 };
  const setBar = (bar, valueEl, speed) => {
    const fraction = Math.max(-1, Math.min(1, speed / wheels.max));
    // The bar grows out from the centre so reversed wheels are visibly opposite.
    bar.style.left = fraction >= 0 ? '50%' : `${50 + fraction * 50}%`;
    bar.style.width = `${Math.abs(fraction) * 50}%`;
    bar.style.background = fraction >= 0 ? COLORS.estimate : COLORS.truth;
    valueEl.textContent = `${speed.toFixed(1)} in/s`;
  };
  setBar(leftWheelBar, leftWheelValue, wheels.left);
  setBar(rightWheelBar, rightWheelValue, wheels.right);
}

function renderUKFState() {
  const est = state.current.ukf_estimate || {};
  const truePose = state.current.true_pose || {};

  ukfStateText.innerHTML = `
    <div class="legend-line"><span class="swatch" style="background:${COLORS.truth}"></span>True &nbsp; X ${formatNumber(truePose.x)} | Y ${formatNumber(truePose.y)} | θ ${formatNumber(degrees(truePose.theta), 1)}°</div>
    <div class="legend-line"><span class="swatch" style="background:${COLORS.estimate}"></span>UKF &nbsp; X ${formatNumber(est.x)} | Y ${formatNumber(est.y)} | θ ${formatNumber(degrees(est.theta), 1)}°</div>
    <div>Position error <strong>${formatNumber(state.current.position_error)} in</strong> &nbsp; Heading error <strong>${formatNumber(degrees(state.current.heading_error), 2)}°</strong></div>
    <div>σx ${formatNumber(state.current.sigma_x)} in | σy ${formatNumber(state.current.sigma_y)} in | σθ ${formatNumber(degrees(state.current.sigma_theta), 2)}°</div>
  `;

  if (state.ukfDetailsVisible && state.current.ukf_details) {
    ukfDetails.textContent = state.current.ukf_details;
    ukfDetails.classList.remove('hidden');
  } else {
    ukfDetails.classList.add('hidden');
  }
}

function drawArena() {
  const world = state.current.world || {};
  const size = world.size || 72;
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const scale = (Math.min(canvas.width, canvas.height) / (2 * size)) * 0.94;
  const mapHalf = size * scale;

  const worldToScreen = (x, y) => ({ x: centerX + x * scale, y: centerY - y * scale });

  ctx.fillStyle = '#09131c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 1;
  for (let i = -size; i <= size; i += GRID_STEP) {
    const isLabelled = Math.abs(i) % LABEL_STEP === 0;
    ctx.strokeStyle = isLabelled ? COLORS.gridFoot : COLORS.grid;
    const gx = centerX + i * scale;
    const gy = centerY + i * scale;
    ctx.beginPath();
    ctx.moveTo(gx, centerY - mapHalf);
    ctx.lineTo(gx, centerY + mapHalf);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(centerX - mapHalf, gy);
    ctx.lineTo(centerX + mapHalf, gy);
    ctx.stroke();
  }

  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(centerX, centerY - mapHalf);
  ctx.lineTo(centerX, centerY + mapHalf);
  ctx.moveTo(centerX - mapHalf, centerY);
  ctx.lineTo(centerX + mapHalf, centerY);
  ctx.stroke();

  ctx.strokeStyle = COLORS.text;
  ctx.lineWidth = 2;
  ctx.strokeRect(centerX - mapHalf, centerY - mapHalf, 2 * mapHalf, 2 * mapHalf);

  ctx.fillStyle = '#7d94b0';
  ctx.font = '11px system-ui, sans-serif';
  for (let i = -size; i <= size; i += LABEL_STEP) {
    if (i === 0) continue;
    ctx.fillText(String(i), centerX + i * scale - 8, centerY + 14);
    ctx.fillText(String(i), centerX + 6, centerY - i * scale + 4);
  }
  ctx.fillText('inches', centerX + mapHalf - 42, centerY + 14);

  const truePose = state.current.true_pose || { x: 0, y: 0, theta: 0 };
  const estPose = state.current.ukf_estimate || { x: 0, y: 0, theta: 0 };
  const halfSize = ((state.current.robot || {}).width || 14) / 2;

  function drawTrack(xs, ys, color) {
    if (!Array.isArray(xs) || xs.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.4;
    ctx.beginPath();
    xs.forEach((x, i) => {
      const p = worldToScreen(x, ys[i]);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawRobot(pose, color, showWheels) {
    const p = worldToScreen(pose.x, pose.y);
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(-pose.theta);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(-halfSize * scale, -halfSize * scale, 2 * halfSize * scale, 2 * halfSize * scale);

    if (showWheels) {
      // Draw the two drive trains along the sides so the chassis reads as a
      // differential drive rather than an omnidirectional block.
      ctx.lineWidth = 4;
      ctx.globalAlpha = 0.7;
      [-1, 1].forEach((side) => {
        ctx.beginPath();
        ctx.moveTo(-halfSize * 0.7 * scale, side * halfSize * scale);
        ctx.lineTo(halfSize * 0.7 * scale, side * halfSize * scale);
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2;
    }

    const tip = (halfSize + 7) * scale;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(tip, 0);
    ctx.moveTo(tip - 4.5 * scale, -3 * scale);
    ctx.lineTo(tip, 0);
    ctx.lineTo(tip - 4.5 * scale, 3 * scale);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  const graphs = state.current.graphs || {};
  drawTrack(graphs.x_track, graphs.y_track, COLORS.truth);
  drawTrack(graphs.est_x_track, graphs.est_y_track, COLORS.estimate);

  // Sensor rays are always drawn. A ray that finds nothing within range simply
  // is not there, which is exactly what the filter sees.
  ctx.font = '10px system-ui, sans-serif';
  for (const ray of state.current.sensor_rays || []) {
    if (!ray.in_range) continue;
    const p1 = worldToScreen(ray.start_x, ray.start_y);
    const p2 = worldToScreen(ray.end_x, ray.end_y);
    ctx.strokeStyle = ray.enabled ? COLORS.ray : COLORS.rayOff;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();

    ctx.fillStyle = ray.enabled ? COLORS.ray : COLORS.rayOff;
    ctx.beginPath();
    ctx.arc(p2.x, p2.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  if (state.current.debug) {
    const ellipse = state.current.covariance_ellipse;
    if (ellipse) {
      const px = worldToScreen(ellipse.cx, ellipse.cy);
      ctx.save();
      ctx.translate(px.x, px.y);
      ctx.rotate(-ellipse.angle);
      ctx.strokeStyle = COLORS.ellipse;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(0, 0, ellipse.rx * scale, ellipse.ry * scale, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    ctx.fillStyle = COLORS.sigma;
    for (const pt of state.current.sigma_points || []) {
      const sp = worldToScreen(pt.x, pt.y);
      ctx.beginPath();
      ctx.arc(sp.x, sp.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const truePos = worldToScreen(truePose.x, truePose.y);
  const estPos = worldToScreen(estPose.x, estPose.y);
  ctx.strokeStyle = 'rgba(255,255,255,0.45)';
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(truePos.x, truePos.y);
  ctx.lineTo(estPos.x, estPos.y);
  ctx.stroke();
  ctx.setLineDash([]);

  drawRobot(truePose, COLORS.truth, true);
  drawRobot(estPose, COLORS.estimate, false);
}

function drawGraphs() {
  const graph = state.current.graphs || {};
  const w = graphCanvas.width;
  const h = graphCanvas.height;
  const padLeft = 44;
  const padRight = 10;
  const padTop = 22;
  const padBottom = 22;

  graphCtx.fillStyle = '#08131c';
  graphCtx.fillRect(0, 0, w, h);

  const series = [
    { data: graph.position_error || [], color: COLORS.truth, label: 'Position error (in)' },
    { data: graph.heading_error || [], color: COLORS.heading, label: 'Heading error (°)' },
  ];

  // Both series share one time axis but have different units, so each gets its
  // own vertical scale anchored at zero. Anchoring at zero is what makes the
  // traces comparable over time instead of self-normalising every frame.
  const scales = series.map(({ data }) =>
    Math.max(data.reduce((acc, v) => Math.max(acc, Math.abs(v)), 0), 1e-3),
  );

  graphCtx.strokeStyle = '#24384f';
  graphCtx.lineWidth = 1;
  graphCtx.beginPath();
  graphCtx.moveTo(padLeft, padTop);
  graphCtx.lineTo(padLeft, h - padBottom);
  graphCtx.lineTo(w - padRight, h - padBottom);
  graphCtx.stroke();

  const midY = (padTop + (h - padBottom)) / 2;
  graphCtx.setLineDash([3, 3]);
  graphCtx.beginPath();
  graphCtx.moveTo(padLeft, midY);
  graphCtx.lineTo(w - padRight, midY);
  graphCtx.stroke();
  graphCtx.setLineDash([]);

  series.forEach(({ data }, idx) => {
    if (data.length < 2) return;
    graphCtx.strokeStyle = series[idx].color;
    graphCtx.lineWidth = 1.5;
    graphCtx.beginPath();
    data.forEach((value, i) => {
      const x = padLeft + (i / (data.length - 1)) * (w - padLeft - padRight);
      const y = midY - (value / scales[idx]) * (midY - padTop);
      if (i === 0) graphCtx.moveTo(x, y);
      else graphCtx.lineTo(x, y);
    });
    graphCtx.stroke();
  });

  graphCtx.font = '10px system-ui, sans-serif';
  graphCtx.fillStyle = '#8ea6c4';
  graphCtx.fillText(`±${scales[0].toFixed(1)}`, 4, padTop + 4);
  graphCtx.fillText('0', 4, midY + 3);
  graphCtx.fillText('last 300 steps', padLeft + 4, h - 6);

  let legendX = padLeft + 90;
  series.forEach(({ color, label }, idx) => {
    graphCtx.fillStyle = color;
    graphCtx.fillRect(legendX, 10, 8, 8);
    graphCtx.fillStyle = '#8ea6c4';
    const text = `${label} ±${scales[idx].toFixed(1)}`;
    graphCtx.fillText(text, legendX + 12, 18);
    legendX += graphCtx.measureText(text).width + 26;
  });
}

function render() {
  if (!state.current) return;
  renderSensorTable();
  renderWheels();
  renderUKFState();
  drawArena();
  drawGraphs();
}

async function tick() {
  // Skip if the previous round trip has not landed, so a slow response cannot
  // pile up an unbounded queue of requests.
  if (state.inFlight) return;
  state.inFlight = true;
  try {
    const response = await postJSON('/api/tick', {});
    if (response.ok) {
      state.current = await response.json();
      render();
    }
  } catch (err) {
    console.error('Tick error', err);
  } finally {
    state.inFlight = false;
  }
}

/**
 * Force the server's held-key set to match this page's.
 *
 * The server only ever hears key transitions, so it has no way to notice a key
 * that was pressed but never released -- by a reload mid-keypress, another tab,
 * or anything else poking /api/keys. A latched key drives the robot forever and
 * looks exactly like the simulation misbehaving. Restating the full set on load
 * makes the browser authoritative instead.
 */
function syncKeysToServer() {
  return Promise.all(
    Object.keys(controls).map((key) =>
      postJSON('/api/keys', { key, pressed: controls[key] }).catch(() => {}),
    ),
  );
}

function handleKeyboard(event) {
  const key = event.key.length === 1 ? event.key.toUpperCase() : event.key;
  if (!(key in controls)) return;
  const pressed = event.type === 'keydown';
  if (controls[key] === pressed) return; // ignore auto-repeat
  controls[key] = pressed;
  postJSON('/api/keys', { key, pressed }).catch((err) => console.error('Keyboard sync error', err));
}

window.addEventListener('keydown', (event) => {
  const key = event.key.length === 1 ? event.key.toUpperCase() : event.key;
  if (key in controls || event.key === ' ') event.preventDefault();
  handleKeyboard(event);
});
window.addEventListener('keyup', handleKeyboard);

// Releasing a key while the window is unfocused would otherwise strand the
// robot driving forever.
window.addEventListener('blur', () => {
  Object.keys(controls).forEach((key) => {
    controls[key] = false;
  });
  syncKeysToServer();
});

const uiActions = {
  pauseBtn: () => sendAction('pause'),
  resumeBtn: () => sendAction('resume'),
  // Reset clears the server's key set, so restate ours or a key held across the
  // click would stay silent until it is released and pressed again.
  resetBtn: async () => {
    sendAction('reset');
    await syncKeysToServer();
  },
  stepBtn: () => sendAction('single_step'),
  ukfToggleBtn: () => {
    state.ukfEnabled = !state.ukfEnabled;
    document.getElementById('ukfToggleBtn').textContent = state.ukfEnabled ? 'Disable UKF' : 'Enable UKF';
    sendAction('toggle_ukf');
  },
  resetUkfBtn: () => sendAction('reset_ukf'),
  showUkfBtn: () => {
    state.ukfDetailsVisible = !state.ukfDetailsVisible;
    document.getElementById('showUkfBtn').textContent = state.ukfDetailsVisible
      ? 'Hide UKF Details'
      : 'Show UKF Details';
  },
};

Object.entries(uiActions).forEach(([id, handler]) => {
  document.getElementById(id).addEventListener('click', handler);
});

document.querySelectorAll('.info-btn').forEach((button) => {
  button.addEventListener('click', () => {
    const panel = document.getElementById(button.dataset.panel);
    const key = button.dataset.info;
    const alreadyOpen = state.openInfo === key && !panel.classList.contains('hidden');

    document.querySelectorAll('.info-panel').forEach((p) => p.classList.add('hidden'));
    document.querySelectorAll('.info-btn').forEach((b) => b.setAttribute('aria-expanded', 'false'));

    if (alreadyOpen) {
      state.openInfo = null;
      return;
    }
    panel.textContent = INFO_TEXT[key] || '';
    panel.classList.remove('hidden');
    button.setAttribute('aria-expanded', 'true');
    state.openInfo = key;
  });
});

[
  configInputs.alpha,
  configInputs.beta,
  configInputs.kappa,
  configInputs.initialCov,
  configInputs.initialHeading,
  configInputs.odomNoise,
  configInputs.odomAngularNoise,
  configInputs.imuNoise,
  configInputs.distanceNoise,
].forEach((input) => input.addEventListener('change', applyConfig));

configInputs.simSpeed.addEventListener('input', () => {
  postJSON('/api/config', { simulation: { speed: Number(configInputs.simSpeed.value) } }).catch((err) =>
    console.error('Speed config error', err),
  );
});

configInputs.debugToggle.addEventListener('change', () => {
  state.debug = configInputs.debugToggle.checked;
  sendAction('set_debug', { enabled: state.debug });
});

document.querySelectorAll('.preset').forEach((btn) => {
  btn.addEventListener('click', async () => {
    sendAction('experiment', { name: btn.dataset.experiment });
    // Presets change noise and toggles server-side; pull the result back so the
    // panel is never showing stale numbers.
    await syncConfigFromServer();
  });
});

document.querySelectorAll('.sensor-toggle').forEach((toggle) => {
  toggle.addEventListener('change', () => {
    const sensors = {};
    document.querySelectorAll('.sensor-toggle').forEach((item) => {
      sensors[item.dataset.key] = item.checked;
    });
    postJSON('/api/config', { sensors }).catch((err) => console.error('Sensor toggle error', err));
  });
});

syncConfigFromServer();
syncKeysToServer();
setInterval(tick, 40);
