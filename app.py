import os

from flask import Flask, jsonify, render_template, request

from simulation.simulation import Simulation

app = Flask(__name__, static_folder="static", template_folder="templates")
sim = Simulation()

ACTIONS = {
    "pause": lambda payload: sim.pause(),
    "resume": lambda payload: sim.resume(),
    "reset": lambda payload: sim.reset(),
    "step": lambda payload: sim.step(),
    "single_step": lambda payload: sim.single_step(),
    "toggle_ukf": lambda payload: sim.toggle_ukf(),
    "reset_ukf": lambda payload: sim.reset_ukf(),
    "experiment": lambda payload: sim.apply_experiment(payload.get("name", "all_sensors")),
    "set_debug": lambda payload: sim.set_debug(bool(payload.get("enabled", False))),
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(sim.get_state())


@app.route("/api/tick", methods=["POST"])
def api_tick():
    """Advance the simulation and return the new state in a single round trip."""
    sim.step()
    return jsonify(sim.get_state())


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        try:
            sim.apply_config(payload)
        except (TypeError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        return jsonify({"status": "ok", "config": sim.get_config()})
    return jsonify(sim.get_config())


@app.route("/api/action", methods=["POST"])
def api_action():
    payload = request.get_json(silent=True) or {}
    action = payload.get("action")
    handler = ACTIONS.get(action)
    if handler is None:
        return jsonify({"status": "error", "message": f"unknown action: {action!r}"}), 400
    handler(payload)
    return jsonify({"status": "ok"})


@app.route("/api/keys", methods=["POST"])
def api_keys():
    payload = request.get_json(silent=True) or {}
    key = payload.get("key")
    if key:
        sim.set_key_state(key, bool(payload.get("pressed", False)))
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    # Not 5000: macOS hands that port to the AirPlay Receiver, which answers 403.
    port = int(os.environ.get("PORT", "5001"))
    print(f"UKF sandbox running on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
