"""Live tuning of the controllers while the mission runs.

  python tune.py        # slider window; every change is written to tune.json and picked up by the running mission
                        # within ~0.2 s (no restart). "Reset" restores the defaults.

The mission reads values with tune.get(name): defaults if tune.json is missing/broken, always clamped to a safe
range, so a bad value can never make the arm jump.
"""
import json, os, time, threading
from pathlib import Path

HERE = Path(__file__).parent
FILE = HERE / "tune.json"
#            name                default  min    max    step   what it does
PARAMS = [("servo_kp",             0.6,  0.0,   1.5,   0.05, "servo P: how hard it corrects toward the capture zone"),
          ("servo_ki",             0.15, 0.0,   0.6,   0.01, "servo I: removes a steady offset (too high = overshoot)"),
          ("servo_kd",             0.10, 0.0,   0.6,   0.01, "servo D: damps oscillation"),
          ("servo_max_step_mm",    4.0,  1.0,   8.0,   0.5,  "servo: largest single correction (mm)"),
          ("servo_period_s",       0.10, 0.05,  0.30,  0.01, "servo: time per correction (longer = softer)"),
          ("peak_deg_s",           40.0, 15.0,  60.0,  1.0,  "big moves: peak joint speed (deg/s)"),
          ("descend_deg_s",        25.0, 8.0,   40.0,  1.0,  "approach/lift lines: speed (deg/s)"),
          ("min_move_s",           0.6,  0.4,   2.0,   0.05, "shortest time for any move (longer = softer starts/stops)"),
          ("sag_ki",               0.04, 0.0,   0.08,  0.005, "gravity-sag integral gain (lower = softer, more sag)")]
DEFAULTS = {p[0]: p[1] for p in PARAMS}
RANGES = {p[0]: (p[2], p[3]) for p in PARAMS}
_cache = {"t": 0.0, "mtime": None, "vals": dict(DEFAULTS)}
_lock = threading.Lock()


def get(name):
    """Current value (clamped). Re-reads tune.json at most every 0.2 s, only when it changed."""
    now = time.time()
    with _lock:
        if now - _cache["t"] > 0.2:
            _cache["t"] = now
            try:
                m = os.stat(FILE).st_mtime
                if m != _cache["mtime"]:
                    _cache["mtime"] = m
                    vals = dict(DEFAULTS); vals.update(json.loads(FILE.read_text()))
                    _cache["vals"] = vals
            except (OSError, ValueError):
                _cache["vals"] = dict(DEFAULTS) if not FILE.exists() else _cache["vals"]
        v = _cache["vals"].get(name, DEFAULTS[name])
    lo, hi = RANGES[name]
    try:
        return min(hi, max(lo, float(v)))
    except (TypeError, ValueError):
        return DEFAULTS[name]


def main():
    import tkinter as tk
    vals = dict(DEFAULTS)
    try:
        vals.update(json.loads(FILE.read_text()))
    except Exception:
        pass
    root = tk.Tk(); root.title("chand live tuning (changes apply immediately)")
    vars_ = {}

    def save(*_):
        FILE.write_text(json.dumps({k: round(v.get(), 4) for k, v in vars_.items()}, indent=1))

    for i, (name, d, lo, hi, step, desc) in enumerate(PARAMS):
        tk.Label(root, text=name, font=("Sans", 11, "bold"), anchor="w", width=18).grid(row=i, column=0, sticky="w")
        v = tk.DoubleVar(value=vals.get(name, d)); vars_[name] = v
        tk.Scale(root, variable=v, from_=lo, to=hi, resolution=step, orient="horizontal", length=320,
                 command=save).grid(row=i, column=1)
        tk.Label(root, text=desc, font=("Sans", 9), anchor="w", width=55).grid(row=i, column=2, sticky="w")

    def reset():
        for k, v in vars_.items():
            v.set(DEFAULTS[k])
        save()
    tk.Button(root, text="Reset to defaults", command=reset).grid(row=len(PARAMS), column=1, pady=6)
    save()
    root.mainloop()


if __name__ == "__main__":
    main()
