"""Auto-tuning while the mission runs (user, 04:40): Muse Spark continuously monitors the arms and fine-tunes the PID
and motion settings; any time an arm runs jittery, smooth it out.

Two loops in one background thread, both writing tune.json (read live by the arms through tune.get):
  FAST, local, every 1 s, free:  each arm's measured-joint jitter (execute.Mover.metrics). Jitter above the limit
        while moving -> SMOOTH NOW: servo gain -15 %, servo period +20 %, peak speed -10 %, min move +0.1 s.
        (Only ever makes things gentler; bounded by tune.RANGES.)
  SLOW, Muse Spark, every SPARK_EVERY_S when new data exists: Spark gets current settings, their safe ranges, per-arm
        smoothness numbers, the last servo runs (steps, oscillation = error sign flips, converged or not) and grasp
        results; it answers JSON {param: new value} + one-line reasons. Each change is clamped to the range and to
        +-30 % per step. Spark tunes for: smooth (no jitter) first, then servo converging in few steps, then speed.
Every change is printed, shown in the mission window and in Rerun, and appended to logs/autotune.jsonl.
"""
import json, threading, time
from pathlib import Path
import numpy as np
import tune, brain, execute, viz

HERE = Path(__file__).parent
JITTER_LIMIT_DEG = 0.35        # RMS high-frequency shake on the measured joints that counts as "jittery"
REVERSALS_LIMIT = 6.0          # back-and-forth velocity flips per second on the main joint
SPARK_EVERY_S = 45.0
MAX_REL_STEP = 0.30


def _write(changes, why, source):
    cur = {k: tune.get(k) for k in tune.DEFAULTS}
    new = dict(cur)
    for k, v in changes.items():
        if k not in tune.RANGES:
            continue
        lo, hi = tune.RANGES[k]
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        base = cur[k] if cur[k] != 0 else (hi - lo) * 0.1
        v = min(cur[k] + abs(base) * MAX_REL_STEP, max(cur[k] - abs(base) * MAX_REL_STEP, v))
        new[k] = round(min(hi, max(lo, v)), 4)
    diff = {k: (cur[k], new[k]) for k in new if abs(new[k] - cur[k]) > 1e-9}
    if not diff:
        return {}
    tune.FILE.write_text(json.dumps(new, indent=1))
    rec = {"t": time.strftime("%H:%M:%S"), "source": source, "why": why, "changes": diff}
    with open(HERE / "logs" / "autotune.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    txt = ", ".join(f"{k} {a:g}->{b:g}" for k, (a, b) in diff.items())
    print(f"  [autotune {source}] {txt}  ({why})", flush=True)
    viz.text(f"autotune {source}: {txt} ({why})")
    return diff


class AutoTuner(threading.Thread):
    def __init__(self, arms, use_spark=True, status=None):
        super().__init__(daemon=True)
        self.arms, self.use_spark, self.status = arms, use_spark, status if status is not None else {}
        self.running = True
        self.last_spark, self.last_smooth = 0.0, 0.0
        self.history = []          # recent per-arm metrics (moving only)
        self.n_servo_seen = 0

    def metrics_now(self):
        out = {}
        for name, a in self.arms.items():
            m = a["mv"].metrics()
            if m is not None and m["moved_deg"] > 3:
                out[name] = m
                viz.value(f"{name}/smooth/jitter_deg", m["jitter_deg"]); viz.value(f"{name}/smooth/reversals_per_s", m["reversals_per_s"])
        return out

    def smooth_now(self, arm, m):
        why = f"{arm} arm jittery: shake {m['jitter_deg']} deg, {m['reversals_per_s']} reversals/s"
        d = _write({"servo_kp": tune.get("servo_kp") * 0.85, "servo_period_s": tune.get("servo_period_s") * 1.2,
                    "peak_deg_s": tune.get("peak_deg_s") * 0.9, "min_move_s": tune.get("min_move_s") + 0.1}, why, "local")
        if d:
            self.status["autotune"] = f"smoothed ({why})"

    def ask_spark(self):
        cur = {k: tune.get(k) for k in tune.DEFAULTS}
        recent = [h for h in self.history if time.time() - h["t"] < 120]
        servo = execute.SERVO_LOG[-6:]
        payload = {
            "current_settings": cur,
            "safe_ranges": {k: list(v) for k, v in tune.RANGES.items()},
            "meaning": {p[0]: p[5] for p in tune.PARAMS},
            "arm_smoothness_last_2_min": {arm: {
                "jitter_deg_mean": round(float(np.mean([h["m"][arm]["jitter_deg"] for h in recent if arm in h["m"]] or [0])), 3),
                "jitter_deg_max": round(float(np.max([h["m"][arm]["jitter_deg"] for h in recent if arm in h["m"]] or [0])), 3),
                "reversals_per_s_max": round(float(np.max([h["m"][arm]["reversals_per_s"] for h in recent if arm in h["m"]] or [0])), 2),
                "track_err_deg_mean": round(float(np.mean([h["m"][arm]["track_err_deg"] for h in recent if arm in h["m"]] or [0])), 2)}
                for arm in self.arms},
            "jitter_limit_deg": JITTER_LIMIT_DEG,
            "last_servo_runs": [{k: v for k, v in r.items() if k != "t"} for r in servo],
        }
        sys_p = ("You tune two small robot arms (SO-101, hobby servos) while they pick objects. Goals in order: (1) motion "
                 "is smooth, no jitter or back-and-forth; (2) the camera servo reaches the capture zone ('inside') in few "
                 "steps without oscillating (error sign flips = oscillation); (3) speed. Change at most 3 settings, small "
                 "steps (<= 30 %), stay inside the safe ranges, change nothing if all is fine. Reply ONLY with JSON: "
                 '{"changes": {"<setting>": <number>, ...}, "why": "<one short sentence>"}')
        t0 = time.time()
        reply = brain.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": json.dumps(payload)}],
                           tag="autotune", reasoning_effort="low")
        d = brain.parse_json(reply)
        diff = _write(d.get("changes") or {}, d.get("why", "spark"), f"spark {time.time() - t0:.0f}s")
        self.status["autotune"] = (f"spark: {d.get('why', '')}" if diff else f"spark: no change ({d.get('why', '')})")

    def run(self):
        while self.running:
            time.sleep(1.0)
            try:
                m = self.metrics_now()
                if m:
                    self.history.append({"t": time.time(), "m": m}); self.history = self.history[-300:]
                for arm, x in m.items():
                    if (x["jitter_deg"] > JITTER_LIMIT_DEG or x["reversals_per_s"] > REVERSALS_LIMIT) and time.time() - self.last_smooth > 4:
                        self.last_smooth = time.time(); self.smooth_now(arm, x)
                new_data = len(execute.SERVO_LOG) > self.n_servo_seen or len(self.history) > 20
                if self.use_spark and new_data and time.time() - self.last_spark > SPARK_EVERY_S:
                    self.last_spark = time.time(); self.n_servo_seen = len(execute.SERVO_LOG)
                    self.ask_spark()
            except Exception as e:
                print("  [autotune] error (ignored):", e, flush=True)
