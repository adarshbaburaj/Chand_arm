"""Chand Moon mode: look -> think -> move -> verify, until no red ore is left.

  python mission.py [--backend meta_api|glimmer_local|rule] [--step] [--offline] [--max-picks N]

--step     pause before every phase (SPACE = next). Without it, SPACE pauses/resumes.
--offline  no network at all except 127.0.0.1 (for the unplugged-ethernet shot)
--no-judge skip the TypeSafe Jev camera checks (target? aimed? held?)
Keys in the window: SPACE pause/next, Q abort (finishes the current safe move, then goes home).
"""
import json, sys, time, socket
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ARGS = sys.argv[1:]
BACKEND = ARGS[ARGS.index("--backend") + 1] if "--backend" in ARGS else "meta_api"
STEP = "--step" in ARGS
JUDGE = "--no-judge" not in ARGS and "--offline" not in ARGS
MAX_PICKS = int(ARGS[ARGS.index("--max-picks") + 1]) if "--max-picks" in ARGS else 8

if "--offline" in ARGS:
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _connect = socket.socket.connect
    def _guard(self, addr):
        if isinstance(addr, tuple) and addr[0] not in ("127.0.0.1", "localhost", "::1"):
            raise ConnectionRefusedError(f"OFFLINE: blocked {addr}")
        return _connect(self, addr)
    socket.socket.connect = _guard

import cv2
import kin, perceive, brain, execute, judge
from execute import Mover

FOLLOWER_PORT, ARM_ID = "/dev/ttyFollowerLeft", "thumbi_right"


def _fails(r, history):
    return sum(1 for h in history if h["result"] != "mined" and np.hypot(h["at"][0] - r["x_mm"], h["at"][1] - r["y_mm"]) < 30)


def plan_next(rule, rocks, history):
    """Brain: pick the next rock. Strict JSON check, one retry, then the plain rule.
    A rock that failed twice (or that Jev said is not a target) is not offered again."""
    allowed = [r["id"] for r in rocks if r["colour"] == rule["pick"]["colour"] and _fails(r, history) < 2
               and not any(h["result"].startswith("not a target") and np.hypot(h["at"][0] - r["x_mm"], h["at"][1] - r["y_mm"]) < 30 for h in history)]
    if not allowed:
        return {"skill": "done", "why": "no target colour left"}, "rule"
    if BACKEND == "rule":
        r = min((r for r in rocks if r["id"] in allowed), key=lambda r: np.hypot(r["x_mm"], r["y_mm"]))
        return {"skill": "pick", "obj": r["id"], "why": "nearest first"}, "rule"
    sys_p = ("You are the planner of a mining rover arm. Rule from the operator: " + rule["rule_text"] +
             ' Reply with ONLY one JSON object, either {"skill":"pick","obj":"<id>","why":"<short>"} or '
             '{"skill":"done","why":"<short>"}. Choose an object that the rule says to pick. Prefer one that is '
             "easy and safe to reach (not too far, not next to other objects). If the last attempt on an object failed, "
             "prefer a different object next.")
    user = json.dumps({"objects_on_table_mm": [{k: r[k] for k in ("id", "colour", "x_mm", "y_mm")} for r in rocks if r["id"] in allowed],
                       "arm_base_at": [0, 0], "reach_mm": execute.REACH_MM, "bin": "red_bin",
                       "already_tried": history[-4:]})
    for attempt in range(2):
        try:
            t0 = time.time()
            reply = brain.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                               backend=BACKEND, tag="plan", reasoning_effort="low")
            d = brain.parse_json(reply)
            if d.get("skill") == "done" or (d.get("skill") == "pick" and d.get("obj") in allowed):
                return d, f"{BACKEND} {time.time() - t0:.1f}s"
            print("  brain gave an invalid choice:", d, flush=True)
        except Exception as e:
            print("  brain error:", e, flush=True)
    r = min((r for r in rocks if r["id"] in allowed), key=lambda r: np.hypot(r["x_mm"], r["y_mm"]))
    return {"skill": "pick", "obj": r["id"], "why": "fallback: nearest"}, "rule-fallback"


def main():
    import tkinter as tk
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    skill = json.load(open(HERE / "skill.json")); rule = skill["rule"]
    calib = perceive.load_calib()
    root = tk.Tk(); root.title("chand mission")
    lab = tk.Label(root, font=("Sans", 15), width=72, height=7, justify="left"); lab.pack()
    st = {"pause": True, "quit": False, "step_wait": False}
    def key(e):
        if e.char == " ":
            if st["step_wait"]: st["step_wait"] = False
            else: st["pause"] = not st["pause"]
        if e.char and e.char.lower() == "q": st["quit"] = True; st["pause"] = False; st["step_wait"] = False
    root.bind("<Key>", key)
    def on_close():  # window X = abort (go home), never destroy the window under the running loop
        st["quit"] = True; st["pause"] = False; st["step_wait"] = False
    root.protocol("WM_DELETE_WINDOW", on_close)
    info = {"mined": 0, "t0": None, "phase": "", "target": "", "brain": "", "judge": ""}
    def show(extra=""):
        try:
            _show(extra)
        except tk.TclError:  # window gone: keep the robot logic alive
            st["pause"] = False
    def _show(extra=""):
        el = 0 if info["t0"] is None else time.time() - info["t0"]
        lab.configure(text=f"CHAND  {'PAUSED (SPACE)' if st['pause'] else 'running'}   mined {info['mined']}   t {el:5.1f}s\n"
                           f"target {info['target']}   phase {info['phase']}\nbrain {info['brain']}\njev {info['judge']}\n{extra}")
        root.update()
    def hold():
        show(); return st["pause"]
    def step(name):
        if st["quit"]:
            raise KeyboardInterrupt("aborted by user")
        info["phase"] = name
        if STEP and not st["quit"]:
            st["step_wait"] = True
            while st["step_wait"]:
                show(f"STEP MODE: SPACE to do '{name}'"); time.sleep(0.05)
        show()

    follower = SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=ARM_ID, max_relative_target=5.0))
    follower.connect(calibrate=False)
    mv = Mover(follower, hold)
    home = mv.q.copy()
    history, log = [], []
    show("SPACE to start the mission")
    while st["pause"] and not st["quit"]:
        show("SPACE to start the mission"); time.sleep(0.05)
    info["t0"] = time.time()
    try:
        for n in range(MAX_PICKS):
            if st["quit"]: break
            info["phase"] = "look"; show()
            frame = perceive.grab()                     # arm is home = out of view
            rocks, _ = perceive.scene(frame, calib)
            cv2.imwrite(str(HERE / "snapshots" / f"mission_{n}.jpg"), perceive.draw(frame, *perceive.detect(frame)))
            info["phase"] = "think"; show()
            d, src = plan_next(rule, rocks, history)
            info["brain"] = f"{src}: {json.dumps(d)}"
            print(f"[{n}] {len(rocks)} red seen -> {src}: {d}", flush=True)
            if d["skill"] == "done":
                break
            rock = next(r for r in rocks if r["id"] == d["obj"])
            info["target"] = f'{rock["id"]} ({rock["x_mm"]},{rock["y_mm"]}) mm'
            if JUDGE:
                try:
                    p = judge.is_target(frame, rock, rule["rule_text"])
                    info["judge"] = f"target? {p:.2f}"; show()
                    print(f"  jev: is target {p:.2f}", flush=True)
                    if p < judge.SURE_NO:
                        history.append({"obj": rock["id"], "at": [rock["x_mm"], rock["y_mm"]], "result": f"not a target (jev {p:.2f})"})
                        continue
                except Exception as e:
                    print("  jev target check failed (ignored):", e, flush=True)
            try:
                plan = execute.plan_pick(rock, skill, calib)
            except ValueError as e:
                print("  cannot plan:", e, flush=True); history.append({"obj": rock["id"], "at": [rock["x_mm"], rock["y_mm"]], "result": f"unreachable: {e}"}); continue
            try:
                vis = None
                if JUDGE:
                    def on_look(d):
                        if "held" in d: info["judge"] = f"held? {d['held']['held']} ({d['held']['src']}, p {d['held']['p']:.2f})"
                        else: info["judge"] = f"aim: between {d['between']:.2f}  box {np.round(d.get('c', (0, 0)), 2).tolist()}  {d['s']} s"
                        show()
                    vis = execute.Vision(skill, calib, rock, lambda: perceive.grab(execute.WRIST_CAM, warm=8), on_look)
                res = execute.pick_place(mv, plan, skill, step, vis=vis)
            except KeyboardInterrupt:
                print("  aborted: opening gripper, going home", flush=True)
                mv.gripper(skill["gripper"]["open"] + 8, wait=0.3)
                if np.abs(mv.q - home).max() > 10:      # away from home: shoulder up a bit before swinging home
                    lift = mv.q.copy(); lift[1] = max(lift[1] - 15, home[1])   # never past the home (limit) value
                    mv.move(lift)
                break
            # verify placement with a fresh look (arm is home again)
            after, _ = perceive.scene(perceive.grab(), calib)
            still = any(np.hypot(r["x_mm"] - rock["x_mm"], r["y_mm"] - rock["y_mm"]) < 25 for r in after)
            ok = res["held"] and not still
            info["mined"] += int(ok)
            result = "mined" if ok else (res.get("reason") or ("missed grasp" if not res["held"] else "rock still on table"))
            history.append({"obj": rock["id"], "at": [rock["x_mm"], rock["y_mm"]], "result": result})
            log.append({"n": n, "rock": rock, "brain": d, "src": src, "held": res["held"], "grip": res["grip"], "result": result,
                        "aim": res.get("aim"), "hold": res.get("hold"), "grasps": res.get("grasps"), "lifted": res.get("lifted"),
                        "t": round(time.time() - info["t0"], 1)})
            print(f"  -> {result}", flush=True)
    finally:
        info["phase"] = "home"; show()
        try:
            if np.abs(mv.read() - home).max() > 3:
                st["pause"] = False; mv.move(home)
        finally:
            json.dump(log, open(HERE / "logs" / time.strftime("mission_%H%M%S.json"), "w"), indent=1)
            print(f"MISSION END: mined {info['mined']} in {time.time() - (info['t0'] or time.time()):.0f}s", flush=True)
            try: root.destroy()
            except Exception: pass
            follower.disconnect()


if __name__ == "__main__":
    main()
