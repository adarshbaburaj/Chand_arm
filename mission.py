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
TWO_ARM = "--one-arm" not in ARGS and (HERE / "calib_left.json").exists() and Path("/dev/ttyFollowerRight").exists()
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
import kin, perceive, brain, execute, judge, cams, viz, two_arm, assist
from execute import Mover

FOLLOWER_PORT, ARM_ID = "/dev/ttyFollowerLeft", "thumbi_right"


def _fails(r, history):
    return sum(1 for h in history if h["result"] != "mined" and np.hypot(h["at"][0] - r["x_mm"], h["at"][1] - r["y_mm"]) < 30)


_ORDER = []   # positions (mm) of the objects in the order Spark planned; ids change between frames, positions don't


def _match(pos, rocks, allowed):
    for r in rocks:
        if r["id"] in allowed and np.hypot(r["x_mm"] - pos[0], r["y_mm"] - pos[1]) < 30:
            return r
    return None


def plan_next(rule, rocks, history):
    """Brain. Muse Spark plans the ORDER of all target objects once (one ~5-10 s call per mission instead of one per
    pick); the plan is reused while picks succeed and asked again after a failure or a surprise.
    Strict JSON check, one retry, then the plain rule. Failed-twice / not-a-target objects are never offered."""
    global _ORDER
    allowed = [r["id"] for r in rocks if r["colour"] == rule["pick"]["colour"] and _fails(r, history) < 2
               and not any(h["result"].startswith("not a target") and np.hypot(h["at"][0] - r["x_mm"], h["at"][1] - r["y_mm"]) < 30 for h in history)]
    if not allowed:
        return {"skill": "done", "why": "no target colour left"}, "rule"
    last_ok = not history or history[-1]["result"] == "mined"
    if BACKEND != "rule" and last_ok:
        while _ORDER:
            r = _match(_ORDER[0], rocks, allowed)
            if r is not None:
                return {"skill": "pick", "obj": r["id"], "why": "next in Spark's plan"}, "plan (cached)"
            _ORDER.pop(0)                                   # already mined or moved: drop it
    if BACKEND == "rule":
        r = min((r for r in rocks if r["id"] in allowed), key=lambda r: np.hypot(r["x_mm"], r["y_mm"]))
        return {"skill": "pick", "obj": r["id"], "why": "nearest first"}, "rule"
    sys_p = ("You are the planner of a mining rover arm. Rule from the operator: " + rule["rule_text"] +
             ' Reply with ONLY one JSON object: {"skill":"plan","order":["<id>", ...],"why":"<short>"} listing EVERY '
             'object the rule says to pick, best first, or {"skill":"done","why":"<short>"}. Put easy, safe, well-'
             "reachable objects first; shapes with more past failures later. After a failed attempt, start with a "
             "different object than the one that failed.")
    mem = {}
    try:
        mem = json.loads((HERE / "grasp_memory.json").read_text())
    except Exception:
        pass
    user = json.dumps({"objects_on_table_mm": [{k: r.get(k) for k in ("id", "colour", "shape", "x_mm", "y_mm", "arm")} for r in rocks if r["id"] in allowed],
                       "arms": "two arms. 'arm' says who can reach each object: right = picks straight into the bin; "
                               "left = the left arm first moves it next to the right arm, then the right arm bins it "
                               "(slower, so put left-only objects last).",
                       "arm_base_at": [0, 0], "reach_mm": execute.REACH_MM, "bin": "red_bin",
                       "grasp_record_by_shape": {k: {"ok": v["ok"], "fail": v["fail"]} for k, v in mem.items()},
                       "already_tried": history[-4:]})
    for attempt in range(2):
        try:
            t0 = time.time()
            reply = brain.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                               backend=BACKEND, tag="plan", reasoning_effort="low")
            d = brain.parse_json(reply)
            if d.get("skill") == "done":
                return d, f"{BACKEND} {time.time() - t0:.1f}s"
            order = [o for o in d.get("order", []) if o in allowed]
            if d.get("skill") == "plan" and order:
                by_id = {r["id"]: r for r in rocks}
                _ORDER = [(by_id[o]["x_mm"], by_id[o]["y_mm"]) for o in order]
                return ({"skill": "pick", "obj": order[0], "why": d.get("why", ""), "plan": order},
                        f"{BACKEND} {time.time() - t0:.1f}s")
            print("  brain gave an invalid plan:", d, flush=True)
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

    if "--no-viz" not in ARGS:
        viz.start()                       # viewer first (see viz.start)
    cam = cams.Cams.start()
    print("cameras on:", list(cam.cams), flush=True)
    viz.stream(cam)
    follower = SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=ARM_ID, max_relative_target=5.0))
    follower.connect(calibrate=False)
    print("gripper boost", execute.boost_gripper(follower), flush=True)
    mv = Mover(follower, hold)
    home = mv.q.copy()
    mvL, homeL, spot = None, None, None
    if TWO_ARM:
        try:
            fL = SOFollower(SOFollowerRobotConfig(port=two_arm.LEFT_PORT, id=two_arm.LEFT_ID, max_relative_target=5.0))
            fL.connect(calibrate=False)
            execute.boost_gripper(fL)
            mvL = Mover(fL, hold, name="left"); homeL = mvL.q.copy()
            spot = two_arm.handoff_spot(skill)
            R2L, tzL = assist.frames()
            print(f"two-arm mode: left follower on, handoff spot px {spot}", flush=True); viz.text(f"two-arm mode, handoff spot {spot}")
        except Exception as e:
            print("two-arm mode OFF:", e, flush=True); mvL = None
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
            drawn = perceive.draw(frame, *perceive.detect(frame))
            cv2.imwrite(str(HERE / "snapshots" / f"mission_{n}.jpg"), drawn); viz.image("ai/top_detections", drawn)
            for r in rocks:     # who can reach it (painted workspace + a real grasp plan)
                if two_arm.can_pick("right", r["u"], r["v"], skill)[0]:
                    r["arm"] = "right"
                elif mvL is not None and two_arm.can_pick("left", r["u"], r["v"], skill)[0]:
                    r["arm"] = "left"
                else:
                    r["arm"] = "none"
            for r in [r for r in rocks if r["arm"] == "none"]:
                if not any(h["result"] == "unreachable by both arms" and np.hypot(h["at"][0] - r["x_mm"], h["at"][1] - r["y_mm"]) < 30 for h in history):
                    history.append({"obj": r["id"], "at": [r["x_mm"], r["y_mm"]], "result": "unreachable by both arms"})
                    print(f"  {r['id']} at {r['x_mm'], r['y_mm']} mm: no arm can reach it", flush=True)
            rocks = [r for r in rocks if r["arm"] != "none"]
            info["phase"] = "think"; show()
            d, src = plan_next(rule, rocks, history)
            info["brain"] = f"{src}: {json.dumps(d)}"
            print(f"[{n}] {len(rocks)} red seen -> {src}: {d}", flush=True); viz.text(f"[{n}] brain {src}: {d}")
            if d["skill"] == "done":
                break
            rock = next(r for r in rocks if r["id"] == d["obj"])
            info["target"] = f'{rock["id"]} {rock.get("shape")} ({rock["x_mm"]},{rock["y_mm"]}) mm -> {rock["arm"]} arm'
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
            rt, rt_src = execute.memory_get(rock.get("shape", "_other"))
            print(f"  shape {rock.get('shape')} -> start offset {rt} mm (radial, tangential) from {rt_src}", flush=True)
            if rock["arm"] == "left":
                print(f"  right arm can't reach it: LEFT arm picks it and puts it at the handoff spot {spot}", flush=True)
                viz.text(f"{rock['id']}: left arm hands it over to the right arm")
                try:
                    xL, yL = two_arm.px_to_arm("left", rock["u"], rock["v"])
                    visL = None
                    if JUDGE:
                        visL = execute.Vision(skill, two_arm.calib("left"), {"x_mm": xL, "y_mm": yL}, lambda: cam.fresh("left_wrist"),
                                              lambda d: None) if "left_wrist" in cam.cams else None
                    res = two_arm.left_handoff(mvL, (rock["u"], rock["v"]), skill, spot, step, vis=visL, shift_rt=rt)
                except KeyboardInterrupt:
                    print("  aborted: left arm opening + home", flush=True)
                    mvL.gripper(skill["gripper"]["open"] + 8, wait=0.3); mvL.move(homeL); break
                except ValueError as e:
                    res = {"held": False, "reason": f"left cannot plan: {e}"}
                result = "handed to the right arm" if res.get("held") else ("left: " + (res.get("reason") or "missed grasp"))
                history.append({"obj": rock["id"], "at": [rock["x_mm"], rock["y_mm"]], "result": result})
                log.append({"n": n, "rock": rock, "arm": "left", "result": result, "grasps": res.get("grasps"), "t": round(time.time() - info["t0"], 1)})
                print(f"  -> {result}", flush=True); viz.text(f"-> {result}")
                continue
            try:
                plan = execute.plan_pick(rock, skill, calib, shift=execute.rt_to_world(rt, np.arctan2(rock["y_mm"], rock["x_mm"])))
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
                push = None
                if mvL is not None:
                    def push(plan_r, rock=rock):
                        try:
                            pp = assist.plan_push((rock["x_mm"], rock["y_mm"]), plan_r["tip"], mvL.q, R2L, tzL)
                        except Exception as e:
                            print("  left push: cannot plan:", e, flush=True); return False
                        if pp["err"] > 6:
                            print(f"  left push: out of reach (IK err {pp['err']} mm)", flush=True); return False
                        print(f"  LEFT ARM PUSH: {pp['push_mm']} mm toward the right gripper", flush=True); viz.text("left arm pushes object into right gripper")
                        assist.do_push(mvL, pp, homeL); return True
                res = execute.pick_place(mv, plan, skill, step, vis=vis, assist=push)
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
            if res.get("reason") != "object lost":
                execute.memory_put(rock.get("shape", "_other"), res.get("shift_rt", rt), ok)
            info["mined"] += int(ok)
            result = "mined" if ok else (res.get("reason") or ("missed grasp" if not res["held"] else "rock still on table"))
            history.append({"obj": rock["id"], "at": [rock["x_mm"], rock["y_mm"]], "result": result})
            log.append({"n": n, "rock": rock, "brain": d, "src": src, "held": res["held"], "grip": res["grip"], "result": result,
                        "aim": res.get("aim"), "hold": res.get("hold"), "grasps": res.get("grasps"), "lifted": res.get("lifted"),
                        "t": round(time.time() - info["t0"], 1)})
            print(f"  -> {result}", flush=True); viz.text(f"-> {result}", "INFO" if ok else "WARN"); viz.value("mission/mined", info["mined"])
    finally:
        info["phase"] = "home"; show()
        try:
            if np.abs(mv.read() - home).max() > 3:
                st["pause"] = False; mv.move(home)
        finally:
            if mvL is not None:
                try:
                    if np.abs(mvL.read() - homeL).max() > 3: mvL.move(homeL)
                    mvL.f.disconnect()
                except Exception as e:
                    print("left arm shutdown:", e, flush=True)
            json.dump(log, open(HERE / "logs" / time.strftime("mission_%H%M%S.json"), "w"), indent=1)
            print(f"MISSION END: mined {info['mined']} in {time.time() - (info['t0'] or time.time()):.0f}s", flush=True)
            try: root.destroy()
            except Exception: pass
            follower.disconnect()
            cam.stop()


if __name__ == "__main__":
    main()
