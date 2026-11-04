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
    if len(allowed) == 1:                                    # nothing to decide: skip the 5-9 s Spark call
        return {"skill": "pick", "obj": allowed[0], "why": "only one target left"}, "rule (single)"
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


# ============================================================================ parallel two-arm mission
# Main thread: Tk window + DISPATCHER (look -> who is free -> give it a job), ~4 looks per second.
# Each arm runs its job in its own thread (separate serial buses, so they really move at the same time).
#   right arm: picks objects it can reach straight into the bin; after a drop it stays ABOVE THE BIN (scan pose) and
#              the camera looks from there; only when nothing is left inside the black lines for IDLE_REST_S does it
#              go to rest (so the camera sees the whole table) and keeps watching.
#   left arm : picks objects ONLY it can reach and puts them on the handoff spot next to the right arm.
# Shared area (both reach maps overlap): one arm at a time (ZONE lock).
# A red blob near a busy arm's gripper (an object being carried) is never taken as a new target.
import threading  # noqa: E402

IDLE_REST_S = 5.0
CARRY_IGNORE_PX = 80
_order_lock = threading.Lock()


def _tip_px(arm, q):
    t = kin.fk_xyz(q)
    return two_arm.arm_to_px(arm, t[0], t[1])


def main():
    import tkinter as tk
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    skill = json.load(open(HERE / "skill.json")); rule = skill["rule"]
    calib = perceive.load_calib()
    root = tk.Tk(); root.title("chand mission (two arms)")
    lab = tk.Label(root, font=("Sans", 14), width=80, height=9, justify="left"); lab.pack()
    st = {"pause": True, "quit": False}
    def key(e):
        if e.char == " ": st["pause"] = not st["pause"]
        if e.char and e.char.lower() == "q": st["quit"] = True; st["pause"] = False
    root.bind("<Key>", key)
    root.protocol("WM_DELETE_WINDOW", lambda: (st.update(quit=True, pause=False)))

    if "--no-viz" not in ARGS:
        viz.start()
    cam = cams.Cams.start(); viz.stream(cam)
    print("cameras on:", list(cam.cams), flush=True)
    fR = SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=ARM_ID, max_relative_target=5.0)); fR.connect(calibrate=False)
    execute.boost_gripper(fR)
    hold = lambda: st["pause"]                       # workers never touch Tk
    arms = {"right": {"mv": Mover(fR, hold, name="right")}}
    arms["right"]["rest"] = arms["right"]["mv"].q.copy()
    if TWO_ARM:
        try:
            fL = SOFollower(SOFollowerRobotConfig(port=two_arm.LEFT_PORT, id=two_arm.LEFT_ID, max_relative_target=5.0)); fL.connect(calibrate=False)
            execute.boost_gripper(fL)
            arms["left"] = {"mv": Mover(fL, hold, name="left")}; arms["left"]["rest"] = arms["left"]["mv"].q.copy()
        except Exception as e:
            print("left arm OFF:", e, flush=True)
    for a in arms.values():
        a.update(job=None, status="idle (rest)", at_rest=True, lock=threading.Lock())
    spot = two_arm.handoff_spot(skill) if "left" in arms else None
    mR, mL = two_arm.mask("right"), two_arm.mask("left")
    overlap = (mR > 0) & (mL > 0) if mR is not None and mL is not None else None
    ZONE = threading.Lock()
    scan_pose = None
    blacklist, history, log = [], [], []
    reach_cache = {}
    info = {"mined": 0, "handed": 0, "t0": None, "last_seen": time.time(), "brain": "", "order": []}

    def ui(extra=""):
        el = 0 if info["t0"] is None else time.time() - info["t0"]
        lines = [f"CHAND two-arm  {'PAUSED (SPACE)' if st['pause'] else 'running'}   mined {info['mined']}   handed over {info['handed']}   t {el:5.1f}s"]
        for n, a in arms.items():
            lines.append(f"{n.upper():5s}: {a['status']}")
        lines += [f"brain: {info['brain']}", extra, "SPACE pause   Q stop (arms go home)"]
        try:
            lab.configure(text="\n".join(lines)); root.update()
        except tk.TclError:
            st["quit"] = True

    def make_step(arm):
        def step(name):
            if st["quit"]:
                raise KeyboardInterrupt("stopped by user")
            arms[arm]["status"] = f"{arms[arm]['job_name']}: {name}"
            viz.text(f"{arm}: {name}")
        return step

    def reach(r):
        k = (r["u"] // 8, r["v"] // 8)
        if k not in reach_cache:
            if two_arm.can_pick("right", r["u"], r["v"], skill)[0]:
                reach_cache[k] = "right"
            elif "left" in arms and two_arm.can_pick("left", r["u"], r["v"], skill)[0]:
                reach_cache[k] = "left"
            else:
                reach_cache[k] = "none"
        return reach_cache[k]

    def near(r, pts_mm, mm=30):
        return any(np.hypot(r["x_mm"] - p[0], r["y_mm"] - p[1]) < mm for p in pts_mm)

    def record(arm, rock, result, res=None):
        history.append({"obj": rock["id"], "at": [rock["x_mm"], rock["y_mm"]], "result": result})
        if result != "mined" and not result.startswith("handed") and _fails(rock, history) >= 2:
            blacklist.append((rock["x_mm"], rock["y_mm"]))
        log.append({"arm": arm, "rock": {k: rock.get(k) for k in ("id", "shape", "x_mm", "y_mm", "u", "v")}, "result": result,
                    "grasps": (res or {}).get("grasps"), "at_drop": (res or {}).get("at_drop"),
                    "t": round(time.time() - info["t0"], 1)})
        print(f"  [{arm}] {rock['id']} {rock.get('shape')} ({rock['x_mm']},{rock['y_mm']}) -> {result}", flush=True)
        viz.text(f"{arm}: {result}", "INFO" if result in ("mined",) or result.startswith("handed") else "WARN")

    # ------------------------------------------------------------------ jobs (run in arm threads)
    def right_pick(rock, frame):
        nonlocal scan_pose
        mv = arms["right"]["mv"]; step = make_step("right")
        if JUDGE:
            try:
                p = judge.is_target(frame, rock, rule["rule_text"])
                if p < judge.SURE_NO:
                    blacklist.append((rock["x_mm"], rock["y_mm"])); record("right", rock, f"not a target (jev {p:.2f})"); return
            except Exception as e:
                print("  jev target check failed (ignored):", e, flush=True)
        rt, _ = execute.memory_get(rock.get("shape", "_other"))
        try:
            plan = execute.plan_pick(rock, skill, calib, shift=execute.rt_to_world(rt, np.arctan2(rock["y_mm"], rock["x_mm"])))
        except ValueError as e:
            blacklist.append((rock["x_mm"], rock["y_mm"])); record("right", rock, f"cannot plan: {e}"); return
        scan_pose = plan["above_bin"]
        in_zone = overlap is not None and overlap[min(479, max(0, rock["v"])), min(639, max(0, rock["u"]))]
        zone_held = {"yes": False}
        def step(name, _s=step):
            if name == "approach" and in_zone and not zone_held["yes"]:
                arms["right"]["status"] = "waiting for the shared area"
                while not ZONE.acquire(timeout=0.3):
                    if st["quit"]: raise KeyboardInterrupt
                zone_held["yes"] = True
            if name in ("carry", "home") and zone_held["yes"]:
                ZONE.release(); zone_held["yes"] = False
            _s(name)
        vis = execute.Vision(skill, calib, rock, lambda: cam.fresh("right_wrist"), lambda d: None) if JUDGE else None
        def push(plan_r):
            if "left" not in arms or not arms["left"]["lock"].acquire(blocking=False):
                return False                                 # left arm busy: no assist this time
            try:
                R2L, tzL = assist.frames()
                pp = assist.plan_push((rock["x_mm"], rock["y_mm"]), plan_r["tip"], arms["left"]["mv"].q, R2L, tzL)
                if pp["err"] > 6:
                    return False
                arms["left"]["status"] = "PUSHING object into the right gripper"
                assist.do_push(arms["left"]["mv"], pp, arms["left"]["rest"]); return True
            except Exception as e:
                print("  push assist failed:", e, flush=True); return False
            finally:
                arms["left"]["status"] = "idle (rest)"; arms["left"]["lock"].release()
        try:
            res = execute.pick_place(mv, plan, skill, step, vis=vis, assist=push, end_pose=scan_pose)
        finally:
            if zone_held["yes"]: ZONE.release()
        ok = bool(res.get("held")) and (res.get("at_drop") or {}).get("held", False)
        if res.get("reason") != "object lost":
            execute.memory_put(rock.get("shape", "_other"), res.get("shift_rt", rt), ok)
        info["mined"] += int(ok)
        record("right", rock, "mined" if ok else (res.get("reason") or ("missed grasp" if not res.get("held") else "dropped on the way")), res)
        viz.value("mission/mined", info["mined"])

    def left_handoff(rock, frame):
        mv = arms["left"]["mv"]; step = make_step("left")
        rt, _ = execute.memory_get(rock.get("shape", "_other"))
        xL, yL = two_arm.px_to_arm("left", rock["u"], rock["v"])
        vis = (execute.Vision(skill, two_arm.calib("left"), {"x_mm": xL, "y_mm": yL}, lambda: cam.fresh("left_wrist"), lambda d: None)
               if JUDGE and "left_wrist" in cam.cams else None)
        def spot_free():
            f = cam.fresh("top"); objs, _ = perceive.detect(f)
            return not any(np.hypot(o["u"] - spot[0], o["v"] - spot[1]) < 40 for o in objs)
        held_zone = {}
        try:
            res = two_arm.left_handoff(mv, (rock["u"], rock["v"]), skill, spot, step, vis=vis, shift_rt=rt,
                                       zone=ZONE, spot_free=spot_free, held_zone=held_zone)
        except ValueError as e:
            res = {"held": False, "reason": f"cannot plan: {e}"}
        finally:
            if held_zone.get("yes"): ZONE.release()
        if res.get("held"):
            info["handed"] += 1
        record("left", rock, "handed to the right arm" if res.get("held") else ("left: " + (res.get("reason") or "missed grasp")), res)

    def go_rest(arm):
        arms[arm]["mv"].move(arms[arm]["rest"]); arms[arm]["at_rest"] = True

    def run_job(arm, name, fn, *a):
        def wrap():
            with arms[arm]["lock"]:
                arms[arm]["job_name"] = name
                try:
                    fn(*a)
                except KeyboardInterrupt:
                    print(f"  [{arm}] stopped: opening gripper, going to rest", flush=True)
                    mv = arms[arm]["mv"]
                    try:
                        mv.gripper(skill["gripper"]["open"] + 8, wait=0.3)
                        if np.abs(mv.q - arms[arm]["rest"]).max() > 10:
                            up = mv.q.copy(); up[1] = max(up[1] - 15, arms[arm]["rest"][1]); mv.move(up)
                        mv.move(arms[arm]["rest"])
                    except Exception as e:
                        print(f"  [{arm}] abort move failed: {e}", flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [{arm}] job error: {e}", flush=True)
                finally:
                    arms[arm]["status"] = "idle (rest)" if arms[arm]["at_rest"] else "idle (scanning above the bin)" if arm == "right" else "idle"
                    arms[arm]["job"] = None
        if arm == "right" and name != "rest": arms[arm]["at_rest"] = False
        if arm == "left": arms[arm]["at_rest"] = name == "rest"
        t = threading.Thread(target=wrap, daemon=True); arms[arm]["job"] = t; t.start()

    def ask_spark(rocks):    # background: Spark orders the right arm's targets (dispatcher never waits for it)
        def go():
            try:
                d, src = plan_next(rule, rocks, list(history))
                info["brain"] = f"{src}: {d.get('why', '')} {d.get('plan', '')}"
            except Exception as e:
                info["brain"] = f"spark error: {e}"
        threading.Thread(target=go, daemon=True).start()

    # ------------------------------------------------------------------ dispatcher
    ui("SPACE to start")
    while st["pause"] and not st["quit"]:
        ui("SPACE to start"); time.sleep(0.05)
    info["t0"] = time.time(); n_targets_seen = 0
    try:
        while not st["quit"]:
            ui()
            if st["pause"]:
                time.sleep(0.05); continue
            frame = cam.fresh("top")
            rocks, _ = perceive.scene(frame, calib)
            busy_tips = [_tip_px(n, getattr(a["mv"], "q_cmd", a["mv"].q)) for n, a in arms.items() if a["job"] is not None]
            cand = []
            for r in rocks:
                if any(np.hypot(r["u"] - t[0], r["v"] - t[1]) < CARRY_IGNORE_PX for t in busy_tips):
                    continue                    # being carried / worked on right now
                if near(r, blacklist):
                    continue
                r["arm"] = reach(r)
                if r["arm"] != "none":
                    cand.append(r)
            if cand:
                info["last_seen"] = time.time()
            if len(cand) > n_targets_seen and BACKEND != "rule":
                ask_spark(cand)                 # new objects appeared: let Spark (re)order them in the background
            n_targets_seen = len(cand)
            right_t = [r for r in cand if r["arm"] == "right"]
            left_t = [r for r in cand if r["arm"] == "left"]
            if arms["right"]["job"] is None and right_t:
                at_spot = [r for r in right_t if spot is not None and np.hypot(r["u"] - spot[0], r["v"] - spot[1]) < 40]
                pick = at_spot[0] if at_spot else None
                if pick is None and _ORDER:
                    with _order_lock:
                        for pos in list(_ORDER):
                            pick = next((r for r in right_t if np.hypot(r["x_mm"] - pos[0], r["y_mm"] - pos[1]) < 30), None)
                            if pick: break
                pick = pick or min(right_t, key=lambda r: np.hypot(r["x_mm"], r["y_mm"]))
                run_job("right", f"pick {pick['id']} {pick.get('shape')}", right_pick, pick, frame)
            if "left" in arms and arms["left"]["job"] is None and left_t:
                pick = min(left_t, key=lambda r: np.hypot(*two_arm.px_to_arm("left", r["u"], r["v"])))
                run_job("left", f"hand over {pick['id']} {pick.get('shape')}", left_handoff, pick, frame)
            # nothing to do for a while: right arm leaves the bin and rests so the camera sees the whole table
            if (not cand and arms["right"]["job"] is None and not arms["right"]["at_rest"]
                    and time.time() - info["last_seen"] > IDLE_REST_S):
                arms["right"]["at_rest"] = True
                run_job("right", "rest", go_rest, "right")
            if not cand and all(a["job"] is None for a in arms.values()) and arms["right"]["at_rest"]:
                ui("table clear inside the black lines: watching for new objects ...")
            time.sleep(0.15)
    finally:
        st["quit"] = True
        for n, a in arms.items():
            if a["job"] is not None: a["job"].join(timeout=60)
        for n, a in arms.items():
            try:
                if np.abs(a["mv"].read() - a["rest"]).max() > 3:
                    st["pause"] = False; a["mv"].hold = None; a["mv"].move(a["rest"])
            except Exception as e:
                print(f"{n} final rest move failed: {e}", flush=True)
        json.dump(log, open(HERE / "logs" / time.strftime("mission_%H%M%S.json"), "w"), indent=1)
        print(f"MISSION END: mined {info['mined']}, handed over {info['handed']} in {time.time() - (info['t0'] or time.time()):.0f}s", flush=True)
        try: root.destroy()
        except Exception: pass
        for a in arms.values():
            try: a["mv"].f.disconnect()
            except Exception: pass
        cam.stop()


if __name__ == "__main__":
    main()
