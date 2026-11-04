"""M4 teach-once. Record ONE leader demo (pick a red rock -> drop in the red bin -> back to rest),
then extract the skill and ask Muse Spark for the rule.

  python teach.py              record (window: SPACE teleop on/off, R start/stop recording, Q quit)
  python teach.py --process    re-process the last recording (no robot)
  add --dry to print the Spark prompt instead of sending it

Start AND end the recording with the arm at rest, out of the top cam view (the first and last frames
are the before/after scene).
"""
import json, sys, time, base64
from pathlib import Path
import cv2, numpy as np
import kin, perceive, brain

HERE = Path(__file__).parent
REC = HERE / "recordings"
J = [f"{j}.pos" for j in kin.JOINTS]
FOLLOWER_PORT, LEADER_PORT, ARM_ID = "/dev/ttyFollowerLeft", "/dev/ttyLeaderRight", "thumbi_right"
WRIST_CAM = "/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.3:1.0-video-index0"


def _cam(path):
    c = cv2.VideoCapture(path, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); c.set(3, 640); c.set(4, 480)
    return c


def record():
    import tkinter as tk
    from PIL import Image, ImageTk
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
    follower = SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=ARM_ID, max_relative_target=5.0))
    leader = SOLeader(SOLeaderTeleopConfig(port=LEADER_PORT, id=ARM_ID))
    follower.connect(calibrate=False); leader.connect(calibrate=False)
    top, wrist = _cam(perceive.TOP_CAM), _cam(WRIST_CAM)
    root = tk.Tk(); root.title("chand teach"); panel = tk.Label(root); panel.pack()
    keys = []
    root.bind("<Key>", lambda e: keys.append(e.char.lower() if e.char else ""))
    root.protocol("WM_DELETE_WINDOW", lambda: keys.append("q"))
    teleop, rec, rows, frames, wframes = False, False, [], [], []
    msg = "PAUSED: match leader to follower, SPACE. Arm at rest, then R to start recording"
    REC.mkdir(exist_ok=True)
    out = REC / time.strftime("demo_%H%M%S")
    t0 = None
    try:
        while True:
            act = leader.get_action()
            if teleop:
                follower.send_action(act)
            ok, f = top.read()
            ok2, w = wrist.read()
            obs = follower.get_observation()
            if rec:
                rows.append([time.time() - t0] + [obs[k] for k in J] + [obs["gripper.pos"], act["gripper.pos"]])
                frames.append(f); wframes.append(w if ok2 else None)
            view = f.copy()
            cv2.putText(view, f"{'REC %.1fs' % (time.time() - t0) if rec else 'not recording'} | {'teleop' if teleop else 'PAUSED'}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if rec else (0, 128, 0), 2)
            cv2.putText(view, msg, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
            panel.configure(image=img); panel.image = img; root.update()
            c = keys.pop(0) if keys else ""
            if c == " ":
                teleop = not teleop; msg = "teleop ON" if teleop else "PAUSED (follower holds)"
            elif c == "r":
                if not rec:
                    rec, t0, rows, frames, wframes = True, time.time(), [], [], []
                    msg = "RECORDING: pick ONE red rock, drop it in the bin, back to rest, then R"
                else:
                    rec = False
                    out.mkdir(parents=True, exist_ok=True)
                    np.save(out / "joints.npy", np.array(rows))
                    cv2.imwrite(str(out / "top_first.jpg"), frames[0]); cv2.imwrite(str(out / "top_last.jpg"), frames[-1])
                    vw = cv2.VideoWriter(str(out / "top.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
                    for fr in frames: vw.write(fr)
                    vw.release()
                    vw = cv2.VideoWriter(str(out / "wrist.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
                    for fr in wframes:
                        if fr is not None: vw.write(fr)
                    vw.release()
                    msg = f"saved {len(rows)} steps ({rows[-1][0]:.1f}s) to {out.name}. Q to quit, R to redo"
                    print(msg, flush=True)
            elif c == "q":
                break
    finally:
        top.release(); wrist.release()
        try: root.destroy()
        except Exception: pass
        follower.disconnect(); leader.disconnect()
    return out


def segment(D, gap=4.0, min_s=0.5):
    """Holding = the rock blocks the gripper: measured stays > commanded by `gap`. The longest such
    stretch is the carry. Works whether the gripper starts open or closed.
    D columns: t, 5 joints, gripper(meas), gripper(cmd)."""
    t, g, gc = D[:, 0], D[:, 6], D[:, 7]
    hold = (g - gc) > gap
    best, i = (0, 0), 0
    while i < len(hold):
        if hold[i]:
            j = i
            while j < len(hold) and hold[j]:
                j += 1
            if t[j - 1] - t[i] > t[best[1] - 1] - t[best[0]] if best[1] else True:
                best = (i, j)
            i = j
        else:
            i += 1
    i_closed, i_release = best
    if not i_release or t[i_release - 1] - t[i_closed] < min_s:
        raise RuntimeError("no holding stretch found (gripper never blocked by a rock)")
    i_release -= 1
    i_close = i_closed                    # walk back to where the closing began (gripper still moving down)
    while i_close > 0 and g[i_close - 1] > g[i_close] + 0.05:
        i_close -= 1
    pre = slice(max(0, i_close - 30), i_close)
    g_open = float(np.median(g[pre]))     # opening used just before the grasp
    return dict(g_open=g_open, g_min=float(g[i_closed:i_release].min()), i_close=i_close, i_closed=i_closed,
                i_release=i_release, i_open=i_release)


def process(out, dry=False):
    D = np.load(out / "joints.npy"); t = D[:, 0]; Q = D[:, 1:6]
    seg = segment(D)
    calib = perceive.load_calib()
    before = cv2.imread(str(out / "top_first.jpg")); after = cv2.imread(str(out / "top_last.jpg"))
    rocks_before, bin_mm = perceive.scene(before, calib)
    rocks_after, _ = perceive.scene(after, calib)
    q_grasp = Q[seg["i_close"]]
    tip = kin.fk(q_grasp)
    picked = min(rocks_before, key=lambda r: np.hypot(r["x_mm"] - tip[0, 3], r["y_mm"] - tip[1, 3]))
    off = [float(tip[0, 3] - picked["x_mm"]), float(tip[1, 3] - picked["y_mm"]), float(tip[2, 3] - calib["rock_top_z_mm"])]
    i_pre = int(np.searchsorted(t, t[seg["i_close"]] - 1.0))    # 1 s before closing = approach pose
    drop_tip = kin.fk_xyz(Q[seg["i_release"]])
    # carry waypoints at 5 Hz from 'closed' to release
    idx = np.arange(seg["i_closed"], seg["i_release"] + 1, 6)
    grip_hold = float(np.median(D[seg["i_closed"]:seg["i_release"], 6]))
    skill = {
        "demo": out.name, "duration_s": round(float(t[-1]), 1),
        "picked": {k: picked[k] for k in ("colour", "x_mm", "y_mm", "angle_deg")},
        "grasp": {"q_deg": q_grasp.round(2).tolist(), "tip_mm": tip[:3, 3].round(1).tolist(),
                  "tool_z": tip[:3, 2].round(3).tolist(), "offset_from_rock_top_mm": np.round(off, 1).tolist(),
                  "wrist_roll_minus_rock_angle": round(float(q_grasp[4] - picked["angle_deg"]), 1)},
        "approach_q_deg": Q[i_pre].round(2).tolist(),
        "gripper": {"open": round(seg["g_open"], 1), "closed_on_rock": round(grip_hold, 1),
                    "close_cmd": round(grip_hold - 3, 1)},   # just past the object, not 0 (overload lesson)
        "carry_q_deg": Q[idx].round(2).tolist(),
        "drop_q_deg": Q[seg["i_release"]].round(2).tolist(), "drop_tip_mm": drop_tip.round(1).tolist(),
        "rest_q_deg": Q[-1].round(2).tolist(),
        "segments_s": {k: round(float(t[v]), 2) for k, v in seg.items() if k.startswith("i_")},
        "scene_before": rocks_before, "scene_after": rocks_after, "bin_mm": bin_mm,
    }
    print(json.dumps({k: v for k, v in skill.items() if k not in ("carry_q_deg",)}, indent=1))
    skill["rule"] = ask_rule(skill, before, after, dry)
    json.dump(skill, open(HERE / "skill.json", "w"), indent=1)
    json.dump(skill, open(out / "skill.json", "w"), indent=1)
    print("rule:", json.dumps(skill["rule"], indent=1))
    return skill


def ask_rule(skill, before, after, dry):
    summary = {
        "table_before": [{"id": r["id"], "colour": r["colour"], "x_mm": r["x_mm"], "y_mm": r["y_mm"]} for r in skill["scene_before"]],
        "note": "White objects are also on the table (the colour detector only lists red ones). The red bin is the container.",
        "operator_picked": skill["picked"],
        "released_at_mm": skill["drop_tip_mm"], "red_bin_centre_mm": skill["bin_mm"],
        "red_left_on_table_after": len(skill["scene_after"]),
    }
    sys_p = ("You turn ONE human demonstration of a robot arm into a general sorting rule for a rover that will "
             "repeat the task alone. Reply with ONLY a JSON object: "
             '{"pick": {"colour": "<colour>"}, "place_in": "<container id>", "leave_alone": ["<colour>", ...], '
             '"stop_when": "<condition>", "rule_text": "<one sentence>"}')
    text = [{"type": "text", "text": "Demonstration summary (arm base frame, mm):\n" + json.dumps(summary, indent=1) +
             "\nThe images are the top camera before and after the demo."}]
    content = text
    for im in (before, after):
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])
        content = content + [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}}]
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": content}]
    if dry:
        brain.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": text[0]["text"] + "\n[+2 images]"}], dry=True)
        return {"pick": {"colour": "red"}, "place_in": "red_bin", "leave_alone": ["white"], "stop_when": "no red left", "rule_text": "(dry run)"}
    try:
        reply = brain.chat(msgs, tag="teach_rule", reasoning_effort="low")
    except Exception as e:  # images not accepted -> text only
        print("image call failed:", e, "-> text only")
        reply = brain.chat([msgs[0], {"role": "user", "content": text[0]["text"]}], tag="teach_rule_text", reasoning_effort="low")
    rule = brain.parse_json(reply)
    rule["source"] = "muse-spark"
    return rule


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    if "--process" in sys.argv:
        out = sorted(REC.glob("demo_*"))[-1]
    else:
        out = record()
    if (out / "joints.npy").exists():
        process(out, dry)
