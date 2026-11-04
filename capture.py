"""Capture zone: where an object must sit in an arm's WRIST camera image so that closing the jaws surely grabs it,
plus the visual servo that moves the gripper until the object is there.

  python capture.py --arm right|left     # TEACH the zone (needs one red cube; ~1 min, automatic)

Teach: you put a cube between the open jaws, the arm grips it, sets it down at 3 spots and at each spot opens the
jaws WITHOUT moving and photographs it: that is exactly the "sure capture" picture. The red blob's centre/bottom
edge/size in those photos (averaged) is saved to capture_<arm>.json.

Servo (used by execute.pick_place before closing):
  local red-blob tracking in the wrist image (~10 Hz, no cloud) -> error to the zone -> PI controller -> small
  smooth tip moves (radial = deeper/shallower, tangential = sideways) through the arm's wrist-image jacobian ->
  until the object is inside the zone for 2 frames -> Jev referee on the frame with the zone drawn as a green box:
  "is the red object inside the green box?" -> close (force feedback decides the rest).
"""
import json, sys, time
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).parent
W, H = 640, 480


def zone(arm):
    f = HERE / f"capture_{arm}.json"
    return json.loads(f.read_text()) if f.exists() else None


def red_blobs(img, min_area=600):
    """Red blobs in a wrist image: centre, bottom edge, box, area (HSV like perceive.red_mask)."""
    import perceive
    m = perceive.red_mask(img)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        out.append({"cx": M["m10"] / M["m00"], "cy": M["m01"] / M["m00"], "top": float(y), "bottom": float(y + h),
                    "box": [x, y, x + w, y + h], "area": a})
    return out


def track(img, z):
    """The blob the gripper is going for: closest to the zone centre (ignores the bin / far objects)."""
    bl = red_blobs(img)
    if not bl:
        return None
    return min(bl, key=lambda b: np.hypot(b["cx"] - z["cx"], b["top"] - z["top"]))


def error(b, z):
    """Normalised image error (x, y) of blob b to the zone. y uses the object's TOP edge: it tracks depth and stays
    visible on both wrist cameras (on the left camera a grasped object's bottom edge is cut off by the image border)."""
    return np.array([(b["cx"] - z["cx"]) / W, (b["top"] - z["top"]) / H])


def inside(b, z):
    e = error(b, z)
    return abs(e[0]) * W <= z["tol_x"] and abs(e[1]) * H <= z["tol_y"]


def draw(img, z, b=None):
    v = img.copy()
    x0, x1 = int(z["cx"] - z["tol_x"]), int(z["cx"] + z["tol_x"])
    y0, y1 = int(z["top"] - z["tol_y"]), int(z["top"] + z["tol_y"])
    cv2.rectangle(v, (x0, y0), (x1, y1), (0, 220, 0), 3)                 # where the object's top-centre must be
    cv2.rectangle(v, tuple(map(int, z["box"][:2])), tuple(map(int, z["box"][2:])), (0, 160, 0), 1)   # taught object box
    if b is not None:
        cv2.circle(v, (int(b["cx"]), int(b["top"])), 8, (255, 0, 0), -1)
    return v


# ---------------------------------------------------------------------------------------------- teach
def teach(arm):
    import tkinter as tk
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    import execute, cams, selfcal
    from execute import Mover
    skill = json.loads((HERE / "skill.json").read_text())
    cal = json.loads((HERE / {"right": "calib.json", "left": "calib_left.json"}[arm]).read_text())
    g = skill["gripper"]; g_open = g["open"] + 8
    port = {"right": "/dev/ttyFollowerLeft", "left": "/dev/ttyFollowerRight"}[arm]
    cam = cams.Cams.start((f"{arm}_wrist",))
    f = SOFollower(SOFollowerRobotConfig(port=port, id=f"thumbi_{arm}", max_relative_target=5.0)); f.connect(calibrate=False)
    execute.boost_gripper(f)
    root = tk.Tk(); root.title(f"chand: teach capture zone ({arm.upper()} arm)")
    lab = tk.Label(root, font=("Sans", 15), width=70, height=5, justify="left"); lab.pack()
    st = {"go": False, "quit": False}
    root.bind("<Key>", lambda e: st.update(go=e.char == " " or st["go"], quit=(e.char or "").lower() == "q" or st["quit"]))
    root.protocol("WM_DELETE_WINDOW", lambda: st.update(quit=True))
    def hold():
        root.update()
        if st["quit"]: raise KeyboardInterrupt
        return False
    mv = Mover(f, hold, name=arm); rest = mv.q.copy()
    selfcal.ARM = arm
    spots = selfcal.pick_spots(skill, cal, 3)
    shots = []
    try:
        mv.gripper(g_open, wait=0.4)
        while True:
            st["go"] = False
            while not st["go"]:
                lab.configure(text=f"{arm.upper()} arm: put ONE red cube between the open jaws, press SPACE"); root.update(); time.sleep(0.05)
                if st["quit"]: raise KeyboardInterrupt
            if mv.grasp_ff(execute.GRIP_CLOSED)["held"]:
                break
            mv.gripper(g_open, wait=0.4)
        for i, c in enumerate(spots):
            plan = execute.plan_pick({"x_mm": float(c[0]), "y_mm": float(c[1])}, skill, cal)
            lab.configure(text=f"spot {i + 1}/{len(spots)}: setting the cube down"); root.update()
            mv.move(plan["q_lift"]); mv.follow(plan["lift"][::-1][1:], peak_deg_s=20)
            mv.gripper(g_open, wait=0.6)                                   # jaws open, arm NOT moving
            time.sleep(0.3)
            img = cam.fresh(f"{arm}_wrist", settle_frames=3)
            cv2.imwrite(str(HERE / "snapshots" / f"capture_{arm}_{i}.jpg"), img)
            bl = red_blobs(img)
            if bl:
                b = max(bl, key=lambda b: b["area"]); shots.append(b)
                print(f"  spot {i + 1}: blob centre ({b['cx']:.0f}, {b['cy']:.0f}) top {b['top']:.0f} box {b['box']}", flush=True)
            ff = mv.grasp_ff(execute.GRIP_CLOSED)                         # grab it again right here
            mv.follow(plan["lift"][1:], peak_deg_s=25)
            if not ff["held"]:
                print("  re-grip missed: stopping", flush=True); break
        mv.move(plan["q_lift"]); mv.follow(plan["lift"][::-1][1:], peak_deg_s=20); mv.gripper(g_open, wait=0.4)
        mv.follow(plan["descend"][::-1][1:], peak_deg_s=25)
    except KeyboardInterrupt:
        try: mv.gripper(g_open, wait=0.3)
        except Exception: pass
    finally:
        try: mv.hold = None; mv.move(rest)
        except Exception as e: print("rest move failed", e)
        try: root.destroy()
        except Exception: pass
        f.disconnect(); cam.stop()
    if shots:
        cx = float(np.mean([s["cx"] for s in shots])); top = float(np.mean([s["top"] for s in shots]))
        box = np.mean([s["box"] for s in shots], axis=0).round().tolist()
        w = box[2] - box[0]
        z = {"arm": arm, "cx": round(cx, 1), "top": round(top, 1), "box": box, "n": len(shots),
             "tol_x": round(max(18.0, 0.18 * w), 1), "tol_y": round(max(18.0, 0.15 * w), 1),
             "spread_px": [round(float(np.std([s["cx"] for s in shots])), 1), round(float(np.std([s["top"] for s in shots])), 1)],
             "t": time.strftime("%Y-%m-%d %H:%M:%S")}
        (HERE / f"capture_{arm}.json").write_text(json.dumps(z, indent=1))
        cv2.imwrite(str(HERE / "snapshots" / f"capture_{arm}_zone.jpg"), draw(cv2.imread(str(HERE / "snapshots" / f"capture_{arm}_0.jpg")), z))
        print(f"WROTE capture_{arm}.json: {z}", flush=True)


if __name__ == "__main__":
    teach("left" if "--arm" in sys.argv and sys.argv[sys.argv.index("--arm") + 1] == "left" else "right")
