"""Self-calibration of the top camera for one arm: the arm places a red cube at ~10 spots by itself.

  python selfcal.py --arm right|left [--n 10]

1. Take every other red object off the table. Put ONE red cube between the open jaws, press SPACE (it grips with
   force feedback).
2. For each spot (spread over the arm's painted workspace): put the cube down exactly where a pick would grasp it,
   open, back off out of the camera view, the top camera finds the cube (pixel), pick it up again from the same
   spot (also a pick test), next spot.
3. Fit pixel -> arm mm (affine, leave-one-out error) and write calib.json / calib_left.json (old file kept as
   *_before_selfcal.json). Same meaning as the touch calibration: the pixel of a resting cube <-> its top centre.
Keys: SPACE start / continue, Q stop (puts the cube down and goes to rest).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import tkinter as tk
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
import kin, perceive, execute, cams
from execute import Mover

HERE = Path(__file__).parent
ARM = "left" if "--arm" in sys.argv and sys.argv[sys.argv.index("--arm") + 1] == "left" else "right"
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10
PORT = {"right": "/dev/ttyFollowerLeft", "left": "/dev/ttyFollowerRight"}[ARM]
CALIB = HERE / {"right": "calib.json", "left": "calib_left.json"}[ARM]
R_MIN, R_MAX = 150.0, 305.0


def pick_spots(skill, calib, n):
    """Farthest-point sampling over the painted workspace (arm mm, camera-independent), only spots a grasp can plan,
    away from the bin."""
    pts = np.array([p[:2] for p in json.loads((HERE / f"workspace_{ARM}.json").read_text())["points_mm_xyh"]], float)
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[(r > R_MIN) & (r < R_MAX)]
    # keep spots that were inside the OLD camera view between the old tape lines = surely on the table
    oc = json.loads((HERE / {"right": "calib_oldcam.json", "left": "calib_left_oldcam.json"}[ARM]).read_text())
    ow = json.loads((HERE / "work_area_oldcam.json").read_text())
    A = np.vstack([np.asarray(oc["A_px_to_mm"], float), [0, 0, 1]]); Ai = np.linalg.inv(A)
    uv = (Ai @ np.c_[pts, np.ones(len(pts))].T).T[:, :2]
    (a, b0), (c, d) = ow["top_line"], ow["bottom_line"]
    on_table = (uv[:, 0] > 15) & (uv[:, 0] < 625) & (uv[:, 1] > a * uv[:, 0] + b0 + 25) & (uv[:, 1] < c * uv[:, 0] + d - 25)
    on_table &= ~((uv[:, 0] > 440) & (uv[:, 1] < 230))          # old bin corner
    pts = pts[on_table]
    if ARM == "right":
        b = kin.fk_xyz(execute.bin_drop_q(skill))[:2]
        pts = pts[np.hypot(pts[:, 0] - b[0], pts[:, 1] - b[1]) > 120]
    chosen = [pts[np.argmin(np.hypot(*(pts - pts.mean(0)).T))]]
    while len(chosen) < n * 3 and len(chosen) < len(pts):
        d = np.min([np.hypot(*(pts - c).T) for c in chosen], axis=0)
        chosen.append(pts[np.argmax(d)])
    ok = []
    for c in chosen:
        try:
            execute.plan_pick({"x_mm": float(c[0]), "y_mm": float(c[1])}, skill, calib)
            ok.append(c)
        except ValueError:
            pass
        if len(ok) == n:
            break
    return ok


def find_cube(img):
    """Largest red blob outside the bin box, anywhere in the image (the calibration cube is the only one)."""
    m = perceive.red_mask(img)
    import cv2
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x0, y0, x1, y1 = perceive.bin_roi()
    best = None
    for c in cs:
        a = cv2.contourArea(c)
        if a < perceive.MIN_AREA:
            continue
        (cx, cy), _, _ = cv2.minAreaRect(c)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            continue
        if best is None or a > best[0]:
            best = (a, cx, cy)
    return None if best is None else (round(best[1], 1), round(best[2], 1))


def fit_affine(px, mm):
    A = np.linalg.lstsq(np.c_[px, np.ones(len(px))], mm, rcond=None)[0].T
    err = np.linalg.norm(np.c_[px, np.ones(len(px))] @ A.T - mm, axis=1)
    loo = []
    for i in range(len(px)):
        k = np.arange(len(px)) != i
        Ai = np.linalg.lstsq(np.c_[px[k], np.ones(k.sum())], mm[k], rcond=None)[0].T
        loo.append(float(np.linalg.norm(np.r_[px[i], 1] @ Ai.T - mm[i])))
    return A, err, np.array(loo)


def main():
    skill = json.loads((HERE / "skill.json").read_text())
    calib = json.loads(CALIB.read_text())                  # only rock_top_z_mm is used for planning (camera-independent)
    g = skill["gripper"]; g_open = g["open"] + 8
    spots = pick_spots(skill, calib, N)
    print(f"{ARM}: {len(spots)} spots (arm mm): {np.round(spots).tolist()}", flush=True)
    cam = cams.Cams.start(("top",))
    f = SOFollower(SOFollowerRobotConfig(port=PORT, id=f"thumbi_{ARM}", max_relative_target=5.0))
    f.connect(calibrate=False); execute.boost_gripper(f)
    root = tk.Tk(); root.title(f"chand self-calibration: {ARM.upper()} arm")
    lab = tk.Label(root, font=("Sans", 15), width=70, height=6, justify="left"); lab.pack()
    st = {"go": False, "quit": False}
    def key(e):
        if e.char == " ": st["go"] = True
        if e.char and e.char.lower() == "q": st["quit"] = True
    root.bind("<Key>", key); root.protocol("WM_DELETE_WINDOW", lambda: st.update(quit=True))
    def say(t):
        lab.configure(text=t); root.update()
    def wait_space(t):
        st["go"] = False
        while not st["go"]:
            if st["quit"]: raise KeyboardInterrupt
            say(t); time.sleep(0.05)
    def hold():
        root.update()
        if st["quit"]: raise KeyboardInterrupt
        return False
    mv = Mover(f, hold, name=ARM); rest = mv.q.copy()
    points = []
    try:
        mv.gripper(g_open, wait=0.5)
        while True:
            wait_space(f"{ARM.upper()} arm: take every other red object off the table.\n"
                       f"Put ONE red cube between the open jaws (middle of the fingers), then press SPACE.")
            ff = mv.grasp_ff(execute.GRIP_CLOSED)
            if ff["held"]:
                break
            mv.gripper(g_open, wait=0.4)
        hand_gripped = True                     # the first grip is by hand: that placement is not exact
        for i, c in enumerate(spots):
            plan = execute.plan_pick({"x_mm": float(c[0]), "y_mm": float(c[1])}, skill, calib)
            say(f"spot {i + 1}/{len(spots)}: putting the cube down at {np.round(c).tolist()} mm")
            mv.move(plan["q_lift"])
            mv.follow(plan["lift"][::-1][1:], peak_deg_s=20)            # straight down into the grasp pose
            mv.gripper(g_open, wait=0.4)
            mv.follow(plan["descend"][::-1][1:], peak_deg_s=25)         # back off along the approach line
            mv.move(rest)                                               # out of the camera view
            time.sleep(0.3)
            px = find_cube(cam.fresh("top", settle_frames=4))
            say(f"spot {i + 1}/{len(spots)}: camera sees the cube at px {px}")
            if px is not None:
                points.append({"px": list(px), "xyz_mm": [round(float(c[0]), 1), round(float(c[1]), 1), calib["rock_top_z_mm"]],
                               "hand": bool(hand_gripped)})
                json.dump(points, open(HERE / f"selfcal_points_{ARM}.json", "w"), indent=1)
                print(f"  spot {i + 1}: mm {np.round(c, 1).tolist()} <- px {px}", flush=True)
            # pick it up again from where we put it (no camera needed: we know where it is)
            mv.move(plan["q_pre"], grip=g_open)
            mv.follow(plan["descend"][1:], grip=g_open, peak_deg_s=25); mv.move(plan["q_grasp"], grip=g_open, settle_s=1.0)
            ff = mv.grasp_ff(execute.GRIP_CLOSED)
            hand_gripped = not ff["held"]
            print(f"    re-pick: {'OK' if ff['held'] else 'MISSED'} (contact {ff['contact']})", flush=True)
            mv.follow(plan["lift"][1:], peak_deg_s=25)
            if not ff["held"]:
                mv.gripper(g_open, wait=0.3)
                wait_space(f"re-pick missed at spot {i + 1}. Put the cube back between the open jaws, press SPACE.")
                ff = mv.grasp_ff(execute.GRIP_CLOSED)
        # end: put the cube down at the first spot, go to rest
        plan = execute.plan_pick({"x_mm": float(spots[0][0]), "y_mm": float(spots[0][1])}, skill, calib)
        mv.move(plan["q_lift"]); mv.follow(plan["lift"][::-1][1:], peak_deg_s=20); mv.gripper(g_open, wait=0.4)
        mv.follow(plan["descend"][::-1][1:], peak_deg_s=25)
    except KeyboardInterrupt:
        print("stopped by user", flush=True)
        try: mv.gripper(g_open, wait=0.3)
        except Exception: pass
    finally:
        try:
            mv.hold = None; mv.move(rest)
        except Exception as e:
            print("rest move failed:", e)
        try: root.destroy()
        except Exception: pass
        f.disconnect(); cam.stop()
    if len(points) >= 5:
        use = [p for p in points if not p.get("hand")] or points       # hand-gripped placements are not exact
        px = np.array([p["px"] for p in use], float); mm = np.array([p["xyz_mm"][:2] for p in use], float)
        A, err, loo = fit_affine(px, mm)
        old = CALIB.with_name(CALIB.stem + "_before_selfcal.json"); old.write_text(CALIB.read_text())
        calib.update({"A_px_to_mm": A.tolist(), "n": len(points), "fit_err_mm": err.round(1).tolist(), "loo_err_mm": loo.round(1).tolist(),
                      "method": "selfcal (arm places cube, camera finds it)", "t": time.strftime("%Y-%m-%d %H:%M:%S")})
        CALIB.write_text(json.dumps(calib, indent=1))
        print(f"WROTE {CALIB.name}: n={len(points)} fit mean {err.mean():.1f} mm, leave-one-out mean {loo.mean():.1f} mm (max {loo.max():.1f})", flush=True)
    else:
        print(f"only {len(points)} points: calibration NOT written", flush=True)


if __name__ == "__main__":
    main()
