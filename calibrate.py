"""Rock-touch calibration: top-cam pixels -> arm-base mm (affine fit, valid for ~30 mm tall rocks).
Teleop: right leader drives the right follower (max_relative_target 10).
Per point:
  1. Move the arm OUT of the top cam view, put the single red rock somewhere.   press S (snap rock pixel)
  2. With the leader, bring the gripper tip to just touch the TOP CENTRE of the rock.  press R (record joints)
Keys in the window: S snap, R record, U undo last point, F fit now, Q save+quit, SPACE e-stop (follower holds, teleop off;
press SPACE again to resume).
Writes calib_points.json after every point and calib.json on fit.
"""
import json, time, sys
import cv2, numpy as np
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
import kin, perceive
from execute import Mover
import tkinter as tk
from PIL import Image, ImageTk

FOLLOWER_PORT, LEADER_PORT, ARM_ID = "/dev/ttyFollowerLeft", "/dev/ttyLeaderRight", "thumbi_right"
J = [f"{j}.pos" for j in kin.JOINTS]
PTS_FILE, CALIB_FILE = "calib_points.json", "calib.json"


def _affine(px, mm):  # 2x3, px -> mm. Beat the homography on leave-one-out (7.4 vs 8.6 mm, 11 pts)
    return np.linalg.lstsq(np.c_[px, np.ones(len(px))], mm, rcond=None)[0].T


def fit(points):
    px = np.float64([p["px"] for p in points]); mm = np.float64([p["xyz_mm"][:2] for p in points])
    A = _affine(px, mm)
    err = np.linalg.norm(np.c_[px, np.ones(len(px))] @ A.T - mm, axis=1)
    loo = []  # leave-one-out: honest error on a point the fit did not see
    if len(points) >= 5:
        for i in range(len(points)):
            m = np.arange(len(points)) != i
            loo.append(float(np.linalg.norm(np.r_[px[i], 1] @ _affine(px[m], mm[m]).T - mm[i])))
    return A, err, loo


def main():
    points = []
    try:
        points = json.load(open(PTS_FILE))
        print(f"resuming with {len(points)} saved points")
    except Exception:
        pass
    follower = SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=ARM_ID, max_relative_target=5.0))
    leader = SOLeader(SOLeaderTeleopConfig(port=LEADER_PORT, id=ARM_ID))
    follower.connect(calibrate=False); leader.connect(calibrate=False)
    cam = cv2.VideoCapture(perceive.TOP_CAM, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cam.set(3, 640); cam.set(4, 480)
    root = tk.Tk(); root.title("chand calibrate"); panel = tk.Label(root); panel.pack()
    keys = []
    root.bind("<Key>", lambda e: keys.append(e.char.lower() if e.char else ""))
    root.protocol("WM_DELETE_WINDOW", lambda: keys.append("q"))
    clicks = []
    panel.bind("<Button-1>", lambda e: clicks.append((e.x, e.y)))
    snap, estop, msg = None, True, "PAUSED: match leader to follower pose, then SPACE"
    try:
        while True:
            if not estop:
                follower.send_action(leader.get_action())
            ok, frame = cam.read()
            if not ok:
                continue
            obs = follower.get_observation()
            q = [obs[k] for k in J]
            tip = kin.fk_xyz(q)
            view = frame.copy()
            if snap is not None:
                cv2.drawMarker(view, tuple(map(int, snap)), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
            for p in points:
                cv2.circle(view, tuple(map(int, p["px"])), 4, (255, 0, 255), -1)
            cv2.putText(view, f"pts {len(points)} | tip mm {tip.round(0)} | {'E-STOP' if estop else 'teleop'}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if estop else (0, 128, 0), 2)
            cv2.putText(view, msg, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
            panel.configure(image=img); panel.image = img
            root.update()
            if clicks:  # click a rock = snap that rock (arm must be out of view)
                cx, cy = clicks.pop(0)
                objs, _ = perceive.detect(frame)
                if objs:
                    o = min(objs, key=lambda o: (o["u"] - cx) ** 2 + (o["v"] - cy) ** 2)
                    if (o["u"] - cx) ** 2 + (o["v"] - cy) ** 2 < 40 ** 2:
                        snap = (o["u"], o["v"]); msg = f"rock at px {snap}. Now touch its top centre, press R"
                        cv2.imwrite(f"snapshots/calib_{len(points)}.jpg", frame)
                    else:
                        msg = "no red rock under the click"
            c = keys.pop(0) if keys else ""
            k = ord(c) if c else 255
            if k == ord(" "):
                estop = not estop; msg = "E-STOP: follower holding. SPACE to resume" if estop else "resumed"
            elif k == ord("s"):
                objs, _ = perceive.detect(frame)
                if len(objs) == 1:
                    snap = (objs[0]["u"], objs[0]["v"]); msg = f"rock at px {snap}. Now touch its top centre, press R"
                    cv2.imwrite(f"snapshots/calib_{len(points)}.jpg", frame)
                else:
                    msg = f"need exactly 1 red rock in view, found {len(objs)} (arm in view? other rocks?)"
            elif k == ord("r"):
                if snap is None:
                    msg = "press S first (rock pixel)"
                else:
                    points.append({"px": list(snap), "q_deg": [round(float(x), 2) for x in q],
                                   "xyz_mm": [round(float(x), 1) for x in tip], "t": time.strftime("%H:%M:%S")})
                    json.dump(points, open(PTS_FILE, "w"), indent=1)
                    print(f"point {len(points)}: px {snap} -> mm {tip.round(1)}  q {np.round(q, 1)}", flush=True)
                    snap = None; msg = f"saved {len(points)}. Move arm away, move rock, press S"
            elif k == ord("u") and points:
                points.pop(); json.dump(points, open(PTS_FILE, "w"), indent=1); msg = f"undone, {len(points)} left"
            elif k in (ord("f"), ord("q")):
                if len(points) >= 4:
                    H, err, loo = fit(points)
                    res = {"A_px_to_mm": H.tolist(), "rock_height_mm": 30.0, "n": len(points), "fit_err_mm": [round(float(e), 1) for e in err],
                           "loo_err_mm": [round(e, 1) for e in loo], "rock_top_z_mm": float(np.mean([p["xyz_mm"][2] for p in points]))}
                    json.dump(res, open(CALIB_FILE, "w"), indent=1)
                    s = f"fit n={len(points)} mean err {err.mean():.1f} mm" + (f", leave-one-out {np.mean(loo):.1f} mm (max {max(loo):.1f})" if loo else "")
                    print(s, flush=True); msg = s
                else:
                    msg = "need >= 4 points to fit"
                if k == ord("q"):
                    break
    finally:
        cam.release()
        try:
            root.destroy()
        except Exception:
            pass
        follower.disconnect(); leader.disconnect()  # disconnect turns follower torque off: keep a hand under it


HOVER_MM, ABOVE_MM = 35.0, 80.0


def plan_hover(x, y, calib, q_seed=None):
    """Joint targets for 'ABOVE_MM above' and 'HOVER_MM above' the rock top. Seeded from the nearest
    hand-taught calibration pose (gripper ~45 deg tilt, a pose the real arm has really been in), so the
    DLS solution stays close to it. Forcing straight-down picked untested elbow/wrist branches."""
    pts = json.load(open(PTS_FILE))
    near = min(pts, key=lambda p: np.hypot(p["xyz_mm"][0] - x, p["xyz_mm"][1] - y))
    seed = np.array(near["q_deg"], float)
    zt = calib["rock_top_z_mm"]
    out = []
    for dz in (HOVER_MM, ABOVE_MM):
        q, e = kin.ik([x, y, zt + dz], seed, down=False)
        out.append((q, e)); seed = q
    return out[::-1]  # (above, hover)


def touch_test(dry=False):
    """M3: hover HOVER_MM above every red rock, one by one. Saves a wrist-cam shot at each hover."""
    calib = perceive.load_calib()
    frame = perceive.grab()
    rocks, _ = perceive.scene(frame, calib)
    cv2.imwrite("snapshots/m3_scene.jpg", perceive.draw(frame, *perceive.detect(frame)))
    seed = [0, 0, 60, 60, -85]
    plans = []
    for r in rocks:
        (qa, ea), (qh, eh) = plan_hover(r["x_mm"], r["y_mm"], calib, seed)
        print(f'{r["id"]} px({r["u"]},{r["v"]}) -> mm({r["x_mm"]},{r["y_mm"]})  ik err above {ea:.1f} hover {eh:.1f} mm  q_hover {qh.round(1)}', flush=True)
        if eh > 5 or ea > 15:
            print("   SKIP: not reachable", flush=True); continue
        plans.append((r, qa, qh))
    if dry or not plans:
        return
    follower = SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=ARM_ID, max_relative_target=5.0))
    follower.connect(calibrate=False)
    wrist = cv2.VideoCapture("/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.3:1.0-video-index0", cv2.CAP_V4L2)
    wrist.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); wrist.set(3, 640); wrist.set(4, 480)
    root = tk.Tk(); root.title("chand M3 touch test  |  SPACE pause/resume   Q abort (goes home)")
    lab = tk.Label(root, font=("Sans", 16), width=60, height=4); lab.pack()
    state = {"pause": True, "quit": False}
    def key(e):
        if e.char == " ": state["pause"] = not state["pause"]
        if e.char and e.char.lower() == "q": state["quit"] = True; state["pause"] = False
    root.bind("<Key>", key)
    def hold():
        lab.configure(text=("PAUSED - SPACE to go" if state["pause"] else "moving")); root.update()
        return state["pause"]
    mv = Mover(follower, hold)
    home = mv.q.copy()
    print("home q", home.round(1), flush=True)
    try:
        for r, qa, qh in plans:
            if state["quit"]: break
            mv.move(qa)
            mv.move(qh, settle_s=1.5)
            time.sleep(0.5)
            for _ in range(5): wrist.read()
            ok, w = wrist.read()
            if ok: cv2.imwrite(f'snapshots/m3_{r["id"]}_wrist.jpg', w)
            qr = mv.read()
            print(f'{r["id"]} hover reached: tip mm {kin.fk_xyz(qr).round(1)} (target {r["x_mm"]},{r["y_mm"]})', flush=True)
            lab.configure(text=f'{r["id"]}: hovering. Is the gripper centred over the rock? (3 s)'); root.update()
            t0 = time.time()
            while time.time() - t0 < 3: hold(); time.sleep(0.05)
            mv.move(qa)
        mv.move(home)
    finally:
        wrist.release()
        try: root.destroy()
        except Exception: pass
        follower.disconnect()  # torque off at home (rest) pose


if __name__ == "__main__":
    if "--test" in sys.argv:
        touch_test(dry="--dry" in sys.argv)
    else:
        main()
