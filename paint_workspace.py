"""Paint an arm's working area with its own gripper tip, on a plain canvas (no camera).

  python paint_workspace.py --arm right|left

Torque is switched OFF: hold the follower and move it by hand. The gripper tip is the brush: wherever the tip goes,
the canvas (top-down view of the arm, forward = up, base = black dot) gets painted. Colour = tip height:
  BLUE   = down at the table (it can pick here)
  GREEN  = up to 10 cm above the table
  ORANGE = higher
Keys: SPACE pause/resume the brush, + / - brush size, C clear, Q save + quit.
Saves workspace_<arm>_canvas.png (the picture), workspace_<arm>.json (all tip points in arm mm) and, from the BLUE
(table-level) part, workspace_<arm>.png (mask in top-camera pixels that the mission uses to decide which arm can
reach an object). Torque stays OFF when it exits: put the arm down at rest yourself.
"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
import kin

HERE = Path(__file__).parent
ARM = "left" if "--arm" in sys.argv and sys.argv[sys.argv.index("--arm") + 1] == "left" else "right"
PORT = {"right": "/dev/ttyFollowerLeft", "left": "/dev/ttyFollowerRight"}[ARM]
CALIB = {"right": "calib.json", "left": "calib_left.json"}[ARM]
TABLE_LEVEL_MM = 45.0       # tip lower than this above the table = "can pick here" (blue)
MID_LEVEL_MM = 100.0
SIZE, MM_PER_PX = 800, 1.0  # canvas covers -400..400 mm around the base
J = [f"{j}.pos" for j in kin.JOINTS]
BLUE, GREEN, ORANGE = (230, 120, 20), (60, 180, 60), (0, 140, 255)     # BGR


def to_canvas(x, y):
    return int(SIZE / 2 - y / MM_PER_PX), int(SIZE / 2 - x / MM_PER_PX)   # forward (x) = up, left (y) = left


def blank():
    c = np.full((SIZE, SIZE, 3), 255, np.uint8)
    for r in range(100, 501, 100):                                       # range rings every 10 cm
        cv2.circle(c, (SIZE // 2, SIZE // 2), int(r / MM_PER_PX), (225, 225, 225), 1)
        cv2.putText(c, f"{r // 10} cm", (SIZE // 2 + 4, SIZE // 2 - int(r / MM_PER_PX) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
    cv2.line(c, (SIZE // 2, 0), (SIZE // 2, SIZE), (235, 235, 235), 1); cv2.line(c, (0, SIZE // 2), (SIZE, SIZE // 2), (235, 235, 235), 1)
    return c


def main():
    calib = json.loads((HERE / CALIB).read_text())
    table_z = calib["rock_top_z_mm"] - 30.0
    base_img = blank()
    levels = np.zeros((SIZE, SIZE), np.uint8)   # 0 none, 1 orange, 2 green, 3 blue
    pts = []                                    # [x, y, z_above_table]
    f = SOFollower(SOFollowerRobotConfig(port=PORT, id=f"thumbi_{ARM}"))
    f.connect(calibrate=False)
    f.bus.disable_torque()
    print(f"{ARM} arm torque OFF: move it by hand, the gripper tip is the brush.", flush=True)
    root = tk.Tk(); root.title(f"chand paint workspace: {ARM.upper()} arm"); panel = tk.Label(root); panel.pack()
    keys = []
    root.bind("<Key>", lambda e: keys.append(e.char.lower() if e.char else ""))
    root.protocol("WM_DELETE_WINDOW", lambda: keys.append("q"))
    on, brush = True, 12
    try:
        while True:
            o = f.get_observation()
            tip = kin.fk_xyz([o[k] for k in J])
            h = float(tip[2] - table_z)
            cu, cv_ = to_canvas(tip[0], tip[1])
            col = BLUE if h < TABLE_LEVEL_MM else GREEN if h < MID_LEVEL_MM else ORANGE
            if on:                       # higher strokes never paint over lower ones (blue = table level wins)
                level = 3 if col == BLUE else 2 if col == GREEN else 1
                y0, y1, x0, x1 = max(0, cv_ - brush), min(SIZE, cv_ + brush + 1), max(0, cu - brush), min(SIZE, cu + brush + 1)
                if y1 > y0 and x1 > x0:
                    yy, xx = np.ogrid[y0:y1, x0:x1]
                    disc = (yy - cv_) ** 2 + (xx - cu) ** 2 <= brush ** 2
                    roi = levels[y0:y1, x0:x1]
                    roi[disc] = np.maximum(roi[disc], level)
                if not pts or np.hypot(pts[-1][0] - tip[0], pts[-1][1] - tip[1]) > 4 or abs(pts[-1][2] - h) > 8:
                    pts.append([round(float(tip[0]), 1), round(float(tip[1]), 1), round(h, 1)])
            view = base_img.copy()
            for lv, colr in ((1, ORANGE), (2, GREEN), (3, BLUE)):
                view[levels == lv] = colr
            cv2.circle(view, (SIZE // 2, SIZE // 2), 10, (0, 0, 0), -1)
            cv2.putText(view, "base", (SIZE // 2 + 12, SIZE // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.circle(view, (cu, cv_), brush, (0, 0, 255), 2)
            cv2.putText(view, f"{ARM.upper()} arm   tip x {tip[0]:.0f}  y {tip[1]:.0f} mm   height {h:.0f} mm   "
                              f"{'BRUSH ON' if on else 'PAUSED'}   brush {brush}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            for i, (c, t) in enumerate(((BLUE, f"at the table (< {TABLE_LEVEL_MM:.0f} mm): can pick"), (GREEN, "up to 10 cm"), (ORANGE, "higher"))):
                cv2.rectangle(view, (10, 40 + 22 * i), (28, 56 + 22 * i), c, -1)
                cv2.putText(view, t, (34, 54 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(view, "SPACE pause   + / - brush   C clear   Q save+quit   (forward = up)", (10, SIZE - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
            panel.configure(image=img); panel.image = img
            root.update()
            time.sleep(0.02)
            c = keys.pop(0) if keys else ""
            if c == " ": on = not on
            elif c in ("+", "="): brush = min(40, brush + 3)
            elif c == "-": brush = max(3, brush - 3)
            elif c == "c": levels[:] = 0; pts = []
            elif c == "q": break
    finally:
        try: root.destroy()
        except Exception: pass
        try: f.disconnect()
        except Exception: pass
        view = base_img.copy()
        for lv, colr in ((1, ORANGE), (2, GREEN), (3, BLUE)):
            view[levels == lv] = colr
        save(view, pts, calib)


def save(paint, pts, calib):
    if paint is not None:
        cv2.imwrite(str(HERE / f"workspace_{ARM}_canvas.png"), paint)
        (HERE / f"workspace_{ARM}.json").write_text(json.dumps({"arm": ARM, "points_mm_xyh": pts, "t": time.strftime("%Y-%m-%d %H:%M:%S")}))
    A = np.vstack([np.asarray(calib["A_px_to_mm"], float), [0, 0, 1]]); Ai = np.linalg.inv(A)
    mm_per_px = float(np.sqrt(abs(np.linalg.det(A[:2, :2]))))
    for name, sel, close_mm in (("", lambda p: True, 60), ("_table", lambda p: p[2] < TABLE_LEVEL_MM, 30)):
        xy = np.array([[p[0], p[1]] for p in pts if sel(p)])
        if len(xy) < 20:
            print(f"workspace_{ARM}{name}.png: only {len(xy)} points, not saved"); continue
        px = (Ai @ np.c_[xy, np.ones(len(xy))].T).T[:, :2]
        m = np.zeros((480, 640), np.uint8)
        for u, v in px:
            cv2.circle(m, (int(u), int(v)), max(3, int(20 / mm_per_px)), 255, -1)
        k = max(3, int(close_mm / mm_per_px)) | 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cv2.imwrite(str(HERE / f"workspace_{ARM}{name}.png"), m)
        print(f"saved workspace_{ARM}{name}.png ({int((m > 0).sum())} camera px from {len(xy)} points)", flush=True)
    # the whole painted area (any height) is the reach envelope the mission uses; the blue (table) part is kept
    # separately as workspace_<arm>_table.png for reference


def rebuild():
    calib = json.loads((HERE / CALIB).read_text())
    pts = json.loads((HERE / f"workspace_{ARM}.json").read_text())["points_mm_xyh"]
    save(None, pts, calib)


if __name__ == "__main__":
    rebuild() if "--rebuild" in sys.argv else main()
