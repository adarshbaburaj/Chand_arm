"""Teach the exact drop point over the red bin (right arm).

  python teach_bin.py

The right arm goes limp (torque off). Hold it and put the gripper tip over the MIDDLE of the bin opening, a few cm
above the rim (where objects should be let go), then press SPACE. Q quits. Saves bin_drop.json (joints + tip mm);
the mission then drops every object from exactly this pose and also looks for the next object from here.
Torque stays OFF afterwards: put the arm down at rest yourself.
"""
import json, time
from pathlib import Path
import tkinter as tk
import numpy as np
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
import kin

HERE = Path(__file__).parent
J = [f"{j}.pos" for j in kin.JOINTS]


def main():
    f = SOFollower(SOFollowerRobotConfig(port="/dev/ttyFollowerLeft", id="thumbi_right"))
    f.connect(calibrate=False)
    f.bus.disable_torque()
    root = tk.Tk(); root.title("chand: teach bin centre (right arm)")
    lab = tk.Label(root, font=("Sans", 16), width=60, height=6, justify="left"); lab.pack()
    keys = []
    root.bind("<Key>", lambda e: keys.append(e.char.lower() if e.char else ""))
    root.protocol("WM_DELETE_WINDOW", lambda: keys.append("q"))
    saved = None
    try:
        while True:
            o = f.get_observation()
            q = np.array([o[k] for k in J]); tip = kin.fk_xyz(q)
            lab.configure(text=f"RIGHT arm is limp. Hold the gripper tip over the MIDDLE of the bin,\n"
                               f"a few cm above the rim, then press SPACE.   Q = quit\n\n"
                               f"tip  x {tip[0]:.0f}   y {tip[1]:.0f}   z {tip[2]:.0f} mm\n"
                               f"{'SAVED: ' + str(saved) if saved else ''}")
            root.update(); time.sleep(0.03)
            c = keys.pop(0) if keys else ""
            if c == " ":
                saved = [round(float(v), 1) for v in tip]
                (HERE / "bin_drop.json").write_text(json.dumps({"q_deg": [round(float(v), 2) for v in q], "tip_mm": saved,
                                                                 "gripper": float(o["gripper.pos"]), "t": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
                print("saved bin_drop.json:", saved, "q", np.round(q, 1), flush=True)
            elif c == "q":
                break
    finally:
        try: root.destroy()
        except Exception: pass
        f.disconnect()


if __name__ == "__main__":
    main()
