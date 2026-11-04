"""Two-arm assist: the LEFT arm pushes an object into the RIGHT arm's open gripper.

Why: on non-cube shapes the right arm stops a little short, the object ends up just beyond its fingertips and the
jaws close on air (run 6). Instead of only re-aiming the right arm, the left arm comes from the far side and slides
the object toward the right arm until it sits between the open jaws; then the right arm closes with force feedback.

Frames: both arms are touch-calibrated to the same top camera (calib.json = right, calib_left.json = left), so a
point in right-arm mm maps to left-arm mm through the camera pixels: mm_L = A_L @ inv(A_R) @ mm_R.
"""
import json
from pathlib import Path
import numpy as np
import kin
from execute import line, Mover

HERE = Path(__file__).parent
FINGER_Z_ABOVE_TABLE = 10.0     # push with the closed left fingertip 10 mm above the table (below the object's middle)
OBJ_RADIUS = 15.0               # ~half the object width
START_BEHIND = 35.0             # start this far behind the object's far edge
STOP_SHORT = 5.0                # object centre stops 5 mm short of the right tip point (still between the jaws);
                                # the pusher then stays OBJ_RADIUS+5 = 20 mm away from the right fingertips
DOWN = np.array([0, 0, -1.0])
LEFT_PORT, LEFT_ID = "/dev/ttyFollowerRight", "thumbi_left"


def _h(A):
    return np.vstack([np.asarray(A, float), [0, 0, 1]])


def frames():
    """3x3 affine right-mm -> left-mm (via camera pixels) and the left arm's table z."""
    cr = json.loads((HERE / "calib.json").read_text()); cl = json.loads((HERE / "calib_left.json").read_text())
    R2L = _h(cl["A_px_to_mm"]) @ np.linalg.inv(_h(cr["A_px_to_mm"]))
    table_z_left = cl["rock_top_z_mm"] - 30.0 + 0.0     # calibration touched 30 mm-tall rock tops
    return R2L, table_z_left


def to_left(R2L, xy_r):
    return (R2L @ np.r_[xy_r[0], xy_r[1], 1.0])[:2]


def plan_push(rock_r, tip_r, q_seed_left, R2L=None, table_z_left=None):
    """rock_r: object centre (right-arm mm). tip_r: right gripper tip point waiting at the grasp (right mm).
    Pushes along the line object -> right tip. Returns dict(path_in, push, path_out, err) in LEFT joint space."""
    if R2L is None:
        R2L, table_z_left = frames()
    rock_r, tip_r = np.asarray(rock_r[:2], float), np.asarray(tip_r[:2], float)
    d = tip_r - rock_r
    n = np.linalg.norm(d)
    u = d / n if n > 1 else -rock_r / np.linalg.norm(rock_r)       # push direction (toward the right gripper)
    # finger contact point = object's far edge, travelling with the object until the centre is at tip + STOP_BEFORE_TIP
    p0_r = rock_r - u * (OBJ_RADIUS + START_BEHIND)
    p1_r = tip_r - u * (STOP_SHORT + OBJ_RADIUS)
    if np.dot(p1_r - p0_r, u) <= 0:
        p1_r = p0_r + u * (START_BEHIND + 5)
    z = table_z_left + FINGER_Z_ABOVE_TABLE
    p0, p1 = np.r_[to_left(R2L, p0_r), z], np.r_[to_left(R2L, p1_r), z]
    above = p0 + [0, 0, 50.0]
    q_above, e0 = kin.ik(above, q_seed_left, down=True)
    path_in, e1 = line(above, p0, q_above, DOWN)
    push, e2 = line(p0, p1, path_in[-1], DOWN, step_mm=6)
    path_out, e3 = line(p1, p1 + [0, 0, 60.0], push[-1], DOWN)
    return dict(q_above=q_above, path_in=path_in, push=push, path_out=path_out, err=round(max(e0, e1, e2, e3), 1),
                p0_left=p0.round(1).tolist(), p1_left=p1.round(1).tolist(), push_mm=round(float(np.linalg.norm(p1 - p0)), 1))


def do_push(left_mv: Mover, plan, left_home):
    """Execute on the left arm (fingers closed = a narrow pusher). Slow through the push."""
    left_mv.move(plan["q_above"], grip=1.0)
    left_mv.follow(plan["path_in"][1:], peak_deg_s=25)
    left_mv.follow(plan["push"][1:], peak_deg_s=12)
    left_mv.follow(plan["path_out"][1:], peak_deg_s=25)
    left_mv.move(left_home)


if __name__ == "__main__":     # dry check: can the left arm push objects at these right-arm positions?
    import sys
    R2L, tz = frames()
    seed = [0, -30, 60, 60, 0]
    for xy in ([200, 150], [250, 100], [180, 220], [300, 150]):
        rock = np.array(xy, float); tip = rock - 20 * rock / np.linalg.norm(rock)
        try:
            p = plan_push(rock, tip, seed, R2L, tz)
            print(xy, "-> left p0", p["p0_left"], "p1", p["p1_left"], "push", p["push_mm"], "mm, IK err", p["err"])
        except Exception as e:
            print(xy, "cannot:", e)
