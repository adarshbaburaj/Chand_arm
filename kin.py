"""SO-101 forward/inverse kinematics in plain numpy (no ikpy/placo).
URDF: SO-ARM100 so101_new_calib.urdf. LeRobot use_degrees=True values are degrees with 0 at mid-range,
which is the zero pose of this URDF, so rad = deg2rad(lerobot_deg) (to be checked against the real arm).
Positions are in mm in the arm base frame (base_link).
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

URDF = Path(__file__).with_name("assets") / "so101_new_calib.urdf"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP = "gripper_frame_link"


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_rot(ax, q):
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def _load_chain():
    root = ET.parse(URDF).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain, link = [], TIP
    while link in by_child:
        j = by_child[link]
        o = j.find("origin")
        T = np.eye(4)
        T[:3, :3] = _rpy(*map(float, o.get("rpy").split()))
        T[:3, 3] = list(map(float, o.get("xyz").split()))
        a = j.find("axis")
        axis = np.array(list(map(float, a.get("xyz").split()))) if a is not None else None
        chain.append((j.get("name"), j.get("type"), T, axis))
        link = j.find("parent").get("link")
    return chain[::-1]


CHAIN = _load_chain()
LIMITS = {}
for _j in ET.parse(URDF).getroot().findall("joint"):
    if _j.find("limit") is not None and _j.get("name") in JOINTS:
        LIMITS[_j.get("name")] = (float(_j.find("limit").get("lower")), float(_j.find("limit").get("upper")))


def fk(q_deg):
    """q_deg: 5 LeRobot degrees (pan, lift, elbow, wrist_flex, wrist_roll) -> 4x4 tip pose, translation in mm."""
    q = dict(zip(JOINTS, np.deg2rad(np.asarray(q_deg, float))))
    T = np.eye(4)
    for name, typ, To, axis in CHAIN:
        T = T @ To
        if typ == "revolute" and name in q:
            R = np.eye(4); R[:3, :3] = _axis_rot(axis, q[name]); T = T @ R
    T = T.copy(); T[:3, 3] *= 1000.0
    return T


def fk_xyz(q_deg):
    return fk(q_deg)[:3, 3]


def ik(target_mm, q0_deg, down=True, iters=200, tol_mm=1.0, axis=None):
    """Damped least squares on pan/lift/elbow/wrist_flex (wrist_roll held). down=True also asks the
    gripper axis to point at the table; axis=(3,) asks for that tool-z direction instead (e.g. the demo's
    45 deg tilt). Returns (q_deg, err_mm)."""
    want = None if axis is None and not down else (np.array([0, 0, -1.0]) if axis is None else np.asarray(axis, float))
    q = np.asarray(q0_deg, float).copy()
    lo = np.rad2deg([LIMITS[j][0] for j in JOINTS]); hi = np.rad2deg([LIMITS[j][1] for j in JOINTS])
    target = np.asarray(target_mm, float)

    def resid(qq):
        T = fk(qq)
        r = T[:3, 3] - target
        if want is not None:  # tool z axis direction, weighted as ~50 mm per radian
            r = np.concatenate([r, 50.0 * (T[:3, 2] - want)])
        return r

    for _ in range(iters):
        r = resid(q)
        if np.linalg.norm(r[:3]) < tol_mm and (want is None or np.linalg.norm(r[3:]) < 2.0):
            break
        J = np.zeros((len(r), 4)); eps = 0.05
        for i in range(4):
            dq = np.zeros(5); dq[i] = eps
            J[:, i] = (resid(q + dq) - r) / eps
        lam = 2.0
        step = -np.linalg.solve(J.T @ J + lam * np.eye(4), J.T @ r)
        q[:4] = np.clip(q[:4] + np.clip(step, -10, 10), lo[:4], hi[:4])
    return q, float(np.linalg.norm(fk_xyz(q) - target))


if __name__ == "__main__":
    print("zero pose tip mm:", fk_xyz([0, 0, 0, 0, 0]).round(1))
    print("tool z at zero:", fk([0, 0, 0, 0, 0])[:3, 2].round(2))
    for t in ([200, 0, 30], [150, -100, 30], [250, 80, 30]):
        q, e = ik(t, [0, -30, 30, 60, 0])
        print("ik", t, "->", q.round(1), "err mm", round(e, 2), "toolz", fk(q)[:3, 2].round(2))
