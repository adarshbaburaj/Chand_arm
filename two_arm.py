"""Two-arm logic: which arm takes an object, and the LEFT arm handing objects to the RIGHT arm.

Rule (user, 01:55): if the right arm cannot reach an object, the left arm picks it and puts it down near the right
arm, at a handoff spot both arms reach (as close as possible to where the right arm grasps best = the teach-demo
spot). The right arm then picks it from there into the bin as usual.

Reach: the painted workspace masks (paint_workspace.py -> workspace_<arm>.png) if they exist, else a geometric +
IK check. Frames: calib.json (right) and calib_left.json (left), both px -> own-base mm.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import execute, kin, perceive

HERE = Path(__file__).parent
LEFT_PORT, LEFT_ID = "/dev/ttyFollowerRight", "thumbi_left"
CAL = {"right": "calib.json", "left": "calib_left.json"}
_masks = {}


def calib(arm):
    return json.loads((HERE / CAL[arm]).read_text())


def mask(arm):
    if arm not in _masks:
        p = HERE / f"workspace_{arm}.png"
        _masks[arm] = cv2.imread(str(p), 0) if p.exists() else None
    return _masks[arm]


def px_to_arm(arm, u, v):
    return perceive.to_mm(u, v, calib(arm))


def arm_to_px(arm, x, y):
    A = np.vstack([np.asarray(calib(arm)["A_px_to_mm"], float), [0, 0, 1]])
    return (np.linalg.inv(A) @ [x, y, 1.0])[:2]


def can_pick(arm, u, v, skill):
    """(ok, why). Painted mask first (with a small margin), then a real grasp plan in that arm's frame."""
    m = mask(arm)
    if m is not None:
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < m.shape[1] and 0 <= vi < m.shape[0]) or m[vi, ui] == 0:
            return False, "outside painted workspace"
    x, y = px_to_arm(arm, u, v)
    try:
        execute.plan_pick({"x_mm": x, "y_mm": y}, skill, calib(arm))
        return True, "ok"
    except ValueError as e:
        return False, str(e)


def handoff_spot(skill):
    """Pixel where the left arm should put objects for the right arm: reachable by both, closest to the right arm's
    demo grasp spot (where it succeeds best). Cached in handoff.json; delete the file to recompute."""
    f = HERE / "handoff.json"
    if f.exists():
        return tuple(json.loads(f.read_text())["px"])
    best_u, best_v = arm_to_px("right", skill["picked"]["x_mm"], skill["picked"]["y_mm"])
    cands = []
    both = (mask("right") > 0) & (mask("left") > 0) if mask("right") is not None and mask("left") is not None else None
    for u in range(20, 620, 15):
        for v in range(20, 460, 15):
            if both is not None and not both[v, u]:
                continue
            b = perceive.bin_roi()
            if b[0] <= u <= b[2] and b[1] <= v <= b[3]:
                continue
            cands.append((np.hypot(u - best_u, v - best_v), u, v))
    for _, u, v in sorted(cands):
        if np.hypot(*px_to_arm("left", u, v)) > 320 or np.hypot(*px_to_arm("right", u, v)) > 320:
            continue                       # keep the spot comfortably inside BOTH arms' reach
        if can_pick("right", u, v, skill)[0] and can_pick("left", u, v, skill)[0]:
            # keep a margin: the right arm's grasp must also work 15 mm around it (objects don't land exactly)
            if all(can_pick("right", u + du, v + dv, skill)[0] for du, dv in ((15, 0), (-15, 0), (0, 15), (0, -15))):
                f.write_text(json.dumps({"px": [u, v], "right_mm": px_to_arm("right", u, v), "left_mm": px_to_arm("left", u, v)}))
                return u, v
    raise RuntimeError("no spot reachable by both arms")


def left_handoff(mv_left, rock_px, skill, spot_px, step=lambda n: None, vis=None, shift_rt=(0.0, 0.0), zone=None, spot_free=None,
                 held_zone=None, extra=None):
    """zone: threading.Lock for the shared area; acquired before entering it, released by the caller after the left arm
    is home (held_zone['yes'] tells the caller it holds it)."""
    held_zone = {} if held_zone is None else held_zone
    """Left arm: pick the object at rock_px (force-feedback grasp + regrasp search, same as the right arm), carry it
    to spot_px and put it down gently with the same grasp pose, open, back off, go home."""
    cl = calib("left")
    x, y = px_to_arm("left", *rock_px)
    plan = execute.plan_pick({"x_mm": x, "y_mm": y, **(extra or {})}, skill, cl, shift=execute.rt_to_world(shift_rt, np.arctan2(y, x)))
    sx, sy = px_to_arm("left", *spot_px)
    put = execute.plan_pick({"x_mm": sx, "y_mm": sy}, skill, cl)        # same grasp pose, at the spot

    def place(mv, plan_used):
        # the handoff spot is in the shared area: wait for the zone (right arm not working there) and a free spot
        import time
        t0 = time.time()
        while True:
            step("left: waiting for the handoff spot")
            if zone is None or zone.acquire(timeout=0.5):
                if spot_free is None or spot_free():
                    break
                if zone is not None: zone.release()
            if time.time() - t0 > 60:
                raise RuntimeError("handoff spot stayed busy for 60 s")
            time.sleep(0.5)
        held_zone["yes"] = zone is not None
        _place(mv)          # the CALLER releases the zone once the left arm is back home (see held_zone)

    def _place(mv):
        step("left: carry to handoff spot")
        mv.move(put["q_lift"])
        mv.follow(put["lift"][::-1][1:], peak_deg_s=20)                  # straight down into the put-down pose
        step("left: release"); mv.gripper(skill["gripper"]["open"] + 8, wait=0.5)
        mv.follow(put["descend"][::-1][1:], peak_deg_s=25)               # back off along the approach line

    return execute.pick_place(mv_left, plan, skill, step, vis=vis, place=place)


if __name__ == "__main__":   # dry check (no motion): reach of both arms + handoff spot
    skill = json.loads((HERE / "skill.json").read_text())
    img = perceive.grab()
    objs, _ = perceive.detect(img)
    for o in objs:
        print(o["id"], o["shape"], (o["u"], o["v"]), "right:", can_pick("right", o["u"], o["v"], skill),
              "left:", can_pick("left", o["u"], o["v"], skill))
    print("handoff spot px:", handoff_spot(skill), json.loads((HERE / "handoff.json").read_text()))
