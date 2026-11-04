"""Chand executor: smooth joint motion for the SO-101 follower (+ later: grasp/carry/drop).

Mover.move(): minimum-jerk profile at 50 Hz, peak speed PEAK_DEG_S. Gravity sag (servos stop short under
load, P gain 16) is handled by an integral bias that is blended in during the last part of each move and
kept between moves, so there are no correction jumps at the end.
"""
import time
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
J = [f"{j}.pos" for j in JOINTS]
HZ = 50
PEAK_DEG_S = 40.0      # fastest joint's peak speed
KI = 0.04              # integral gain per tick (≈ 0.5 s time constant at 50 Hz)
MAX_BIAS = 12.0        # deg
INTEGRATE_FROM = 0.6   # start integrating once 60 % of the move is done (servo lag early on is not sag)


def _minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Mover:
    def __init__(self, follower, hold=None):
        self.f = follower
        self.hold = hold                 # callable -> True while paused
        self.bias = np.zeros(5)          # carried between moves
        o = follower.get_observation()
        self.q = np.array([o[k] for k in J])
        self.grip = float(o["gripper.pos"])

    def read(self):
        o = self.f.get_observation()
        return np.array([o[k] for k in J])

    def _send(self, q, grip):
        self.f.send_action({**{k: float(v) for k, v in zip(J, q)}, "gripper.pos": float(grip)})

    def move(self, q_to, grip=None, settle_s=0.8, tol=0.6):
        grip = self.grip if grip is None else grip
        q_from, q_to = self.q.copy(), np.asarray(q_to, float)
        d = np.abs(q_to - q_from).max()
        T = max(0.6, 1.875 * d / PEAK_DEG_S)          # min-jerk peak speed = 1.875 * d / T
        n = int(T * HZ)
        t_next = time.perf_counter()
        for i in range(1, n + 1):
            while self.hold and self.hold():
                time.sleep(0.05); t_next = time.perf_counter()
            s = i / n
            qd = q_from + (q_to - q_from) * _minjerk(s)
            if s >= INTEGRATE_FROM:
                self.bias = np.clip(self.bias + KI * (qd - self.read()), -MAX_BIAS, MAX_BIAS)
            self._send(qd + self.bias, grip)
            t_next += 1 / HZ
            time.sleep(max(0.0, t_next - time.perf_counter()))
        # gentle settle: same integrator, same rate, no jumps
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < settle_s:
            e = q_to - self.read()
            if np.abs(e).max() < tol:
                break
            self.bias = np.clip(self.bias + KI * e, -MAX_BIAS, MAX_BIAS)
            self._send(q_to + self.bias, grip)
            time.sleep(1 / HZ)
        e = q_to - self.read()
        print(f"  move {d:.0f} deg in {T:.1f} s, final err {np.round(e, 1)} bias {np.round(self.bias, 1)}", flush=True)
        self.q, self.grip = q_to, grip
        return e

    def follow(self, path, grip=None, peak_deg_s=PEAK_DEG_S):
        """Stream a joint path (N x 5) without stopping at each waypoint: constant speed along the path,
        eased in/out over the whole length, same integral sag bias as move()."""
        grip = self.grip if grip is None else grip
        P = np.vstack([self.q, np.asarray(path, float)])
        seg = np.abs(np.diff(P, axis=0)).max(axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        L = cum[-1]
        if L < 0.5:
            return
        T = max(0.8, 1.875 * L / peak_deg_s)
        n = int(T * HZ)
        t_next = time.perf_counter()
        for i in range(1, n + 1):
            while self.hold and self.hold():
                time.sleep(0.05); t_next = time.perf_counter()
            d = _minjerk(i / n) * L
            k = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
            a = 0 if seg[k] == 0 else (d - cum[k]) / seg[k]
            qd = P[k] + (P[k + 1] - P[k]) * min(1.0, a)
            self.bias = np.clip(self.bias + 0.5 * KI * (qd - self.read()), -MAX_BIAS, MAX_BIAS)
            self._send(qd + self.bias, grip)
            t_next += 1 / HZ
            time.sleep(max(0.0, t_next - time.perf_counter()))
        self.q, self.grip = P[-1].copy(), grip

    def gripper(self, value, wait=0.6, deg_s=60.0):
        """Ramp the gripper to `value` while holding the arm. A single command is not enough:
        max_relative_target clips every command to 5 deg from the present position."""
        start = float(self.f.get_observation()["gripper.pos"])
        n = max(1, int(abs(value - start) / deg_s * HZ))
        for i in range(1, n + 1):
            self._send(self.q + self.bias, start + (value - start) * i / n)
            time.sleep(1 / HZ)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < wait:   # keep commanding: the clip re-applies each tick
            self._send(self.q + self.bias, value)
            time.sleep(1 / HZ)
        self.grip = value
        return float(self.f.get_observation()["gripper.pos"])


# ---------------------------------------------------------------- pick & place from the taught skill
import kin  # noqa: E402

APPROACH_MM = 60.0   # back off along the tool axis before the grasp
LIFT_MM = 70.0
REACH_MM = (90.0, 375.0)


def _rotz(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def line(a, b, q_seed, axis, step_mm=8.0, backwards=False):
    """IK along the straight segment a->b. backwards=True solves from b to a (q_seed is the pose at b)
    and returns the path in a->b order. Returns (path Nx5, worst position error mm)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step_mm)) + 1)
    pts = [a + (b - a) * t for t in np.linspace(0, 1, n)]
    if backwards:
        pts = pts[::-1]
    q, path, worst = np.asarray(q_seed, float), [], 0.0
    for p in pts:
        q, e = kin.ik(p, q, down=False, axis=axis)
        path.append(q.copy()); worst = max(worst, e)
    path = np.array(path[::-1] if backwards else path)
    if np.abs(np.diff(path, axis=0)).max() > 12:
        worst = max(worst, 99.0)   # posture jump between neighbours: refuse
    return path, worst


def plan_pick(rock, skill, calib):
    """Joint targets for one rock, from the demo's grasp style:
    the demo's tip offset and tool tilt are expressed relative to the rock's direction from the base and
    rotated to the new rock. Seeded from the demo grasp joints so the arm uses the same posture."""
    x, y = rock["x_mm"], rock["y_mm"]
    r = np.hypot(x, y)
    if not REACH_MM[0] <= r <= REACH_MM[1]:
        raise ValueError(f"rock at {r:.0f} mm from base, outside reach {REACH_MM}")
    d_az = np.arctan2(y, x) - np.arctan2(skill["picked"]["y_mm"], skill["picked"]["x_mm"])
    off = _rotz(skill["grasp"]["offset_from_rock_top_mm"], d_az)
    axis0 = _rotz(skill["grasp"]["tool_z"], d_az)
    tip = np.array([x, y, calib["rock_top_z_mm"]]) + off
    seed = np.array(skill["grasp"]["q_deg"], float); seed[0] += np.rad2deg(d_az)
    # demo tilt first; near the base that pins the elbow at its limit, so try steeper (more vertical) tilts
    for blend in (0.0, 0.25, 0.5, 0.75):
        axis = (1 - blend) * axis0 + blend * np.array([0, 0, -1.0]); axis /= np.linalg.norm(axis)
        q_grasp, e1 = kin.ik(tip, seed, down=False, axis=axis)
        # straight lines in space (joint interpolation near the base swings the tip in an arc):
        # IK every ~8 mm, each seeded from the previous point, refused if the posture jumps
        descend, e2 = line(tip - APPROACH_MM * axis, tip, q_grasp, axis, backwards=True)
        lift, e3 = line(tip, tip + [0, 0, LIFT_MM], q_grasp, axis)
        err = max(e1, e2, e3)
        if err <= 6:
            break
    else:
        raise ValueError(f"no smooth straight-line grasp (IK/posture error {err:.1f})")
    q_pre, q_lift = descend[0], lift[-1]
    # carry: demo path with the grasp difference faded out, so it ends exactly at the demo drop pose
    C = np.array(skill["carry_q_deg"], float)
    delta = q_grasp - np.array(skill["grasp"]["q_deg"], float)
    s = np.linspace(0, 1, len(C))[:, None]
    carry = C + delta * (1 - s) ** 2
    # demo carry starts at its grasp: start ours from the lift pose (first third replaced by lift)
    k = max(1, len(C) // 4)
    carry = np.vstack([q_lift, carry[k:]])
    tz = [kin.fk_xyz(q)[2] for q in carry]
    return dict(q_pre=q_pre, q_grasp=q_grasp, q_lift=q_lift, descend=descend, lift=lift, carry=carry, drop=np.array(skill["drop_q_deg"], float),
                tip=tip.round(1).tolist(), ik_err=round(err, 1), tilt_blend=blend, carry_min_z=round(float(min(tz)), 1))


def pick_place(mv, plan, skill, step=lambda name: None):
    """Executes a plan. step(name) is called before each phase (the mission uses it for step mode / HUD).
    Returns dict(held=bool, grip=value)."""
    g = skill["gripper"]
    g_open = g["open"] + 8          # a little wider than the demo for slack
    home = mv.q.copy()
    step("approach"); mv.move(plan["q_pre"], grip=g_open)
    step("descend");  mv.follow(plan["descend"][1:], grip=g_open, peak_deg_s=25); mv.move(plan["q_grasp"], grip=g_open, settle_s=1.2)
    step("close");    gm = mv.gripper(g["close_cmd"], wait=0.8)
    held = gm > g["close_cmd"] + 2.0   # blocked by the rock = holding (verify)
    print(f"  gripper cmd {g['close_cmd']} measured {gm:.1f} -> {'HOLDING' if held else 'MISSED'}", flush=True)
    step("lift");     mv.follow(plan["lift"][1:], peak_deg_s=25)
    if held:
        step("carry");  mv.follow(plan["carry"][1:])
        mv.move(plan["drop"], settle_s=0.5)
        step("drop");   mv.gripper(g_open, wait=0.6)
    else:
        mv.gripper(g_open, wait=0.3)
    step("home");     mv.move(home, grip=g["close_cmd"] + 6)  # gripper nearly closed at rest, not squeezing
    return dict(held=held, grip=gm)
