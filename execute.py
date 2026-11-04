"""Chand executor: smooth joint motion for the SO-101 follower (+ later: grasp/carry/drop).

Mover.move(): minimum-jerk profile at 50 Hz, peak speed PEAK_DEG_S. Gravity sag (servos stop short under
load, P gain 16) is handled by an integral bias that is blended in during the last part of each move and
kept between moves, so there are no correction jumps at the end.
"""
import time
import numpy as np
import viz

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
J = [f"{j}.pos" for j in JOINTS]
HZ = 50
PEAK_DEG_S = 40.0      # fastest joint's peak speed
KI = 0.04              # integral gain per tick (≈ 0.5 s time constant at 50 Hz)
MAX_BIAS = 12.0        # deg
INTEGRATE_FROM = 0.6   # start integrating once 60 % of the move is done (servo lag early on is not sag)


def _minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


GRIP_CLOSED = 0.0          # gripper.pos is 0-100 % of range; empty jaws close freely to ~0.9
HOLD_LOAD, HOLD_CUR = 10.0, 6   # probe 01:50: holding a 30 mm cube -> load 15.6 %, current 11; empty -> 3-6 %, 0-2


# Gripper strength. LeRobot's SOFollower.configure() writes (on EVERY connect) gripper Max_Torque_Limit 500 (50 %),
# Protection_Current 250, Overload_Torque 25 %. With Protection_Time 200 (~2 s) the servo trips "Overload" when it
# squeezes harder than 25 % for ~1-2 s and drops to Protective_Torque 20 %. That is the "motor cuts out" the user saw.
# We raise only the overload threshold + torque cap for the gripper, and keep the over-current and 70 °C
# temperature protections. The hard squeeze lasts only while carrying (a few s); temperature is watched.
GRIP_BOOST = {"Max_Torque_Limit": 800, "Torque_Limit": 800, "Overload_Torque": 80, "Protection_Time": 250}
GRIP_HOT_C = 55          # above this, squeeze gently


def boost_gripper(follower):
    b = follower.bus
    with b.torque_disabled("gripper"):
        for k, v in GRIP_BOOST.items():
            b.write(k, "gripper", v)
    return {k: b.read(k, "gripper", normalize=False) for k in GRIP_BOOST}


def pushback(load, cur):
    return abs(load) > HOLD_LOAD or cur > HOLD_CUR


class Mover:
    def __init__(self, follower, hold=None, name="right"):
        self.name = name
        self.f = follower
        self.hold = hold                 # callable -> True while paused
        self.bias = np.zeros(5)          # carried between moves
        o = follower.get_observation()
        self.q = np.array([o[k] for k in J])
        self.grip = float(o["gripper.pos"])

    def read(self):
        o = self.f.get_observation()
        q = np.array([o[k] for k in J])
        viz.joints(self.name, q_meas=q)
        return q

    def _send(self, q, grip):
        viz.joints(self.name, q_cmd=q); viz.gripper(self.name, cmd=grip)
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

    def grip_sense(self):
        """Gripper (position deg, load %, current raw). STS3215 Present_Load: bit 10 = direction, low 10 bits =
        0..1000 (0.1 % of max torque). This is the 'force feedback': the jaw pushing against something."""
        b = self.f.bus
        ph, port = b.packet_handler, b.port_handler
        raw, _, self.grip_err = ph.read2ByteTxRx(port, 6, 60)      # status flags (e.g. overload) must not crash us
        load = (raw & 0x3FF) / 10.0 * (-1 if raw & 0x400 else 1)
        cur, _, _ = ph.read2ByteTxRx(port, 6, 69)
        pos = float(self.f.get_observation()["gripper.pos"]) if self.grip_err else float(b.read("Present_Position", "gripper"))
        viz.gripper(self.name, pos=pos, load=load, cur=cur)
        return pos, load, int(cur)

    def grip_temp(self):
        v, _, _ = self.f.bus.packet_handler.read1ByteTxRx(self.f.bus.port_handler, 6, 63)
        return int(v)

    def grasp_ff(self, close_cmd, deg_s=20.0, lag_deg=2.0, load_pct=None, squeeze=8.0, log=None):
        """Force-feedback close: close slowly; contact = the jaw stops following the command (lag) or the
        load rises. On contact, command only `squeeze` deg past the contact point (gentle, no squirting the
        object out), then check the pushback. Returns dict(contact, held, pos, load, trace)."""
        pos0, load0, _ = self.grip_sense()
        trace, contact, cmd = [], None, pos0
        n_hit = 0
        while cmd > close_cmd:
            cmd = max(close_cmd, cmd - deg_s / HZ)
            self._send(self.q + self.bias, cmd)
            time.sleep(1 / HZ)
            pos, load, cur = self.grip_sense()
            trace.append((round(cmd, 2), round(pos, 2), load, cur))
            hit = (pos - cmd) > lag_deg or (load_pct is not None and abs(load) > load_pct)
            n_hit = n_hit + 1 if hit else 0
            if n_hit >= 3:
                contact = pos
                break
        if contact is not None:
            if self.grip_temp() > GRIP_HOT_C:
                squeeze = min(squeeze, 3.0)
            target = max(close_cmd, contact - squeeze)
            for _ in range(int(0.6 * HZ)):
                self._send(self.q + self.bias, target)
                time.sleep(1 / HZ)
            self.grip = target
        else:
            for _ in range(int(0.3 * HZ)):
                self._send(self.q + self.bias, close_cmd); time.sleep(1 / HZ)
            self.grip = close_cmd
        pos, load, cur = self.grip_sense()
        held = contact is not None and pos > close_cmd + 2.0 and pushback(load, cur)   # something in the jaws
        res = dict(contact=contact, held=held, pos=round(pos, 1), load=load, cur=cur, cmd=round(self.grip, 1), trace=trace)
        if log is not None: log.append(res)
        return res

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


# ---- grasp memory: per shape, the (radial, tangential) mm offset that last worked. Deep by default: run 6 showed
# non-cube shapes ending just beyond the fingertips (arm stopped short), so start 12 mm deeper than the demo.
MEMORY_FILE = None
DEFAULT_RT = {"square": (0.0, 0.0), "_other": (12.0, 0.0)}


def _mem_path():
    from pathlib import Path
    return Path(__file__).with_name("grasp_memory.json")


def memory_get(shape):
    import json
    m = json.loads(_mem_path().read_text()) if _mem_path().exists() else {}
    if shape in m:
        return tuple(m[shape]["rt"]), "memory"
    return DEFAULT_RT.get(shape, DEFAULT_RT["_other"]), "default"


def memory_put(shape, rt, ok):
    """Successful offsets pull the memory toward them (EMA 0.5); failures are only counted."""
    import json
    m = json.loads(_mem_path().read_text()) if _mem_path().exists() else {}
    e = m.setdefault(shape, {"rt": list(DEFAULT_RT.get(shape, DEFAULT_RT["_other"])), "ok": 0, "fail": 0})
    if ok:
        e["rt"] = [round(0.5 * a + 0.5 * b, 1) for a, b in zip(e["rt"], rt)]
        e["ok"] += 1
    else:
        e["fail"] += 1
    _mem_path().write_text(json.dumps(m, indent=1))


def rt_to_world(rt, az):
    return (rt[0] * np.cos(az) - rt[1] * np.sin(az), rt[0] * np.sin(az) + rt[1] * np.cos(az))


def world_to_rt(sh, az):
    return (sh[0] * np.cos(az) + sh[1] * np.sin(az), -sh[0] * np.sin(az) + sh[1] * np.cos(az))


def plan_pick(rock, skill, calib, shift=(0.0, 0.0)):
    """Joint targets for one rock, from the demo's grasp style:
    the demo's tip offset and tool tilt are expressed relative to the rock's direction from the base and
    rotated to the new rock. Seeded from the demo grasp joints so the arm uses the same posture."""
    x, y = rock["x_mm"] + shift[0], rock["y_mm"] + shift[1]   # shift: visual aim correction (world mm)
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
                tip=tip.round(1).tolist(), ik_err=round(err, 1), tilt_blend=blend, carry_min_z=round(float(min(tz)), 1),
                pre_tip=(tip - APPROACH_MM * axis), axis=axis, az=float(np.arctan2(y, x)), shift=list(shift),
                rock=rock, calib=calib)


MAX_GRASPS = 3


def _add_shift(plan, dp):
    az = plan["az"]
    return (plan["shift"][0] + dp[0] * np.cos(az) - dp[1] * np.sin(az), plan["shift"][1] + dp[0] * np.sin(az) + dp[1] * np.cos(az))


def _retreat(mv, plan):
    """Back up along the straight descend line to q_pre (never drag the open fingers sideways at the table)."""
    mv.follow(plan["descend"][::-1][1:], peak_deg_s=25)


def pick_place(mv, plan, skill, step=lambda name: None, vis=None, assist=None):
    """Executes a plan. step(name) is called before each phase (the mission uses it for step mode / HUD).
    vis (optional, see Vision) adds the camera checks: aim before closing, hold after lifting.
    Returns dict(held=bool, grip=value, aim=[...], hold=...)."""
    g = skill["gripper"]
    g_open = g["open"] + 8          # a little wider than the demo for slack
    home = mv.q.copy()
    out = {"aim": []}
    step("approach"); mv.move(plan["q_pre"], grip=g_open)
    if vis is not None and vis.jac is None and not Vision.calib_failed:
        step("calibrate wrist cam"); vis.calibrate(mv, plan)
        Vision.calib_failed = vis.jac is None     # don't waste ~6 s per pick retrying
    step("descend");  mv.follow(plan["descend"][1:], grip=g_open, peak_deg_s=25); mv.move(plan["q_grasp"], grip=g_open, settle_s=1.2)
    if vis is not None:
        for k in range(3):
            step("check aim"); a = vis.aim()
            out["aim"].append({k2: a[k2] for k2 in ("between", "c", "w", "dp") if k2 in a})
            if a["ok"]:
                break
            if a.get("dp") is None or k == 2:
                if a["between"] < 0.3:   # sure it's not between the fingers: closing would only push it away
                    print("  aim: object not between the fingers -> skip close (no push)", flush=True)
                    _retreat(mv, plan); mv.gripper(g["close_cmd"] + 6, wait=0.2)
                    mv.move(home); out.update(held=False, grip=None, reason="misaligned"); return out
                break                    # unsure: try the grasp anyway
            step("re-aim"); _retreat(mv, plan)
            plan = vis.replan(plan, a["dp"])
            mv.move(plan["q_pre"], grip=g_open)
            mv.follow(plan["descend"][1:], grip=g_open, peak_deg_s=25); mv.move(plan["q_grasp"], grip=g_open, settle_s=1.2)
    # ---- force-feedback grasp with regrasp search: close until the jaw feels the object; if it closes on air,
    # reopen, back straight up, shift the aim (camera-guided if possible, else a small search pattern), try again
    # run 6: wrist cam showed the object just BEYOND the fingertips (arm stopped short) -> search deeper first
    pattern = [(12.0, 0.0), (12.0, 0.0), (-24.0, 8.0), (0.0, -16.0)]   # (radial, tangential) mm, cumulative shifts
    tang = [(0.0, 8.0), (0.0, -16.0)]
    out["grasps"] = []
    for k in range(MAX_GRASPS):
        step("close (force feedback)")
        ff = mv.grasp_ff(GRIP_CLOSED)
        out["grasps"].append({kk: ff[kk] for kk in ("contact", "held", "pos", "load")} | {"shift": plan.get("shift")})
        viz.text(f"grasp {k + 1}: contact {ff['contact']} load {ff['load']} -> {'FEELS OBJECT' if ff['held'] else 'closed on air'}",
                 "INFO" if ff["held"] else "WARN")
        print(f"  grasp {k + 1}: contact {ff['contact']} pos {ff['pos']} load {ff['load']} -> {'FEELS OBJECT' if ff['held'] else 'closed on air'}", flush=True)
        if ff["held"] or k == MAX_GRASPS - 1:
            break
        step("regrasp"); mv.gripper(g_open, wait=0.3)
        if assist is not None and not out.get("assisted"):
            # two-arm: right gripper stays open at the grasp pose, the left arm slides the object into the jaws
            step("left arm push"); out["assisted"] = bool(assist(plan))
            if out["assisted"]:
                continue                       # close again right here, no re-aim
        dp = pattern[k % len(pattern)]
        if vis is not None:            # look again at grasp height: which way is the object?
            w = vis.where()
            out.setdefault("where", []).append(w)
            if w.get("choice") == "none" and w["p"].get("none", 0) > 0.7:
                print("  object no longer near the fingers (pushed away?) -> give up on this one", flush=True)
                _retreat(mv, plan); mv.move(home, grip=g["close_cmd"] + 6)
                out.update(held=False, grip=ff["pos"], reason="object lost"); return out
            if w["p"].get("beyond", 0) > 0.8:
                dp = (12.0, 0.0)                      # deeper, accumulates with earlier shifts
            elif w.get("choice") in ("left", "right"):
                dp = tang[k % 2]
        _retreat(mv, plan)
        plan = vis.replan(plan, dp) if vis is not None else plan_pick(plan["rock"], skill, plan["calib"], shift=_add_shift(plan, dp))
        mv.move(plan["q_pre"], grip=g_open)
        mv.follow(plan["descend"][1:], grip=g_open, peak_deg_s=25); mv.move(plan["q_grasp"], grip=g_open, settle_s=1.2)
    gm = ff["pos"]
    held = ff["held"]
    step("lift");     mv.follow(plan["lift"][1:], peak_deg_s=25)
    pos, load, cur = mv.grip_sense()                   # still pushing back after lifting = did not slip out
    out["lifted"] = {"pos": round(pos, 1), "load": load, "cur": cur}
    held = held and pos > GRIP_CLOSED + 2.0 and pushback(load, cur)
    print(f"  after lift: pos {pos:.1f} load {load} -> {'still holding' if held else 'empty'}", flush=True)
    if vis is not None and not (not held and pos < GRIP_CLOSED + 3.0):   # jaws fully shut = surely empty, no need to ask
        step("check hold"); h = vis.hold(held)
        out["hold"] = h; held = h["held"]
        print(f"  hold check: {h}", flush=True); viz.text(f"hold check: {h}")
    if held:
        step("carry");  mv.follow(plan["carry"][1:])
        mv.move(plan["drop"], settle_s=0.5)
        step("drop");   mv.gripper(g_open, wait=0.6)
    else:
        mv.gripper(g_open, wait=0.3)
    step("home");     mv.move(home, grip=g["close_cmd"] + 6)  # gripper nearly closed at rest, not squeezing
    out.update(held=held, grip=gm, shift_rt=[round(float(v), 1) for v in world_to_rt(plan["shift"], plan["az"])])
    return out


# ---------------------------------------------------------------- wrist-camera visual aim (Jev boxes)
import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

WRIST_CAM = "/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.3:1.0-video-index0"
JAC_FILE = _Path(__file__).with_name("wrist_jac.json")
JOG_MM = 15.0
MAX_SHIFT_MM = 25.0


class Vision:
    calib_failed = False
    """Image Jacobian between the tip's (radial, tangential) motion in mm and the red object's box centre in the
    wrist image. Measured once by two small jogs at the approach pose (self-calibration), saved to wrist_jac.json,
    scaled by the box width at grasp time (closer = bigger = more pixels per mm)."""

    def __init__(self, skill, calib, rock, grab, on_look=lambda d: None):
        import judge
        self.judge, self.skill, self.calib, self.rock, self.grab, self.on_look = judge, skill, calib, rock, grab, on_look
        self.jac = _json.loads(JAC_FILE.read_text()) if JAC_FILE.exists() else None

    def _look(self, tag):
        img = self.grab()
        try:
            import cv2
            cv2.imwrite(str(_Path(__file__).with_name("snapshots") / time.strftime(f"wrist_%H%M%S_{tag}.jpg")), img)
        except Exception:
            pass
        d = self.judge.wrist_look(img, tag)
        viz.image("ai/wrist_checked", img)
        viz.text(f"jev {tag}: between {d['between']:.2f} box {d.get('c')}")
        viz.value("ai/jev_between", d["between"])
        self.on_look(d)
        return d

    def _ik_move(self, mv, tip, axis):
        q, e = kin.ik(tip, mv.q, down=False, axis=axis)
        if e > 3:
            raise ValueError(f"jog IK error {e:.1f} mm")
        mv.move(q, settle_s=0.6)

    def calibrate(self, mv, plan):
        try:
            self._calibrate(mv, plan)
        except Exception as e:   # camera/network trouble must never stop the pick
            print("  wrist calibration failed (aim correction off):", e, flush=True)

    def _calibrate(self, mv, plan):
        az, axis, p0 = plan["az"], plan["axis"], np.asarray(plan["pre_tip"], float)
        ur, ut = np.array([np.cos(az), np.sin(az), 0]), np.array([-np.sin(az), np.cos(az), 0])
        q0 = mv.q.copy()
        b0 = self._look("jac0")
        try:
            self._ik_move(mv, p0 + JOG_MM * ur, axis); b1 = self._look("jac_r")
            self._ik_move(mv, p0 + JOG_MM * ut, axis); b2 = self._look("jac_t")
        finally:
            mv.move(q0, settle_s=0.6)
        if not all(b.get("box") for b in (b0, b1, b2)):
            print("  wrist calibration: object not visible in every view, aim correction off for this pick", flush=True)
            return
        J = np.column_stack([(np.subtract(b1["c"], b0["c"])) / JOG_MM, (np.subtract(b2["c"], b0["c"])) / JOG_MM])
        if abs(np.linalg.det(J)) < 1e-5:
            print("  wrist calibration: degenerate jacobian", np.round(J, 4), flush=True); return
        self.jac = {"J": J.tolist(), "w_ref": b0["w"]}
        JAC_FILE.write_text(_json.dumps(self.jac, indent=1))
        print(f"  wrist jacobian (img/mm) {np.round(J, 4).tolist()}  box w {b0['w']:.2f}", flush=True)

    def aim(self, at_pre=False):
        try:
            return self._aim(at_pre)
        except Exception as e:
            print("  aim check failed, grasping blind:", e, flush=True)
            return {"ok": True, "between": -1}

    def _aim(self, at_pre=False):
        d = self._look("aim_pre" if at_pre else "aim")
        d["ok"] = d["between"] >= self.judge.SURE_YES and not at_pre
        if not d["ok"] and d.get("box") and self.jac:
            J = np.array(self.jac["J"]) * (d["w"] / self.jac["w_ref"])
            ref = self.jac.get("ref_pre") if at_pre else self.judge.REF_GRASP
            if ref is None:
                return d
            e = np.subtract(ref, d["c"])
            dp = 0.8 * np.linalg.solve(J, e)                 # (radial, tangential) mm
            n = np.linalg.norm(dp)
            if n > MAX_SHIFT_MM: dp *= MAX_SHIFT_MM / n
            d["dp"] = dp.round(1).tolist()
        print(f"  aim: between {d['between']:.2f} box c {np.round(d.get('c', (0, 0)), 2).tolist()} -> "
              f"{'OK' if d['ok'] else 'shift ' + str(d.get('dp'))} ({d['s']} s)", flush=True)
        return d

    def replan(self, plan, dp):
        return plan_pick(self.rock, self.skill, self.calib, shift=_add_shift(plan, dp))

    def where(self):
        img = self.grab()
        try:
            import cv2
            cv2.imwrite(str(_Path(__file__).with_name("snapshots") / time.strftime("wrist_%H%M%S_where.jpg")), img)
            w = self.judge.where(img)
        except Exception as e:
            print("  where check failed:", e, flush=True); return {"choice": None, "p": {}}
        viz.image("ai/wrist_checked", img); viz.text(f"jev where: {w['choice']} {w['p']}")
        print(f"  where: {w['choice']} {({k: round(v, 2) for k, v in w['p'].items()})} ({w['s']} s)", flush=True)
        self.on_look({"between": w["p"].get("between", 0), "s": w["s"]})
        return w

    def hold(self, grip_says_held):
        try:
            h = self.judge.held(self.grab(), grip_says_held)
        except Exception as e:
            h = {"held": grip_says_held, "p": -1, "src": f"servo only ({e})"}
        self.on_look({"held": h})
        return h
