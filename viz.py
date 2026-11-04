"""Live Rerun view of a mission: 3 cameras, joint angles (commanded vs measured), gripper force, AI answers.

  viz.start()          # spawns the Rerun viewer window (DISPLAY must be set)
Everything here is best-effort: if Rerun fails, the robot keeps working.
"""
import threading, time
import cv2

try:
    import rerun as rr
except Exception:      # rerun missing: all calls become no-ops
    rr = None

_on = False
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def start():
    """Open the viewer. Call BEFORE the cameras are opened: the spawned viewer process would otherwise inherit
    the open camera file handles and keep the devices busy after we exit."""
    global _on
    if rr is None:
        print("viz: rerun not installed"); return
    try:
        rr.init("chand_mission", spawn=True)
        _on = True
    except Exception as e:
        print("viz: rerun failed:", e)


def stream(cams, hz=10):
    """Stream the always-on cameras at `hz` from a background thread."""
    if _on and cams is not None:
        def loop():
            while _on:
                for name in cams.cams:
                    f = cams.latest(name)
                    if f is not None:
                        image(f"cams/{name}", f, small=True)
                time.sleep(1 / hz)
        threading.Thread(target=loop, daemon=True).start()


def _t():
    rr.set_time("wall", timestamp=time.time())


def image(path, bgr, small=False):
    if not _on: return
    try:
        if small:
            bgr = cv2.resize(bgr, (320, 240))
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        _t(); rr.log(path, rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg"))
    except Exception:
        pass


def joints(arm, q_meas=None, q_cmd=None):
    if not _on: return
    try:
        _t()
        for i, j in enumerate(JOINTS):
            if q_meas is not None: rr.log(f"{arm}/joints/{j}/measured", rr.Scalars(float(q_meas[i])))
            if q_cmd is not None: rr.log(f"{arm}/joints/{j}/commanded", rr.Scalars(float(q_cmd[i])))
    except Exception:
        pass


def gripper(arm, pos=None, load=None, cur=None, cmd=None):
    if not _on: return
    try:
        _t()
        for k, v in (("position", pos), ("load_pct", load), ("current", cur), ("commanded", cmd)):
            if v is not None: rr.log(f"{arm}/gripper/{k}", rr.Scalars(float(v)))
    except Exception:
        pass


def value(path, v):
    if not _on: return
    try:
        _t(); rr.log(path, rr.Scalars(float(v)))
    except Exception:
        pass


def text(msg, level="INFO"):
    if not _on: return
    try:
        _t(); rr.log("events", rr.TextLog(msg, level=level))
    except Exception:
        pass


def boxes(path, boxes_xyxy, labels=None):
    if not _on: return
    try:
        _t(); rr.log(path, rr.Boxes2D(array=boxes_xyxy, array_format=rr.Box2DFormat.XYXY, labels=labels))
    except Exception:
        pass
