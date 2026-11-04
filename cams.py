"""Always-on cameras: one thread per camera keeps the latest frame, so nobody re-opens a device (~0.5-1 s each
time) and a live viewer can show all three while the robot works.

  cams = Cams.start()            # top, right wrist, left wrist (missing devices are skipped)
  img = cams.fresh("top")        # waits for a frame captured AFTER this call (never a stale one)
"""
import threading, time
from pathlib import Path
import cv2

PATHS = {
    "top": "/dev/v4l/by-path/platform-3610000.usb-usb-0:4.2:1.0-video-index0",
    "right_wrist": "/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.3:1.0-video-index0",
    "left_wrist": "/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.4:1.0-video-index0",
}
_active = None


class _Cam(threading.Thread):
    def __init__(self, name, path):
        super().__init__(daemon=True)
        self.name, self.path = name, path
        self.frame, self.t, self.ok = None, 0.0, False
        self.cond = threading.Condition()
        self.running = True

    def run(self):
        c = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(3, 640); c.set(4, 480); c.set(cv2.CAP_PROP_FPS, 30)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ok, f = c.read()
            if not ok:
                time.sleep(0.05); continue
            with self.cond:
                self.frame, self.t, self.ok = f, time.time(), True
                self.cond.notify_all()
        c.release()


class Cams:
    def __init__(self, names):
        self.cams = {n: _Cam(n, PATHS[n]) for n in names if Path(PATHS[n]).exists()}

    @classmethod
    def start(cls, names=("top", "right_wrist", "left_wrist")):
        global _active
        if _active is None:
            _active = cls(names)
            for c in _active.cams.values():
                c.start()
            t0 = time.time()
            while time.time() - t0 < 5 and not all(c.ok for c in _active.cams.values()):
                time.sleep(0.05)
        return _active

    def latest(self, name):
        c = self.cams.get(name)
        return None if c is None else c.frame

    def fresh(self, name, settle_frames=2, timeout=2.0):
        """A frame captured after this call (skip `settle_frames` to let auto-exposure/motion blur settle)."""
        c = self.cams[name]
        t_call, n = time.time(), 0
        with c.cond:
            while True:
                if not c.cond.wait(timeout):
                    raise RuntimeError(f"camera {name} timed out")
                if c.t > t_call:
                    n += 1
                    if n > settle_frames:
                        return c.frame.copy()

    def stop(self):
        global _active
        for c in self.cams.values():
            c.running = False
        for c in self.cams.values():
            c.join(timeout=1.0)
        _active = None


def active():
    return _active


def by_path(path):
    """Name of an always-on camera for a device path (perceive.grab uses this), else None."""
    if _active is None:
        return None
    for n, p in PATHS.items():
        if p == path and n in _active.cams:
            return n
    return None
