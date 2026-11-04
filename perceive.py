"""Chand eyes: top-cam snapshot -> object list.
Current backend: HSV (plan B). SAM 3 gets added later as the primary backend.
Usage: python perceive.py [image.jpg]   (no arg = grab from top cam)
"""
import sys, json, cv2, numpy as np

TOP_CAM = "/dev/v4l/by-path/platform-3610000.usb-usb-0:4.2:1.0-video-index0"
MIN_AREA = 400          # px, ignores specks
BIN_ROI = (465, 20, 640, 200)  # x0,y0,x1,y1 px of the red bin in the top cam. Nothing in here is ore.
                               # Re-set this if the bin or the camera moves.


def grab(path=TOP_CAM, warm=15):
    c = cv2.VideoCapture(path, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(3, 640); c.set(4, 480)
    f = None
    for _ in range(warm):
        ok, f = c.read()
    c.release()
    if f is None:
        raise RuntimeError("top cam read failed")
    return f


def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 90, 40), (10, 255, 255)) | cv2.inRange(hsv, (165, 90, 40), (180, 255, 255))
    k = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def detect(img):
    objs = []
    cnts, _ = cv2.findContours(red_mask(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x0, y0, x1, y1 = BIN_ROI
    bin_box = (x0, y0, x1 - x0, y1 - y0)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        a = cv2.contourArea(c)
        if a < MIN_AREA:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            continue
        objs.append({"id": f"red_{len(objs)}", "colour": "red", "u": round(cx), "v": round(cy),
                     "angle_deg": round(ang, 1), "size_px": round(max(w, h)), "contour": c})
    return objs, bin_box


def draw(img, objs, bin_box):
    out = img.copy()
    if bin_box:
        x, y, w, h = bin_box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(out, "red_bin", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for o in objs:
        cv2.drawContours(out, [o["contour"]], -1, (0, 255, 0), 2)
        cv2.putText(out, o["id"], (o["u"] + 8, o["v"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else grab()
    objs, bin_box = detect(img)
    cv2.imwrite("snapshots/perceive_last.jpg", draw(img, objs, bin_box))
    print(json.dumps({"red_bin_px": bin_box,
                      "objects": [{k: v for k, v in o.items() if k != "contour"} for o in objs]}, indent=1))


# ---- pixels -> arm mm (calib.json from calibrate.py) ----
def load_calib(path="calib.json"):
    return json.load(open(path))


def to_mm(u, v, calib):
    A = np.asarray(calib["A_px_to_mm"])
    return (A @ np.array([u, v, 1.0])).tolist()


def scene(img, calib):
    """Object list in arm mm, for the brain/executor. Bin centre in mm too."""
    objs, bin_box = detect(img)
    out = []
    for o in objs:
        x, y = to_mm(o["u"], o["v"], calib)
        out.append({"id": o["id"], "colour": o["colour"], "x_mm": round(x), "y_mm": round(y),
                    "angle_deg": o["angle_deg"], "u": o["u"], "v": o["v"]})
    b = None
    if bin_box:
        bx, by, bw, bh = bin_box
        b = [round(c) for c in to_mm(bx + bw / 2, by + bh / 2, calib)]
    return out, b
