"""Chand judge: fast yes/no questions from the cameras, answered by TypeSafe Jev (~0.4 s, cloud).
Muse Spark is asked only when Jev is unsure or disagrees with the servo reading.

  wrist_look(img)            -> where the red object is in the wrist camera, is it between the fingers
  held(img, grip_says_held)  -> fused verdict: servo reading + Jev vision (+ Spark tie-break)
  is_target(top_img, rock)   -> is this detection really a red object to pick (not bin / arm / shadow)

Every call is logged to logs/api_usage.jsonl. Jev sees only camera frames and numbers, nothing personal.
"""
import base64, json, os, time
from pathlib import Path
import cv2
import httpx

HERE = Path(__file__).parent
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
MAX_CALLS = int(os.environ.get("CHAND_MAX_JEV_CALLS", "200"))
SURE_YES, SURE_NO = 0.7, 0.3          # noul thresholds; in between = unsure -> Spark
REF_GRASP = (0.565, 0.57)            # cube centre in the wrist image at the demo grasp (normalised x, y)
_calls = 0
last = {}                            # last answers, for the HUD


def _key():
    for line in (HERE / ".env").read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("TYPESAFE_API_KEY missing in .env")


def _b64(img, max_w=640):
    if img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])))
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


def ask(state, questions, img=None, tag=""):
    """One Jev request. img (BGR) is referenced in state as IMG1. Returns the answers dict.
    Failures raise (a failed call must never be read as 'no')."""
    global _calls
    if _calls >= MAX_CALLS:
        raise RuntimeError(f"Jev call cap reached ({MAX_CALLS})")
    _calls += 1
    headers = {"Authorization": f"Bearer {_key()}"}
    body = {"model": MODEL, "state": state, "questions": questions}
    if img is not None:
        headers["TypeSafe-Danger-Zone-Images"] = "true"
        body["assets"] = {"IMG1": {"type": "image", "format": "jpg", "data_base64": _b64(img)}}
    t0 = time.time()
    r = httpx.post(URL, headers=headers, json=body, timeout=20)
    dt = round(time.time() - t0, 2)
    d = r.json()
    (HERE / "logs").mkdir(exist_ok=True)
    with open(HERE / "logs" / "api_usage.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": "typesafe", "model": d.get("model", MODEL),
                            "tag": tag, "status": r.status_code, "s": dt, "usage": d.get("usage", {})}) + "\n")
    if r.status_code != 200:
        raise RuntimeError(f"Jev {r.status_code}: {str(d)[:200]}")
    d["answers"]["_s"] = dt
    return d["answers"]


# ------------------------------------------------------------------ wrist camera questions
WRIST_Q = {
    "cube": {"type": "box", "instructions": "the red object nearest to the camera, between or in front of the white gripper "
                                            "fingers (not the dark red bin)"},
    "between": {"type": "noul", "instructions": "A red object is centred in the gap between the two open white gripper "
                                                "fingers, so closing the fingers would grip it."},
}


def wrist_look(img, tag="wrist"):
    a = ask("Wrist camera image from a robot gripper: IMG1", WRIST_Q, img, tag)
    b = a["cube"].get("box")
    out = {"between": a["between"]["noul"], "box_conf": a["cube"].get("confidence"), "s": a["_s"], "box": None}
    if b and (b["right"] - b["left"]) > 0.06 and (a["cube"].get("confidence") or 0) >= 0.35:
        out["box"] = b
        out["c"] = ((b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2)
        out["w"] = b["right"] - b["left"]
    last["wrist"] = out
    return out


WHERE_Q = {"where": {"type": "choice", "instructions": "Wrist camera of an open robot gripper about to close. The two white "
                     "fingers with rubber bands at their tips are at the bottom-left and bottom-right. Where is the red object "
                     "relative to the fingers?",
                     "criteria": {"between": "The red object sits down between the two fingers, level with the rubber-band tips "
                                             "or closer to the camera; closing would grip it.",
                                  "beyond": "The red object is further away than the fingertips: it appears above the "
                                            "rubber-band tips in the image, so the gripper must move forward.",
                                  "left": "The red object is mostly left of the gap between the fingers.",
                                  "right": "The red object is mostly right of the gap between the fingers.",
                                  "none": "No red object near the fingers."}}}


def where(img, tag="where"):
    """After a grasp closed on air: which way is the object? (choice + probabilities). Biased toward 'beyond'
    (even good grasps score ~0.6 beyond), so callers compare against the miss pattern, not argmax alone."""
    a = ask("Wrist camera image: IMG1", WHERE_Q, img, tag)["where"]
    out = {"choice": a.get("choice"), "p": a.get("probabilities") or {}, "s": a["_s"]}
    last["where"] = out
    return out


HELD_Q = {"held": {"type": "noul",
                   "instructions": "The gripper is holding a red object: it is pinched between the two white gripper "
                                   "fingers, close to the camera, lifted off the table.",
                   "criteria": {"true": "A red object fills the gap between the two white fingers.",
                                "false": "The gap between the fingers is empty, or a red object is only visible further "
                                         "away on the table."}}}


def held(img, grip_says_held, spark=True):
    """Fuse the servo reading with Jev. Returns dict(held, p, src)."""
    p = ask("Wrist camera image from a robot gripper, just after lifting: IMG1", HELD_Q, img, "held")["held"]["noul"]
    if grip_says_held and p >= SURE_NO or (not grip_says_held and p <= SURE_YES):
        v = {"held": grip_says_held, "p": p, "src": "servo+jev agree" if (p >= SURE_YES) == grip_says_held else "servo (jev unsure)"}
    elif spark:
        s = spark_yes_no(img, "This is the wrist camera of a robot gripper that just tried to pick up a small red object "
                              "and lift it. Is the red object held between the gripper fingers?")
        v = {"held": s, "p": p, "src": f"spark tie-break (servo {'held' if grip_says_held else 'empty'}, jev {p:.2f})"}
    else:
        v = {"held": grip_says_held, "p": p, "src": "servo (jev disagrees, no spark)"}
    last["held"] = v
    return v


# ------------------------------------------------------------------ top camera: is the detection a real target?
def is_target(top_img, rock, rule_text, pad=70):
    u, v = int(rock["u"]), int(rock["v"])
    h, w = top_img.shape[:2]
    crop = top_img[max(0, v - pad):min(h, v + pad), max(0, u - pad):min(w, u + pad)]
    crop = cv2.resize(crop, (280, 280), interpolation=cv2.INTER_CUBIC)
    p = ask({"rule": rule_text, "image": "Close-up from the top camera, centred on a detected object: IMG1"},
            {"target": {"type": "noul", "instructions": "The object at the centre of the image is a loose red object on "
                                                        "the table that the rule says to pick (not the red bin, not part "
                                                        "of the robot, not a shadow)."}}, crop, "target")["target"]["noul"]
    last["target"] = p
    return p


# ------------------------------------------------------------------ Spark cross-check (expensive, rare)
def spark_yes_no(img, question):
    import brain
    data = "data:image/jpeg;base64," + _b64(img)
    reply = brain.chat([{"role": "user", "content": [
        {"type": "text", "text": question + ' Reply with ONLY {"answer": "yes"} or {"answer": "no"}.'},
        {"type": "image_url", "image_url": {"url": data}}]}], tag="crosscheck", reasoning_effort="low")
    ans = brain.parse_json(reply).get("answer", "").lower() == "yes"
    last["spark"] = ans
    return ans


if __name__ == "__main__":   # quick self-test on saved frames: python judge.py a.jpg b.jpg
    import sys
    for f in sys.argv[1:]:
        im = cv2.imread(f)
        print(f, wrist_look(im), held(im, False, spark=False))
