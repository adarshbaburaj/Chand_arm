"""Gripper force-feedback probe: python grip_probe.py [empty|object]
Arm stays where it is (torque holds it). Opens the gripper, then closes it with Mover.grasp_ff and prints the
position/load trace. Run once empty and once with an object held between the jaws by hand."""
import sys, json, time
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from execute import Mover
skill = json.load(open("skill.json")); g = skill["gripper"]
ARM = "left" if "left" in sys.argv else "right"
f = SOFollower(SOFollowerRobotConfig(port={"right": "/dev/ttyFollowerLeft", "left": "/dev/ttyFollowerRight"}[ARM], id="thumbi_" + ARM, max_relative_target=5.0))
f.connect(calibrate=False)
try:
    mv = Mover(f)
    from execute import boost_gripper; print("gripper boost", boost_gripper(f))
    print("idle sense", mv.grip_sense())
    mv.gripper(g["open"] + 8, wait=0.8)
    print("open sense", mv.grip_sense())
    if len(sys.argv) > 1 and sys.argv[1] == "object":
        print("put the object between the jaws now (10 s)", flush=True); time.sleep(10)
    close = float(sys.argv[sys.argv.index("--close") + 1]) if "--close" in sys.argv else g["close_cmd"]
    sq = float(sys.argv[sys.argv.index("--squeeze") + 1]) if "--squeeze" in sys.argv else 8.0
    r = mv.grasp_ff(close, squeeze=sq)
    for t in r["trace"][::3]: print(" cmd %.1f pos %.1f load %.1f cur %d" % t)
    print({k: v for k, v in r.items() if k != "trace"})
    for i in range(5):
        time.sleep(1.0); print(f"after {i + 1} s (try pulling the cube)", mv.grip_sense(), "err", mv.grip_err, "temp", mv.grip_temp(), flush=True)
    mv.gripper(g["close_cmd"] + 6, wait=0.5)
finally:
    f.disconnect()
