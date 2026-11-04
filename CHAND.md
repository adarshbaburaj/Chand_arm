# Chand — Meta hackathon side project (read this first)

Created 2026-11-04 18:58 Dubai. **Submit by 05:30 Dubai on 2026-11-05 (hard deadline 06:00).** That is about 10.5 h from creation.
This is a side project, separate from the Malta challenge. Malta context lives in `~/SO101_handoff.md`.
Project folder: `~/chand/`. All new code goes here. Nothing in `~/so101_*` or `~/workspaces/lerobot_v061` gets edited.

**Next-version design (two arms, progress monitor, VLA/FineART data path): see `PLAN_V2.md`.**

## FOR ANY AI ASSISTANT PICKING THIS UP (kept current, 2026-11-05 01:15 Dubai)
Several AI assistants help build this. Read this section, then `## User requirements`, then the code. Run everything in
`source ~/workspaces/lerobot_v061_env/bin/activate` (torch 2.11.0 + CUDA must stay; check `python -c "import torch;print(torch.__version__, torch.cuda.is_available())"` before/after any install).
Secrets live in `.env` (gitignored, chmod 600): `META_API_KEY`, `TYPESAFE_API_KEY`. Never commit or print them.
Commits carry the user's name only (no Co-Authored-By lines).

### Current architecture (what actually runs)
```
top cam ─▶ perceive.py (HSV red, bin box ignored, affine px→mm from calib.json)
            │
            ▼
  Muse Spark (brain.py, cloud, ~4-9 s)  ── picks WHICH object (JSON, validated; failed-twice rocks excluded)
            │
  Jev (judge.py, TypeSafe cloud, ~0.4 s) ── fast typed yes/no + boxes from camera frames:
            │     is_target (top crop) · wrist_look "between the fingers?" + box · held (after lift)
            │     unsure (0.3-0.7) or disagrees with the servo → Muse Spark cross-check with the image
            ▼
  execute.py  plan_pick: demo grasp style rotated to the new rock, IK (kin.py), straight-line descend/lift
              Mover: min-jerk 50 Hz, gravity-sag integral bias, every command clipped to 5° (max_relative_target)
              grasp_ff: FORCE FEEDBACK — close slowly, contact = jaw lags the command / load rises (STS3215
                        Present_Load), squeeze only 2.5° past contact, held = pushback after settling and after lift
              regrasp: closed on air → reopen, look (Jev box → shift via wrist image jacobian) or search pattern
                       (±8 mm tangential/radial), back up straight, re-descend; up to 3 grasps per object
            ▼
  verify: fresh top-cam look, object gone from its spot + held → "mined"
```
Low-level control is NEVER an AI model (needs 50 Hz). Models decide what/whether; code decides how.

### Commands
- Mission: `DISPLAY=:1 python mission.py --backend meta_api --max-picks 6` (window starts PAUSED, SPACE start, Q = safe abort home). `--no-judge` turns Jev off, `--offline` blocks all network except localhost.
- Gripper force probe: `python grip_probe.py empty` / `python grip_probe.py object` (object held in the jaws by hand).
- Judge self-test on saved frames: `python judge.py a.jpg b.jpg`.
- Logs: `logs/m5_runN.log`, `logs/mission_HHMMSS.json` (per-pick aim/grasps/force/hold), `logs/api_usage.jsonl` (every Spark + Jev call). Wrist frames: `snapshots/wrist_*`.

### Known problems / open work (01:15)
- Grasps miss (gripper closes to 13.1 = air) mostly on the LEFT side of the table (weak calibration area) — run 4: 1 mined, 5 missed; run 5: Jev said "between fingers 0.85" yet closed on air. Force-feedback + regrasp search added to fix this, not yet tested on the arm.
- ROOT CAUSE of many misses (01:50): gripper.pos is 0-100 % of range and the old close command 12.8 (copied from the demo) left a gap — thin objects slipped through. Empty jaws close freely to ~0.9. Now grasp_ff closes toward 0 and stops on contact. Force probe: holding a 30 mm cube → contact at 17.2, load 15.6 %, current 11; empty → load 3-6 %, current 0-2. Thresholds HOLD_LOAD 10 %, HOLD_CUR 6.
- Grip strength (01:20): LeRobot's configure() sets gripper Overload_Torque 25 % → servo trips 'Overload' after ~1-2 s of a firm squeeze (happened at squeeze 10 and 5). `execute.boost_gripper()` (called after every connect) raises Max_Torque_Limit/Torque_Limit 800, Overload_Torque 80, Protection_Time 250; over-current (250) and 70 °C protections kept; squeeze drops to 3 above 55 °C. Probe: squeeze 8 → load 31 %, current ~50, 5 s hold, no trip, 37 °C.
- Wrist-image jacobian self-calibration (2 jogs at the approach pose) failed in run 5: object not visible from the approach pose. Aim correction then falls back to the search pattern.
- Left follower (thumbi_left, /dev/ttyFollowerRight — cables are swapped vs the original naming) is not used yet. Idea: second arm as a backstop/fence or to push objects into the well-calibrated zone. Needs its own touch calibration.
- M6 (Glimmer local offline: needs the mmproj vision file), M7 (earth_lag + HUD), video: not done.

## User requirements (log of what the user asked for, newest first — keep appending)
- 04:30 — Live (real-time) tuning of the PID controllers; arm motion too rigid → smoother, including the PID servo; pause it if it risks crashing the project.
- 04:15 — New concept: from the wrist camera field, approximate how near the object is / whether it WILL be grasped: a mask ("capture zone") around the grippers (tip + centre) where a grasp is sure, and adjust the arm smoothly until the object is in it — PID + Jev together.
- 04:05 — Arms did well, only the left arm didn't grab well. Show the COST IN DOLLARS of each run. Run again.
- 03:55 — Left still couldn't grasp even with wide jaws → next attempt must CHANGE THE ANGLE OF ATTACK and move the wrist a little higher (z). Right arm must NOT stay above the bin after a drop: go back to home, but don't turn the wrist roll back or fold the wrist camera away from the box — body to rest, camera keeps watching.
- 03:35 — When the right arm can't grasp and the left comes to help, the RIGHT arm must back off a few cm first (no interference). No going home while ANY red object is inside the black box: if the right is done, it waits (backed off) for the left to bring objects near it. Grasp 1 fails → grasp 2 IMMEDIATELY, wider and deeper; still no luck → back off and let the left assist.
- 03:00 — Top camera RAISED so the whole tape rectangle + bin are visible → chose SELF-CALIBRATION (arm places a cube, camera finds it). 03:20: left arm re-picks came up short ("grasped a little in front of the dropped object; deeper would be better, like the right").
- 02:50 — Left and right arm "did the same movement" (each must do its own job, truly in parallel, no confusion; rebuild if needed). When a grasp failed the arm didn't go deeper or open the gripper wider — fix.
- 02:28 — Run 8 went really well; now SPEED: (1) drops always in the CENTRE of the red bin (answer: hand-teach the centre once); (2) the left arm picks the far-left item AT THE SAME TIME as the right arm works on its own object; (3) no going back to rest after every drop — actively scan from over the bin (answer), rest only when nothing is left inside the BLACK LINES, then KEEP WATCHING for new objects (answer); ask when in doubt.
- 01:55 — Left follower calibrated (10 points, right leader driving it). NEW RULE: when the right arm can't reach an object, the LEFT arm picks it and drops it near the right arm (then the right arm bins it). Wanted a tool to "paint" each arm's whole reachable area with the gripper tip as the brush; then: no camera in that tool, just a canvas + brush + colours, one arm at a time.
- 01:35 — Cube picks work perfectly; other shapes (pentagon) fail. Asked: (1) use the LEFT arm to push the object into the right arm's gripper; (2) is there a VLA that runs efficiently on the Orin AGX without issues; (3) show live camera feeds of all 3 cams when a test run starts — use **Rerun** (best data graphs); (4) what to do with the left follower, can the **right leader drive the left follower**; (5) if a grasp misses, go deeper next time — ideally grasp deep from the start; (6) what is slowing the whole process.
- 01:18 — Gripping is not strong enough; asked how to remove the limit that cuts the motor out when it grasps.
- 01:45 — Left follower plugged in (on /dev/ttyFollowerRight, socket 4.3, calibration thumbi_left matches the motors; left leader NOT connected). Did the cube-in-gripper force test on the right arm.
- 01:30 — Hackathon video already recorded (one arm, pick + drop) → now build **as ambitious as possible**. Use Scale's FineART data to make the bot smarter; copy the "π0.5 VLA + planner where the policy predicts task progress and the planner executes the next step" pattern; reuse the VLAs already on disk; **left follower will be connected ~50 cm from the right follower**; use the APIs and logic properly; write ONE best design to a .md for review with ChatGPT → **`PLAN_V2.md`**. Set up `gh` login so tokens are never pasted again.
- 01:10 — Grasping is not right. Add **force feedback from the gripper**: once the object is in the gripper it feels push-back, and that tells the bot the grab happened. If not, the bot should try the best way to grab without failing. Optionally bring in the **left follower arm** to assist the right one. "Don't just choose the easiest, try the hardest and get the best outcome." Look at Scale's FineART (labs.scale.com/blog/fineart: dataset + FineART-VLA, System 2 subtask reasoning / System 1 actions) for ideas so the system *ensures* the grasp happened. **Keep GitHub code and this .md updated with everything the user says, other AI models will help build.**
- 00:45 — Use **TypeSafe Jev** (typesafe.ai, typed yes/no, ~0.4 s) as the fast judge: "is that what we want to grab? did we grab it?" Use Meta's smartest model (Muse Spark) only when needed / to cross-check Jev. Idea: Glimmer local for fast checks, Spark for high-level.
- 00:30 — Explain how the whole pick process works (for the video).
- 23:40 — Pick ALL the red stuff into the bin.
- 21:40 — Push all Chand files to https://github.com/adarshbaburaj/Chand_arm.git, commits with only the user's name.
- 21:00 — Motion must be smooth (not shaky).

## What we are building
A SO-101 arm on the Orin "mines iron ore" (red 3D-printed shapes) into the red bin with **zero training**.
Story: an astronaut teaches it **once** from Earth (one leader-arm demo), then the rover works alone on the Moon, fully offline.

- **Earth mode (the problem):** leader drives the follower through a 2.6 s delay buffer. Clumsy on purpose.
- **Teach once (Earth, internet OK):** one leader demo of picking a red object into the red bin. The Muse Spark API reads the demo and writes the rule ("red rocks go in the bin, white stay").
- **Moon mode (offline):** top cam + SAM 3 find rocks. Muse Glimmer (local) picks the next target. The executor adapts the demo motion to the new rock, the verifier checks, and it loops until no red rocks remain.
- **Wow shot:** unplug the ethernet mid-mission and it keeps mining.

## Rules carried over from the SO101 handoff (still apply)
1. **The Jetson torch must not be replaced.** The env is `~/workspaces/lerobot_v061_env` (py3.12, self-built torch 2.11.0 sm_87, CUDA 12.6). Before and after every install, run
   `~/workspaces/lerobot_v061_env/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → must print `2.11.0 True`.
   Install with `--no-deps` or a constraints file. Never pull `nvidia-*` pip wheels.
2. **No motion, torque or calibration unless the user is present.**
3. Use the arm names and aliases exactly (below). Never use `ttyCH341USBn` or `videoN`.
4. Never read the motors while a lerobot process holds the port. Kill by pid, not `pkill -f`.
5. **Gripper overload lesson:** the STS3215 gripper trips "Overload error" (Overload_Torque 25) if it squeezes for a long time. Command the close to just past the object width, not to 0, and open before any long wait.
6. Log every check/test in `~/so101_daily/checks_log.md` (prefix `[CHAND]`) and keep this file current.
7. No duplicate scripts. One file per job, as listed below.

## Hardware (facts from the rig, not placeholders)
- Orin AGX 64 GB, JetPack 6.2.3 / L4T R36.5.2, CUDA 12.6, MAXN already set. 748 GB free on the NVMe, 61 GB RAM.
- **Use the RIGHT arm.** Calibration files `thumbi_right` (follower + leader) already exist in `~/.cache/huggingface/lerobot/calibration/`.

| Role | Path | Note |
|---|---|---|
| right follower | **`/dev/ttyFollowerLeft`** (top socket 1-4.4) | Plugged into the TOP socket. Verified 19:10 by reading Homing_Offset: all 6 match `thumbi_right`. So for Chand use port `/dev/ttyFollowerLeft` + id `thumbi_right`. |
| right leader | `/dev/ttyLeaderRight` (cable 4, 1-1.3) | present |
| top cam | `/dev/v4l/by-path/platform-3610000.usb-usb-0:4.2:1.0-video-index0` | present (640x480@30 MJPG) |
| right wrist cam | `/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.3:1.0-video-index0` | present |
| left follower | — | not plugged in |

`lerobot-find-port` / `find-cameras` are not needed. The udev aliases are fixed per socket.

## Hard rules for this project
1. Never run SAM and Glimmer at the same time. Perceive, then think, then move.
2. Glimmer is never in the motion loop. One call per decision.
3. Glimmer gets the scene as **text** (an object list in mm), not images.
4. Take snapshots only with the arm parked out of the top cam's view.
5. Safety: low max joint speed (`max_relative_target` ≤ 10), workspace limits checked before every move, and a software e-stop.
6. Moon mode makes **zero** network calls. `--offline` sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and patches `socket.socket.connect` so that anything other than 127.0.0.1 raises.
7. One arm working end to end before the second arm.

## Architecture (one engine)
`instruction → brain (task list JSON) → executor (grasp pose, IK, motion) → verifier → loop`
Eyes (top cam + SAM → objects in arm mm) feed the brain, executor and verifier.

The brain may ONLY output:
```json
{"skill": "pick", "obj": "<id>", "arm": "right"}
{"skill": "place_in", "target": "red_bin"}
{"skill": "done"}
```

## Files (all in ~/chand/)
| File | Job |
|---|---|
| `config.yaml` | ports/cams (the aliases above), speed limits, bin drop pose (joints), park pose (joints), workspace box, backend |
| `calibrate.py` | Touch calibration: jog the tip onto taped crosses with the leader, record FK (x, y), click the same crosses in the top image, `cv2.findHomography` → `calib.json`. Print the reprojection error. |
| `kin.py` | FK/IK for SO-101 (URDF from SO-ARM100). LeRobot degrees → URDF radians. Top-down grasp, wrist roll = object angle. |
| `perceive.py` | snapshot → SAM 3 ("red object", "white object") → masks → HSV check → centroid + `minAreaRect` angle → homography → `[{id, colour, x_mm, y_mm, angle_deg, size_mm}]`. Fallback: pure HSV. |
| `teach.py` | Record ONE demo at 30 Hz (joints, gripper, top frames). Split at gripper close/open. Grasp style = FK at grasp vs the rock seen in frame 0. Keep carry + drop joints. Muse Spark → rule JSON. Save `skill.json`. |
| `brain.py` | `plan(instruction, scene, rule)` → task list. Backends `meta_api` and `glimmer_local` (llama.cpp server, OpenAI-compatible). Strict JSON validation, one retry, then a plain rule fallback. |
| `execute.py` | IK for the new grasp = rock pose + grasp style. Blend the demo-vs-new joint offset in over the approach and out over the carry. Hover 60 mm → descend → close → lift → carry → drop → park. |
| `verify.py` | Holding check from the gripper position. Placement check from a fresh snapshot. Up to 2 retries. |
| `hud.py` | Live window: feed, mask outlines, object list, state, mined counter, mission timer. |
| `mission.py` | Moon-mode loop, `--offline`, `--backend glimmer_local\|meta_api`. |
| `earth_lag.py` | leader → follower through a 2.6 s ring buffer, with the HUD showing the delay. |

## Milestones (each one's test must pass before moving on)
- **M0** Setup: torch check, Glimmer GGUF downloading in tmux, llama.cpp built with CUDA (sm_87), SAM weights downloaded.
- **M1** Right-arm teleop works. Top cam fixed, exposure and white balance locked.
- **M2** `perceive.py` finds every red object in < 2 s and draws outlines.
- **M3** Touch test: the arm hovers 20 mm above 5 random rocks, each within ~10 mm. **Make or break. Do not continue until it passes.**
- **M4** `teach.py`: one demo recorded, `skill.json` + rule JSON saved.
- **M5** Moon mode with `meta_api`: 3 red rocks into the bin, two runs in a row, white rocks untouched.
- **M6** Switch to `glimmer_local`, run with the ethernet unplugged.
- **M7** `earth_lag.py` + HUD polish.
- **M8** Stretch: left arm. Probably skip (see the changes below).

## Model notes (checked on the Hub 2026-11-04)
- **Muse Glimmer 30B:** `unsloth/Muse-Glimmer-30B-GGUF`, UD-Q4_K_XL (~18 GB). Also `ggml-org/Muse-Glimmer-30B-GGUF`. Run with `llama-server` on 127.0.0.1.
- **SAM:** `facebook/sam3.1` is a raw **checkpoint** (needs Meta's sam3 repo code, which is a new install). `facebook/sam3` is in **transformers format**, and the env already has transformers 5.5.4 with `Sam3Model`. → Use `facebook/sam3` (no install). Both may be gated: accept the licence on the HF site and run `hf auth login`.
- **Muse Spark:** Meta Model API, key in `~/chand/.env` as `META_API_KEY` (chmod 600; doesn't exist yet). Used only for the teach step.

## Current state of the env (checked 18:57)
- torch 2.11.0, CUDA True. transformers 5.5.4 (Sam3 present). httpx, requests, PyYAML and pynput are present.
- Missing: `ikpy`, `scipy`, `placo` (LeRobot's own `model/kinematics.py` needs placo), llama.cpp, system cmake (only `~/workspaces/lerobot_env/bin/cmake` exists), the SAM and Glimmer weights.

## Changes I suggest to the original brief
1. **SAM 3 instead of 3.1** (above). Same idea, no new code stack.
2. **Write the kinematics in numpy (DLS IK, about 80 lines) instead of ikpy.** ikpy pulls scipy and sympy into the torch env, and placo has no reliable aarch64 wheel. The 5-joint chain is small. The joint-frame conversion we already proved with π0.5/MolmoAct2 is signs `[1,-1,1,1,1,1]`, offsets `[0,90,90,0,0,0]` deg (verify against the URDF zero pose before trusting it).
3. **Calibrate with 6–9 crosses, not 4.** 4 points fit a homography exactly, so errors are invisible. More points give a reprojection error to check against M3's 10 mm. Grasp at rock-top height: the homography is for the table plane, so tall rocks shift a few mm off-centre in the image (correct using the known rock height and the camera position).
4. **Top-down grasp limits reach.** A 5-DoF SO-101 with the gripper pointing straight down reaches only about 120–280 mm from the base. Put the rocks and bin inside that ring, or allow a pitch angle of up to 30° off vertical at the edge.
5. **E-stop via the HUD window key (cv2.waitKey), not a terminal space bar.** pynput needs focus or X. Torque-off makes the arm fall, so the e-stop should first send "hold current position", and turn torque off on a second press (or keep a hand on the 12 V plug as before).
6. **Placement check = rock seen inside the bin ROI, or gone from the table with the gripper open over the bin.** "Gone from the table" alone also passes if the rock fell on the floor.
7. **Skip M8 (left arm).** The left follower runs on the left leader's board, so there is no left leader for a demo. With 10 h, one polished arm wins more.
8. **Plan B for the time limit:** if SAM is slow or gated, run M2–M5 on HSV only (red vs white on a dark mat is easy) and add SAM afterwards. M3 (calibration accuracy) is the real risk. Start it early.
9. Use the right **wrist cam** for the holding check if the gripper position is ambiguous on thin shapes.

## Decisions (19:10)
- **Brain backend: Muse Spark API first** (user). Glimmer keeps downloading in the background for the offline wow shot (M6).
- **API key budget:** the user gives the key when asked. Every API call is logged (count + tokens) to `logs/api_usage.jsonl`. Hard cap per run (`max_api_calls` in config.yaml, default 20). Never call in a loop without the user knowing. A dry-run mode prints the prompt instead of sending it.
- Cameras: use top (planning/verify) + right wrist (holding check / close-up).
- Everything on the table is within reach (user).
- Submission = **video + public GitHub repo** `https://github.com/adarshbaburaj/chand_meta.git`. Local repo `~/chand` (git init done, remote set, nothing pushed). Models in `~/chand_models` (outside the repo).

## Scene from the top cam (19:04, `snapshots/top_1904.jpg`)
- White mat with a black tape border. The arm base sits bottom-right, partly in frame.
- **Red bin** top-right (tall, open top). A white tray at top-left and a black box at top-centre are distractors.
- Red ore: cylinder, cube, pentagon prism, hexagon prism (4). White: cube, pentagon prism, cylinder lying down (3).
- Notes for perception: (1) the bin is red too → mask the fixed bin ROI out of the ore search. (2) White on a white mat has low contrast → HSV can't find white rocks; SAM is needed for them (they only need to be "left alone", so red-only HSV is fine as plan B). (3) Slight pink colour cast → lock white balance/exposure.
- The wrist cam image is rotated ~90° (cam mounted sideways). It sees a red hex prism + the bin.

## Status 20:05
- HF login done (user adarshababurj). `META_API_KEY` in `.env` (600). **API WORKS: base `https://api.meta.ai/v1` (OpenAI-compatible, also api.ai.meta.com). Use `muse-spark-1.3` (~4 s, reasoning model: set max_tokens ≥ 600 or content is null). Fallback `muse-wonder-1.0` (~6 s, fewer reasoning tokens). Also listed: sam-3.1, muse-image-1.0, muse-voice-transcribe-1.0, spark 1.1/1.2. api.llama.com gives 401.**
- Glimmer GGUF downloaded (15 GB). llama.cpp built (`~/workspaces/llama.cpp/build/bin`).
- `perceive.py` HSV v1: 4/4 red rocks + bin box in 0.2 s (`snapshots/perceive_last.jpg`). White rocks not detected yet (SAM step). Pixel coords only, so mm needs calib.json (M3).

## Running jobs (old, done)
- Glimmer download: `~/chand_models/glimmer/Muse-Glimmer-30B-UD-Q4_K_XL.gguf` (15.9 GB, WiFi ~8 MB/s), log `logs/dl_glimmer.log`.
- llama.cpp (commit 5851a8e) CUDA sm_87 build: `~/workspaces/llama.cpp/build`, log `logs/llama_build.log`.
- SAM 3 (`facebook/sam3`, 3.4 GB): waiting for the user's HF login + licence accept.

## Open questions for the user
1. ~~Follower port~~ answered: right follower in the top socket. (Old note: at 18:57 `/dev/ttyFollowerLeft` (socket 1-4.4, cable 1) is present but `/dev/ttyFollowerRight` is not. Is that the left follower, or did the right follower's cable go into the top socket? To use the right arm, plug its USB into the socket below (1-4.3) and add the 12 V.
2. Is the ethernet wired now (needed for the 18 GB Glimmer download), and is your HF token set up? Have you already been granted access to SAM?
3. Do you have the `META_API_KEY` and the Muse Spark endpoint/model name?
4. What mat colour is under the rocks, and how big are the red/white shapes (size and height)?
5. What does the submission need: a video, a repo, a live demo?

## Log
- 2026-11-04 18:58 — File created. Read-only checks only (devices, env, Hub). No motors touched.
- 2026-11-04 19:05 — Project renamed from Lunar Prospector to **Chand** (folder ~/chand, this file CHAND.md).
- 2026-11-04 20:05 — HF ok, API key saved (401 on llama.com), Glimmer + llama.cpp done, perceive.py HSV v1 works.
- 2026-11-04 20:15 — Meta API found at api.meta.ai/v1; muse-spark-1.3 and muse-wonder-1.0 both return valid JSON.
- 2026-11-04 21:50 — **M3 PASSED.** calib 11 pts affine (LOO 7.4 mm), table_z -28.1 mm. IK seeded from nearest taught pose (45° tilt). settle() integral bias fixes gravity sag (up to 10° on elbow/lift). 4/4 rocks centred in wrist cam at 35 mm hover. Next: M4 teach.py.
- 2026-11-04 21:22 — **M4 done.** skill.json from demo_211726 + Spark rule ("red → red_bin, leave white, stop when no red"). Use `reasoning_effort="low"` + max_tokens ≥ 4000 on every Muse call (else reasoning eats all tokens, content null). Smooth motion lives in execute.py (Mover). perceive.BIN_ROI fixed box. Next: M5 execute pick-place loop.
- 2026-11-04 21:40 — M5 runs 3/4: mined 2 then 1; gripper ramp fix (single close command was clipped to 5°).
- 2026-11-05 00:50 — Jev layer (judge.py): target check, aim check before closing, hold check after lift, Spark tie-break. Run 5: 0 mined (aim said OK, closed on air).
- 2026-11-05 01:15 — Force-feedback grasp (grasp_ff, Present_Load) + regrasp search (up to 3), grip_probe.py, wrist frames saved, abort no longer drives the shoulder past home. Empty-gripper probe: no false contact, settles 13.1 / 3.2 % load.
- 2026-11-05 01:27 — Run 6 (force grip + Jev): pick 1 MINED (contact 16.8, load 31 %, servo+Jev agree). Pick 2 (pentagon): 3 closes on air — wrist frames show the object just BEYOND the fingertips (arm stopped short), Jev noul "between" falsely 0.8. User aborted. Fixes: regrasp search goes deeper first (+12 mm radial, cumulative), Jev choice question (between/beyond/left/right/none) steers the direction after an air close, no vision/Spark check when the jaws shut fully (surely empty), wrist jacobian calibration tried once per mission only. NOTE: log times written 00:50-01:25 were estimates and some ran ahead of the real clock.
- 2026-11-05 01:45 — Built: `cams.py` (always-on top/right-wrist/left-wrist threads, fresh frame 0.15 s instead of re-opening the device 0.5-1 s), `viz.py` (Rerun viewer: 3 cams, joints cmd vs measured, gripper pos/load/current, Jev answers, events; spawn BEFORE opening cameras or the viewer process inherits the camera handles), shape classes in perceive (square/pentagon/hexagon/round/long), `grasp_memory.json` (per-shape start offset; default +12 mm deeper for non-squares; EMA of what worked), Spark plans the ORDER of all objects once per mission (cached by position, re-asked after a failure; 2.9 s), `calibrate.py --arm left` (right leader drives left follower → calib_left.json), `assist.py` (left arm pushes object into the right gripper; hooked into the regrasp ladder, not yet wired to the mission — needs calib_left.json).
- What was slow (measured): Spark 3-11 s per pick (now once per mission), camera re-open 0.5-1 s per photo x ~4 per pick (now 0.15 s), wrist-jacobian calibration ~6 s per pick (now once), a 7.4 s Spark tie-break when the force sensor already knew (removed). Still slow on purpose: descend 25°/s, settle 0.8-1.2 s per move, gripper close 20°/s, Jev ~0.4 s per question.
- VLA on the Orin (facts from ~/SO101_handoff.md benchmarks): ACT 56 ms per call (fast, needs our own demos), SmolVLA ~0.8 s per 50-step chunk (OK with async inference), π0.5 SO-100/101 ~1.4 s per chunk (best zero-shot: grabbed the cube then dropped it after 2 s, jittery). None works reliably zero-shot on our setup; all need fine-tuning on our own demos (Orin overnight for ACT/SmolVLA, cloud GPU for π0.5 / FineART-VLA). Plan: the auto-logged mission attempts become that dataset (PLAN_V2 §6).
- 2026-11-05 01:58 — Left calibration: 10 pts, mean 6.1 mm, LOO 9.9 mm; bases 473 mm apart (left base at (-33, 435) in the right frame). `paint_workspace.py --arm X` (torque off, hand-guided, canvas coloured by tip height; `--rebuild` regenerates masks from the saved points): workspace_<arm>.png = whole painted envelope in camera px (used for reach), workspace_<arm>_table.png = table-level part. `two_arm.py`: can_pick(arm) = mask + real grasp plan; handoff_spot = pixel both arms reach (with ±15 mm margin for the right), closest to the right arm's demo grasp spot, cached in handoff.json; left_handoff = left force-feedback pick → put down at the spot. mission.py: two-arm mode auto (`--one-arm` to disable): reach tag per object (right / left / none), Spark told which arm reaches what, left-only objects get handed over, and the left arm push-assist runs after the right closes on air.
- 2026-11-05 02:15 — Left workspace painted (556 pts). FIX: plan_pick seeded the pan the wrong way (pan = -azimuth, checked with FK); harmless for the right arm near the demo direction, but the left arm's grasp plans all failed (IK error ~228 mm, 14 s each). After the fix both arms plan across the shared area. Handoff spot = camera px (305, 305) = right (173, 213) mm r 274 / left (219, -225) mm r 313 (both kept inside 320 mm). Reach maps overlap 78k camera px.
- 2026-11-05 02:21 — **Run 8, first two-arm run: mined 4 in 271 s, 2 left→right handoffs, EVERY grasp succeeded on the first try** (force contact 16.6-23.4, load ~31 %, servo + Jev agree each time). Shapes: square, hexagon (left arm), pentagon, square. One object at (375, 104) mm was outside both arms' reach and was skipped. Spark planned in 4.5-8.7 s per pick because only one object was left each time → now skipped when only one target remains.
- 2026-11-05 02:45 — Parallel mission (mission.py rewritten; sequential version = commit f4bccac): Tk + dispatcher in the main thread (~4 looks/s), each arm's job in its own thread. Right arm: picks → carry ends 45 mm above the TAUGHT bin centre (teach_bin.py → bin_drop.json, 6 cm from the old demo drop) → straight down → force check before release (= "mined") → stays above the bin (scan pose); rests only after 5 s with no target, keeps watching. Left arm: left-only objects → handoff spot, at the same time. Shared area (overlap of the painted maps) = ZONE lock, left holds it until it is home again; handoff spot must be free (top cam). Red blobs within 80 px of a busy gripper are ignored (objects being carried). Work area = between the two black tape lines (work_area.json, fitted lines + 6 px inset). Spark orders targets in a background thread when new objects appear; the dispatcher never waits for it.
- 2026-11-05 02:55 — Run 9 (first parallel run): right mined 1; left missed a long cylinder 3x. CAUSE of "same movement": both arms ran the first-time wrist-camera jacobian calibration (identical +15 mm jog dance) at the same moment and both wrote ONE shared wrist_jac.json. Fixes: per-arm state (wrist_jac_<arm>.json, per-arm calib_failed, per-arm snapshot names, lock on grasp_memory writes); judge.where() crashed ('_s') so the miss direction was never read — fixed; each regrasp now opens the jaws wider (31 → 45 → 58 %) and goes deeper (+15, +30 mm); long objects: wrist roll chosen so the jaws close ACROSS the object (hinge = gripper_frame y axis from the URDF), 1-10° error in planning. Threads are fine for parallel motion (separate serial buses, I/O releases the GIL); CUDA is not involved in motion control.
- 2026-11-05 03:25 — New camera: work_area.json = tape-rectangle corner polygon + bin box (old view kept as *_oldcam.json). `selfcal.py --arm X`: arm places a held cube at 10 spots (from its painted workspace, restricted to the old on-table view), steps out, camera finds it, re-picks it; first (hand-gripped) placement dropped from the fit. Right: 9 pts, LOO 11.6 mm, re-picks 10/10. Left: 9 pts, LOO 12.6 mm, re-picks 6/10 (all misses short) → LEFT_EXTRA_DEPTH_MM = 10 in mission. Bases 467 mm apart (consistent). Reach maps rebuilt from the saved painted mm points (no repaint), handoff spot recomputed: px (335, 335), r 286 / 319 mm.
- 2026-11-05 03:45 — Run 11 (parallel, new camera): mined 2, 2 handoffs; one right job died on a lost serial status packet (gripper id 6) → Mover now retries reads/writes. New grasp ladder in pick_place: grasp 2 immediately (jaws 45 %, slide 15 mm deeper at grasp height, no backing out); after 2 misses the right backs off ~6 cm + 4 cm up, releases the shared area, and the LEFT arm (if free + in reach) carries the object to the handoff spot; the right then picks it there at once (known position). No assist possible → grasp 3 (58 %, Jev direction). Left-only handoffs queue a direct right pick at the spot. The right rests only when no workable red object is left AND the left arm is idle; otherwise it waits above the bin. The old push assist (assist.py) is no longer used by the mission.
- 2026-11-05 04:00 — Run 12: right mined 1 (grasp 2 "slide deeper" fired), left missed the hexagon 3x with Jev "beyond" each time → LEFT_EXTRA_DEPTH_MM 18. Grasp ladder now 4 steps: 1 normal → 2 immediately wider (45 %) + 15 mm deeper in place → (right only: back off ~6 cm + 4 cm up, left assist) → 3 STEEPER ANGLE OF ATTACK (tilt blend ≥ 0.5: tool ~20° from vertical instead of ~40°) + 8 mm higher, jaws 58 % → 4 normal angle, Jev direction. plan_pick(tilt_min, dz) kept through replans. Right "home" between picks = rest pan/lift/elbow with wrist_flex 20° and the drop roll (home_watch), so the wrist camera watches the table; the true rest only at mission end.
- 2026-11-05 04:10 — Run 13: right mined 1, left closed on air twice (wrist cam: object a full width beyond the tips). `cost.py` + live "cost so far" line in the mission window + RUN COST at the end + cost in logs/mission_*.json. Prices: muse-spark $1.25 / $4.25 per M tokens (in/out incl. reasoning; third-party listings of Meta's rate sheet), Jev $0.042 per M input. Past runs $0.003-0.02 each; all logged AI calls so far $0.16. Pre-close depth servo: Jev 'where' before closing, 'beyond' > 0.8 → slide 15 mm deeper in place (up to 3x); grasp memory per arm (left:shape keys); LEFT_EXTRA_DEPTH_MM 30; wrist jacobian calibration removed.
- 2026-11-05 04:25 — Run 14: left handed 1, right mined 1, then several misses where Jev 'between' said 0.79-0.94 but the jaws closed on air (wrist frame: object far left of the fingers) → Jev alone is bad at depth from one 2D image. Built `capture.py`: per-arm capture zone TAUGHT automatically (arm sets a cube down, opens the jaws without moving, photographs = the sure-grasp picture, 3 spots averaged → capture_<arm>.json); `Vision.servo`: local red-blob tracking (~10 Hz) → error of the object's top-centre to the zone → PI (KP 0.6, KI 0.15) through the per-arm wrist jacobian → ≤4 mm IK nudges (Mover.nudge, 0.12 s each) at grasp height, max 45 mm; then Jev referee on the frame with the zone drawn as a green box. Used before every close when the arm has a capture zone (else the old Jev 'where' loop).
- 2026-11-05 04:35 — Capture zone right taught (3 spots, spread 7/11 px, tol 32/27 px). `tune.py`: slider window → tune.json, read live by the mission (tune.get, re-read ≤0.2 s, always clamped to a safe range, defaults if missing): servo kp/ki/kd/max step/period, peak speed, approach-line speed, min move time, sag KI. Servo is now continuous: newest frame without waiting (cams.latest), PID on the image error, each nudge flows into the next; P/I/D terms plotted in Rerun. Softer default motion: peak 34 deg/s (was 40), min move 0.8 s (was 0.6).

