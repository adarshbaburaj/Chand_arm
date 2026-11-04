# Chand v2: two-arm, self-checking, plan-and-progress robot (design for review)

Written 2026-11-05 01:30 Dubai. Status: **design, not built yet.** The hackathon video already exists (one arm picking an object into the bin), so v2 aims as high as possible.
Read `CHAND.md` first for the hardware, the files and what already works. This document is meant to be reviewed by other AI assistants (ChatGPT etc.). Please challenge any point you think is wrong.

---

## 1. Goal

The operator says one sentence ("put all the red stuff in the bin, leave the white stuff") and two SO-101 follower arms do it:

- they never act blind;
- they *know* whether each step worked (touch, camera and a planner that tracks progress);
- they recover from failure: re-aim, regrasp, use the second arm as a helper, or hand the object over;
- they get better over time, because every attempt is logged as training data with subtask labels.

## 2. What we have (facts, checked 2026-11-05)

| Thing | State |
|---|---|
| Right follower (`thumbi_right`, on `/dev/ttyFollowerLeft` in the Chand setup, top socket) | Works. Calibrated to the top camera (11 touch points, affine, LOO 7.4 mm). |
| Left follower (`thumbi_left`) | Its controller board was replaced (it died in late October). The user will connect it **about 50 cm from the right follower**. Not calibrated to the camera yet. |
| Cameras | top (`usb-0:4.2`), right wrist (`usb-0:4.1.3`), left wrist (`usb-0:4.1.4`) |
| Gripper force signal | STS3215 `Present_Load` + position lag. Probe: an empty gripper closes to 13.1 at about 3 % load with no false contact. The value with an object is still to be measured. |
| Meta API | `muse-spark-1.3` (planner, about 4-9 s), `muse-wonder-1.0` fallback, also `sam-3.1` and `muse-image-1.0` |
| TypeSafe Jev | typed yes/no, choice, score and box answers on images, about 0.35-0.55 s, cloud |
| Muse Glimmer 30B (local GGUF, llama.cpp CUDA build) | text works. **Vision needs the mmproj file (not downloaded).** |
| VLAs on disk | `hqfang/pi05-so100_101` (π0.5 for SO-100/101, **best zero-shot so far**: picked the cube, dropped it after about 2 s, jittery, about 1.4 s per 50-step chunk on the Orin), `MolmoAct2-SO100_101` (2 copies, never run on the arm), `smolvla_so101_pickandplace` (useless zero-shot), `Cosmos-Reason1-7B` (VLM planner only). See `~/SO101_handoff.md`. |
| Scale AI FineART | Dataset: 40,543 episodes, 1,718 h, **every episode labelled with subtasks** (about 13 each), on a **Trossen ALOHA bimanual robot (14-dim)**, *not SO-101*. Model `lerobot/FineART-VLA-subtask-midtrained`: built on π0.5, **predicts the next subtask in language, then acts on it**, 9.35 GB, bimanual ALOHA joint space. |

### What FineART can and cannot do for us (honest)
- ❌ Its actions cannot drive our arms directly. The robot, the joints, the cameras and the action space are all different. Zero-shot on SO-101 would be meaningless.
- ✅ **Format:** use its *method* and *data format*. We record our own episodes in LeRobot v3 format with the same `language_persistent` subtask timeline, so a later fine-tune of FineART-VLA (or π0.5 SO-100/101) on **bimanual SO-101** data needs no new tooling.
- ✅ **Vocabulary:** use its *subtask vocabulary* as the template for our planner's step language ("pick up the red cube with the right arm", "hand the cube to the left arm", ...). We can mine `meta/tasks.parquet` plus a few episodes' subtask lists and feed them to the planner as few-shot examples.
- ✅ **Initialisation:** for the later fine-tune (cloud GPU), start from FineART-VLA, which has bimanual priors and subtask prediction, *or* from π0.5 SO-100/101, which has SO-101 priors. Try both. (Open question for the reviewer: which is the better starting point?)

### The "π0.5 + planner that tracks progress" idea (from the video the user saw)
The video's pattern: the policy predicts **task progress**, and a planner (they called it "Astra"; I could not find that exact project) moves the **plan** to the next step once the current one is done. π0.5 itself predicts the semantic subtask first and then the action chunk. FineART-VLA does the same. We copy this structure with the parts we have, and make every layer checkable.

---

## 3. Architecture: four layers, each one checkable

```
 S3  PLANNER        Muse Spark (cloud, ~5 s, rare)
     instruction + scene graph + per-arm success maps  →  PLAN = ordered subtasks in language
     + structured skill calls; replans when S2 reports a failure it cannot fix
        │  subtask i: {"skill":"pick","obj":"red_2","arm":"left","helper":"right:fence", "done_when":"..."}
        ▼
 S2  PROGRESS MONITOR   Jev (cloud, ~0.4 s, every phase)  |  Glimmer-vision (local, offline fallback)
     typed questions about camera frames + force numbers:
       score   "progress of this subtask: not started / approaching / in gripper / lifted / over bin / released"
       noul    "subtask done?"  "object between fingers?"  "object still on table?"
       box     where the object is in the wrist / top image (for re-aim)
     unsure (0.3-0.7) or disagrees with the force sensor → asks Spark with the image (rare)
        │  decides: advance to i+1 / retry step / run recovery / escalate to S3
        ▼
 S1  SKILLS (executors)
     scripted, reliable (built from the one teach demo + IK):  pick_ff · place · push · fence · handover · park
     learned, optional:  π0.5 SO-100/101 as skill "vla(<subtask text>)" for things the scripts can't do;
                         S2 watches its progress and stops it if the score stops rising for N s
        ▼
 S0  MOTION + SAFETY  (50 Hz, never an AI model)
     min-jerk, gravity-sag bias, 5° clip per command, force-limited gripper, shared-zone mutex,
     both arms' poses checked for collisions before every move
```

Rule: **models decide *what* and *whether*, code decides *how*.** No model sits in the 50 Hz loop.
The exception is the VLA skill, which only runs inside a guarded sandbox: a speed limit, a workspace box and the S2 progress watchdog.

## 4. Two arms, 50 cm apart

### 4.1 One shared table frame
- Touch-calibrate **each** arm to the top camera, using the existing `calibrate.py` with `--arm left|right`. That gives each arm a px→mm map and a table z.
- The base-to-base transform follows from the two maps. Check it: 3 points touched by both arms should agree to within 10 mm.
- Each arm reaches about 90-375 mm. The bases are 500 mm apart, so the **overlap zone** is the strip about 125-375 mm from each base. Handovers and assists happen there.

### 4.2 Choosing the arm
For each object, score each arm:
`score = success_map[arm](x,y) · reach_margin · (1 - ik_error/6) · free_path`
- `success_map` is a coarse grid (40 mm cells) of past outcomes per arm. It starts from a prior (the calibrated zone counts as good, the edges as poor) and is **updated after every attempt**. This is the bot "getting smarter" without any training.
- The planner (Spark) receives these scores, so its choice is explainable ("left arm: 0.82, right arm: 0.41, because the right arm missed twice here").

### 4.3 Two-arm skills (the "hardest, best outcome" part)
1. **fence / backstop:** the helper arm lowers its *open* gripper on the far side of the object (a wall 15 mm off the table). The worker arm's closing jaw then cannot push the object away, which is the exact failure we saw in runs 4 and 5.
2. **push to the sweet spot:** if an object is in the worker's low-success zone, the helper pushes it (gripper closed, a straight line at 8 mm height) into the cell where the worker's success rate is highest.
3. **handover:** the object can only be reached by arm A, but only arm B reaches the bin (or A failed twice). A picks it with force feedback and carries it to the handover point in the overlap zone. B closes on it and feels contact. A opens, and its load drops to about 0, which confirms B has it. B then places it.
4. **parallel picking:** if the objects sit in non-overlapping zones, both arms pick at the same time. The overlap zone is a mutex: only one arm at a time.
5. **re-grip / reorient (stretch):** pass an object between the grippers to change which way it is held.

### 4.4 Collision safety
- Before every move: forward kinematics of both arms → capsules around the links → check the minimum distance along the path (> 40 mm).
- Mutex for the overlap zone. The idle arm parks outside it.
- A watchdog: if any joint's tracking error is above 8° for more than 0.3 s, stop both arms (collision or stall).

## 5. "Make sure the grasp happened": checking a grasp in 4 ways

| # | Signal | Time | Says |
|---|---|---|---|
| 1 | **force:** while closing, the jaw stops following the command or the load rises → contact. After settling, pushback load > threshold and position > closed + 2° | in the loop | something is in the jaws |
| 2 | **lift + shake:** lift 20 mm, rotate the wrist ±5°, the load stays up | 1 s | it is held firmly and won't slip |
| 3 | **wrist camera (Jev):** "the object is pinched between the fingers, lifted off the table" | 0.4 s | it is *the right* object |
| 4 | **top camera (Jev or HSV):** the object is gone from its spot | 0.5 s | it really left the table |

`held` = 1 ∧ 2 ∧ (3 ∨ 4). If 1 or 2 is true and 3 is unsure → Spark looks at the image.

### Recovery ladder (S2 goes down it, cheapest first)
1. re-aim using the wrist-camera box (Jev) → regrasp (already built, up to 3 tries)
2. a small search pattern (±8 mm tangential and radial) (already built)
3. the helper arm builds a **fence**, then regrasp
4. the helper **pushes** the object into the worker's best cell, then regrasp
5. switch arms (the other arm picks it), or do a handover
6. Spark replans with images and the full attempt history
7. give up on this object, report it, and keep going with the others

## 6. Getting smarter with data (the FineART path)

Every attempt is logged automatically as a **LeRobot v3 episode**:
- 3 cameras, 12-dim state and action (2 × 5 joints + gripper), gripper load
- `language_persistent` subtask timeline, written **automatically** by the state machine (the planner's subtask text + the start time of each phase), in the FineART format
- success metadata from the 4-way check

This gives us, for free, a growing **bimanual SO-101 dataset with subtask labels**. It feeds:
1. **now:** the success maps (§4.2)
2. **soon:** grasp-offset correction per table region (offline regression: where we aimed versus where the force sensor felt contact)
3. **later (cloud GPU, hours):** fine-tune **FineART-VLA** or **π0.5 SO-100/101** on this data, so the learned skill predicts *our* subtasks and progress. The hierarchy stays the same; the scripted S1 skills are replaced one by one only after the learned version beats them in A/B runs.

Data rules: keep failures, since they are the most valuable data. Never train on episodes where the safety stop fired.

## 7. Build order (each step has a pass test)

| Step | Build | Pass test |
|---|---|---|
| A | `grip_probe.py object` → set the force thresholds | held vs empty clearly separated (load gap > 3×) |
| B | Left arm bring-up: port, id `thumbi_left`, teleop with the left leader | smooth teleop |
| C | `calibrate.py --arm left` + shared frame check | 3 shared points agree within 10 mm |
| D | `arms.py`: two Movers, shared-zone mutex, capsule collision check, park poses | both arms move at the same time without coming closer than 40 mm |
| E | Skills: `fence`, `push`, `handover` (scripted, IK) | each one 3/3 on the cube |
| F | `progress.py` (S2): Jev score + noul per phase, recovery ladder | a deliberately offset grasp recovers by itself |
| G | `planner.py` (S3): Spark plan with arm choice + helper skills, success maps | 4 red objects across the whole table, both arms used, 0 human help |
| H | Episode logger in LeRobot v3 format with the subtask timeline | `LeRobotDataset(root=...)` loads it, subtasks show |
| I | Optional: π0.5 SO-100/101 as the guarded `vla(...)` skill | the watchdog stops it when progress stalls |
| J | Later: cloud fine-tune of FineART-VLA or π0.5 on our episodes | beats the scripted pick in an A/B test |

## 8. Open questions for the reviewer
1. For a 2×SO-101 fine-tune, is FineART-VLA (bimanual ALOHA priors) or π0.5 SO-100/101 (single-arm SO priors) the better starting point?
2. Is a coarse grid of success rates enough, or should we fit a small model (e.g. Gaussian process) on (x, y, arm) → success?
3. Is Jev's ordinal "score" question a reliable enough progress signal, or should progress come from the force and joint state (cheap, deterministic) with vision only as a cross-check?
4. Can the handover be checked by force alone (the giver's load falls while the taker's rises), or does it need vision too?
5. Is the fence skill (the helper's open gripper as a wall) physically sound for a 30 mm cube on a smooth mat, or will the cube slide along the fence?

## 9. Not doing
- No AI model in the 50 Hz loop.
- No zero-shot FineART-VLA on SO-101 (wrong robot).
- No cloud model in scored Malta runs (Malta rule). Chand v2 is a separate project; the offline fallbacks are Glimmer-vision and the scripted skills.
