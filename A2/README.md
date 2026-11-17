# A2: learning pick and place from demonstrations (ACT)

Same task as before: pick up the red cube and place it in the blue tray. This time the arm learns it from demonstrations recorded with the leader arm, using the **wrist camera** and a fixed **scene camera**.

- **Recording** uses our own fast recorder. It saves raw JPEGs and joint data, with no video encoding while recording.
- **Conversion** happens once, afterwards, into a standard LeRobot dataset for training ACT, which is the next step.

Ports and arm ids come from `../common/robot.sh`. Everything else is set in [`scripts/config.sh`](scripts/config.sh). All scripts and settings are in `scripts/`; run them from the `A2` folder (e.g. `scripts/02_record.sh`) with the `lerobot` conda env active.

## 1. Set up the cameras (once)

1. Plug in both cameras and fix the scene camera rigidly. It must see the whole workspace: every place where the cube and the tray will be, and the arm.
2. Run `scripts/00_find_cameras.sh` to list the cameras. In `scripts/config.sh`, `WRIST_CAMERA` (`1080p`, set in `../common/robot.sh`) and `SCENE_CAMERA` (`720P`) pick them by part of their name.
3. Run `scripts/01_view_cameras.sh` to see both cameras live, side by side (`g` grid, `s` snapshot, `q` quit). Aim the scene camera so the cube, the tray and the gripper are always in view.

**Don't move the scene camera** after you start recording. The policy learns from that exact viewpoint.

## 2. Record demonstrations

```bash
scripts/02_record.sh          # record until you quit
scripts/02_record.sh 10       # stop after 10 saved demonstrations
```

1. **The window:** it shows both cameras, and the follower copies the leader the whole time. At startup the follower glides to the leader's pose over 2 s.
2. **Starting and stopping is manual**, with keys in the camera window:

| Key | Action |
|---|---|
| `SPACE` | start a demonstration; press again to stop and **save** it |
| `d` | discard the demonstration being recorded |
| `[` / `]` | previous / next initial-configuration ID (which taped cube and tray position you're demonstrating) |
| `q` or `ESC` | quit (all images are finished writing first) |

3. **Between demonstrations** you're idle: reset the cube and tray, move the leader to the start pose, then press `SPACE`.
4. **Saving is instant.** Images are written in the background as you record.
5. **Several sessions are fine.** Running it again adds episodes to the same dataset, in `A2/data/raw/so101_pick_place/`.
6. **When you quit, the follower goes limp.** Support it, or quit with it resting.
7. **Every attempt is logged** in `data/raw/<name>/demos.csv`, as required for the submission:
   - the episode, its status (saved, discarded, too short, or quit), the start time, duration and frames;
   - the measured loop rate and the number of frames over the 33 ms budget;
   - the configuration ID and the operator.

   The `notes` column is for you to fill in. Set who is demonstrating and the starting configuration with `OPERATOR=Faizal CONFIG=3 scripts/02_record.sh`.
8. **End every demonstration the same way:** release the cube in the tray, lift the gripper slightly, then press `SPACE`. The success criterion requires the release and the lift, so the policy has to see them.

## 3. Convert to a LeRobot dataset (once, after recording)

```bash
scripts/03_convert.sh                # raw -> A2/data/lerobot/so101_pick_place/
scripts/03_convert.sh --overwrite    # rebuild after recording more episodes
```
This encodes the videos. It takes about 30 minutes for 50 demonstrations of about 20 s, and runs unattended.

## 4. Check the data

```bash
scripts/06_dataset_summary.sh       # episodes, frames, lengths (raw and converted)
scripts/04_view_episode.sh 3        # play episode 3 of the converted dataset (videos + joint plots)
scripts/05_replay_episode.sh 3      # replay episode 3 on the follower (the arm moves)
```

## 5. Merge batches, train, run a policy

```bash
scripts/08_merge.sh so101_pick_place_all so101_pick_place so101_pick_place_b2   # merge converted datasets
```
Training runs on a GPU machine (`lerobot-train`, dataset via the Hugging Face Hub); checkpoints come back into
`outputs/train/`. To run one on the arm, with a camera window:
```bash
N_ACTION_STEPS=40 scripts/07_run_policy.sh outputs/train/<run>/<step>/pretrained_model
```
Keys in the window: `SPACE` start / pause, `r` back to the demos' start pose, `q` quit. Options (environment
variables): `N_ACTION_STEPS` (default half the chunk), `MAX_STEP_DEG` (arm step limit, default 0.75),
`MAX_STEP_GRIPPER` (default none), `ENSEMBLE`, `INTERP`, `DURATION`.

## 6. Task 1 evaluation

Code, plan and keys: [`eval/README.md`](eval/README.md). Results, with every reported table and figure:
[`results/task1/README.md`](results/task1/README.md).

## Tips for good demonstrations

- **Record 50+ episodes.** Spread the cube and tray over the workspace: for example 5 cube positions × 2 tray positions, about 5 episodes each, and vary the cube's rotation.
- **Be consistent.** Use the same strategy every time: approach from above, grasp, lift, move over the tray, lower, open, return to the start pose. ACT learns the average of what you show it, so mixing strategies blurs it.
- **Start and end in the same rest pose.**
- **Move smoothly, and not too fast.** Hesitations and corrections get learned too.
- **Throw away bad episodes** with `d` rather than keeping them.
- **Keep the lighting and the background the same** as when the policy will run.
- **A good check:** you should be able to do the task yourself by looking only at the two camera images.

## Data layout

Everything under `A2/data/` is kept out of git.

**Raw recording**, in `data/raw/<name>/`:
```
meta.json                      fps, joint names, cameras, image size, task
episode_000/wrist/000000.jpg   one JPEG per frame and camera (640x480)
episode_000/scene/000000.jpg
episode_000/data.npz           state (N,6) follower joints, action (N,6) leader joints, timestamp (N,)
demos.csv                      log of every recording attempt
```

**Converted dataset**, in `data/lerobot/<name>/`: a standard LeRobot dataset (v3) with exactly the features `lerobot-record` would produce:
- `observation.images.wrist` and `observation.images.scene` (videos);
- `observation.state` and `action` (6 joints each);
- the task text.
