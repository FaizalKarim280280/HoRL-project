# Task 1 evaluation

Code and pre-registered plan for the Task 1 comparison: two ACT policies (chunk 50) trained on **all** of v3
(147 episodes) and on a **random half** (73), each run with **40** and **15** executed actions per chunk
(variants `F40 F15 H40 H15`), on matched configurations, compared with STEP.
**Results:** [`../results/task1/README.md`](../results/task1/README.md).

| File | Contents |
|---|---|
| `01_make_schedule.sh` | `make_schedule.py`: samples the configurations (cube + tray target poses in the scene image, near where the demos started), the randomised schedule and the home pose into `plan/` |
| `02_evaluate.sh` | `evaluate.py`: runs the trials (blind, target-guided setup, videos + log) into `../results/task1/` |
| `03_export_results.sh` | `export_results.py`: builds `../results/task1/` (README with all results, tables, figures, manifest) |
| `publish.py` | copy of `../results/task1/` with browser-playable (H.264) videos into `../data/hf/`, for the Hugging Face upload |
| `eval_results.ipynb` | the same analysis, interactively |
| `analysis.py` | success table + Wilson CIs, times, progress, failure stages, STEP pairs, CLD, violins; applies `corrections.yaml` |
| `scene_detect.py` | red cube / blue tray detection in the scene camera |
| `plan/` | `eval_plan.yaml` (pre-registered settings), `configs.csv` + `configs.jpg`, `schedule.csv`, `pilot_schedule.csv`, `home_pose.json`, `scene_reference.jpg`, `half_episodes.json` |

## Running an evaluation

From `A2/`:
1. `eval/01_make_schedule.sh` (refuses to overwrite an existing schedule). Check `plan/configs.jpg`.
2. `PILOT=1 OPERATOR=Name eval/02_evaluate.sh`: practice trials, not counted.
3. Commit `plan/` before the first counted trial (pre-registration).
4. `OPERATOR=Name eval/02_evaluate.sh`: in sessions; quit with `q` on the setup screen, rerun to continue.
   Don't look at results or change anything until all trials are done.
5. Corrections, if any, go in `../results/task1/logs/corrections.yaml` (the raw log is never edited).
   Then `eval/03_export_results.sh`.

The scene camera must stay where it was for the demonstrations: every rollout logs its measured shift
(`camera_shift_px`, warning above 8 px).

## Keys (evaluation window)

The window never shows which variant is running, so outcomes are labelled blind.

| When | Key | Action |
|---|---|---|
| setup | (place) | move the cube and tray onto the dashed target outlines until the bar shows `READY` (within 10 px and 10 deg) |
| | `SPACE` | start (only when `READY`; the measured poses and errors are logged) |
| | `g` | start anyway, e.g. if glare breaks the detection (logged as `setup_override`) |
| | `o` / `h` / `q` | reference photo on/off / back to home pose / quit |
| running | `s` | **success**: cube in the tray, released, arm lifted. Records the completion time |
| | `f` | stop as a **failure** (stuck, cube knocked away, unsafe motion) |
| | `x` | abort as **invalid**: not the policy's fault. Reason typed in the terminal; the trial is repeated |
| | (60 s) | timeout = failure |
| after a failure | `0`–`3` | stages completed: none / reached cube / grasped + lifted / carried to the tray |

Each configuration is one block: place the cube and tray on the targets, run its 4 variants in the scheduled
order, putting them back between rollouts. The arm returns to the home pose before every rollout.
Quit only from the setup screen: a hard stop (Ctrl+C) during a rollout leaves an unlogged `rollouts/<id>/`
folder, which must be renamed before restarting.
