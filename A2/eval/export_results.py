#!/usr/bin/env python
"""Build the Task 1 results folder (A2/results/task1) from the evaluation logs.

Reads plan/ and results/task1/logs/ (raw eval_log.csv + corrections.yaml) and writes, in results/task1/:
    README.md       every result the assignment asks for, explained, with figures
    logs/           manifest.csv (one row per rollout: policy, config, label, partial score, timing, video),
                    demos_batch{1,2,3}.csv (demonstration logs), the raw logs and corrections (already there)
    tables/         results, STEP, failure stages, failures, factors, per-config outcomes, data checks, sensitivity
    figures/        violins (posterior + CLD), progress, completion times, per-config outcomes, setup quality, configs
    plan/           copy of the pre-registered plan, configurations and schedule
    checkpoints/    config.json + train_config.json of the two evaluated checkpoints (weights stay in outputs/)
Rerun after any change to the logs or corrections; everything here is regenerated.
"""

import importlib.metadata
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import analysis as an  # noqa: E402
import publish  # noqa: E402  (Hub repo name)

HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan"
OUT = an.RESULTS
DEMOS = {"batch1": "so101_pick_place", "batch2": "so101_pick_place_b2", "batch3": "so101_pick_place_b3"}


def version(p: str) -> str:
    try:
        return importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:
        return "n/a"


def save(df: pd.DataFrame, name: str, index: bool = True) -> str:
    df.to_csv(OUT / "tables" / f"{name}.csv", index=index)
    return an.md_table(df, index=index)


def manifest(log: pd.DataFrame) -> pd.DataFrame:
    """One row per rollout (counted, invalid, excluded and pilot), linking it to its policy, config, label and video."""
    pilot = pd.read_csv(OUT / "logs" / "pilot_log.csv").assign(included=False)
    m = pd.concat([log, pilot], ignore_index=True)
    m["analysed"] = m.included.fillna(False).astype(bool)
    m["category"] = np.select([m.status == "pilot", m.status == "invalid", ~m.analysed], ["pilot", "invalid (repeated)", "excluded"], "analysed")
    m["video_wrist"] = [f"rollouts/{r}/wrist.mp4" if (OUT / "rollouts" / r).is_dir() else "missing" for r in m.rollout_id]
    m["video_scene"] = [f"rollouts/{r}/scene.mp4" if (OUT / "rollouts" / r).is_dir() else "missing" for r in m.rollout_id]
    # Online copies: H.264 videos in the results repo (play in the browser)
    base = f"https://huggingface.co/datasets/{publish.RESULTS_REPO}/resolve/main"
    for cam in ("wrist", "scene"):
        m[f"url_{cam}"] = [f"{base}/{v}" if v != "missing" else "" for v in m[f"video_{cam}"]]
    cols = ["rollout_id", "category", "trial", "block", "config_id", "variant", "policy", "n_action_steps", "success",
            "stages_completed", "progress", "end_reason", "failure_stage", "time_to_success_s", "duration_s", "start_time",
            "cube_err_px", "tray_err_px", "setup_override", "notes", "video_wrist", "video_scene", "url_wrist",
            "url_scene", "checkpoint"]
    return m[cols]


def sensitivity(plan: dict) -> pd.DataFrame:
    """STEP decisions with and without each post-hoc correction."""
    corr = yaml.safe_load((OUT / "logs" / "corrections.yaml").read_text())
    cases = {"both corrections (reported)": corr, "no corrections (raw log)": {},
             "relabel only": {"relabel": corr["relabel"]}, "exclusion only": {"exclude_trials": corr["exclude_trials"]}}
    rows = []
    for name, c in cases.items():
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(c, f)
        _, _, seqs, _ = an.load(corrections_path=Path(f.name))
        Path(f.name).unlink()
        pairs = an.step_pairwise(plan, seqs)
        rates = {v: d.success.mean() for v, d in seqs.items()}
        rows.append({"case": name, **{v: f"{int(d.success.sum())}/{len(d)}" for v, d in seqs.items()},
                     "pairs separated": ", ".join(f"{r.favoured} > {r.B if r.favoured == r.A else r.A}"
                                                  for r in pairs.itertuples() if r.favoured != "inconclusive") or "none",
                     "CLD": " ".join(f"{v}:{x}" for v, x in an.cld_letters(list(seqs), rates, pairs).items())})
    return pd.DataFrame(rows).set_index("case")


def figures(plan: dict, seqs: dict, pc: pd.DataFrame) -> None:
    fig_dir = OUT / "figures"
    names = list(seqs)
    colours = dict(zip(names, ["#2b6cb0", "#63a4e0", "#c05621", "#f0a060"]))
    # 1. success rate with 90% Wilson interval
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (v, d) in enumerate(seqs.items()):
        s_, n_ = int(d.success.sum()), len(d)
        lo, hi = an.wilson(s_, n_)
        ax.bar(i, s_ / n_, color=colours[v], width=0.6)
        ax.errorbar(i, s_ / n_, yerr=[[s_ / n_ - lo], [hi - s_ / n_]], color="k", capsize=6, lw=1.5)
        ax.text(i, hi + 0.03, f"{s_}/{n_}", ha="center", fontsize=10)
    ax.set_xticks(range(len(names)), names)
    ax.set(ylim=(0, 1), ylabel="success rate", title="Success rate with 90% Wilson interval")
    fig.tight_layout(); fig.savefig(fig_dir / "success_rates.png", dpi=150); plt.close(fig)
    # 2. completion time of successes; end time and reason of failures
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True)
    times = [d.loc[d.success == 1, "time_to_success_s"].dropna().astype(float) for d in seqs.values()]
    ax[0].boxplot(times, tick_labels=[f"{v}\n(n={len(t)})" for v, t in zip(names, times)], showfliers=False)
    for i, t in enumerate(times, 1):
        ax[0].scatter(np.random.default_rng(i).normal(i, 0.06, len(t)), t, s=16, alpha=0.75, color=colours[names[i - 1]])
    ax[0].set(ylabel="seconds", title="Successful trials: completion time")
    for i, (v, d) in enumerate(seqs.items(), 1):
        f = d[d.success == 0]
        for reason, marker, col in (("stopped", "o", "#555555"), ("timeout", "x", "#d62728")):
            y = f.loc[f.end_reason == reason, "duration_s"].astype(float)
            ax[1].scatter(np.random.default_rng(10 + i).normal(i, 0.07, len(y)), y, marker=marker, s=22, color=col, alpha=0.75,
                          label=("stopped by operator" if reason == "stopped" else f"timeout ({plan['timeout_s']} s)") if i == 1 else None)
    ax[1].axhline(plan["timeout_s"], color="#d62728", ls="--", lw=1)
    ax[1].set_xticks(range(1, len(names) + 1), [f"{v}\n(n={int((d.success == 0).sum())})" for v, d in seqs.items()])
    ax[1].set(title="Failed trials: when and why they ended")
    ax[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(); fig.savefig(fig_dir / "completion_times.png", dpi=150); plt.close(fig)
    # 4. STEP pairwise decisions
    pairs = an.step_pairwise(plan, seqs)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    grid = np.full((len(names), len(names)), np.nan)
    for r in pairs.to_dict("records"):
        a, b = names.index(r["A"]), names.index(r["B"])
        for row, col, me in ((a, b, r["A"]), (b, a, r["B"])):
            grid[row, col] = 0.5 if r["favoured"] == "inconclusive" else (1.0 if r["favoured"] == me else 0.0)
            txt = "inconclusive" if r["favoured"] == "inconclusive" else ("better" if r["favoured"] == me else "worse")
            if r["favoured"] != "inconclusive":
                txt += f"\n(trial {int(r['trials per policy at decision'])})"
            ax.text(col, row, txt, ha="center", va="center", fontsize=8,
                    color="black" if r["favoured"] == "inconclusive" else "white")
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names); ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="compared with", ylabel="variant", title="STEP pairwise decisions (row vs column)")
    fig.tight_layout(); fig.savefig(fig_dir / "step_matrix.png", dpi=150); plt.close(fig)
    # per-configuration outcomes
    fig, ax = plt.subplots(figsize=(4.2, 8))
    ax.imshow(pc[list(seqs)].to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(seqs)), list(seqs)); ax.set_yticks(range(len(pc)), pc.index, fontsize=7)
    ax.set_title("Success per configuration\n(green = success, block order)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "per_config.png", dpi=150); plt.close(fig)
    # setup quality
    d = pd.concat(seqs.values())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].hist(d.cube_err_px.clip(upper=60), bins=30, alpha=0.7, label="cube")
    ax[0].hist(d.tray_err_px.clip(upper=60), bins=30, alpha=0.7, label="tray")
    ax[0].axvline(plan["configurations"]["tolerance_px"], c="k", ls="--", label="tolerance")
    ax[0].set(title="Setup error vs target (px, clipped at 60)", xlabel="px"); ax[0].legend()
    ax[1].hist(d.camera_shift_px.clip(upper=80), bins=30)
    ax[1].set(title="Scene-camera shift vs demonstrations (px)", xlabel="px")
    fig.tight_layout(); fig.savefig(fig_dir / "setup_quality.png", dpi=150); plt.close(fig)
    # violins and progress
    rates = {v: s.success.mean() for v, s in seqs.items()}
    an.violin_plot(seqs, an.cld_letters(list(seqs), rates, an.step_pairwise(plan, seqs)), path=fig_dir / "violins.png")
    an.progress_plot(seqs, path=fig_dir / "progress.png")
    plt.close("all")
    shutil.copy(PLAN / "configs.jpg", fig_dir / "configs.jpg")


def copy_inputs(plan: dict) -> None:
    for f in ["eval_plan.yaml", "configs.csv", "configs.jpg", "schedule.csv", "pilot_schedule.csv", "home_pose.json",
              "half_episodes.json", "scene_reference.jpg"]:
        shutil.copy(PLAN / f, OUT / "plan" / f)
    for name, folder in DEMOS.items():
        shutil.copy(HERE.parent / "data" / "raw" / folder / "demos.csv", OUT / "logs" / f"demos_{name}.csv")
    for name, spec in plan["policies"].items():
        src = (PLAN / spec["checkpoint"]).resolve()
        for f in ["config.json", "train_config.json"]:
            shutil.copy(src / f, OUT / "checkpoints" / name / f)
        (OUT / "checkpoints" / name / "SOURCE.txt").write_text(
            f"Weights: {src.relative_to(HERE.parent)}/model.safetensors (+ processor files)\nData: {spec['data']}\n")


def main() -> None:
    shutil.rmtree(OUT / "tables", ignore_errors=True)  # regenerated in full
    for d in ["tables", "figures", "plan", "checkpoints/full", "checkpoints/half"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    plan, log, seqs, applied = an.load()
    copy_inputs(plan)
    man = manifest(log)
    man.to_csv(OUT / "logs" / "manifest.csv", index=False)

    summary = an.summary_table(plan, seqs)
    fs = an.failure_stages(seqs)
    pairs = an.step_pairwise(plan, seqs)
    rates = {v: d.success.mean() for v, d in seqs.items()}
    letters = an.cld_letters(list(seqs), rates, pairs)
    factors = an.main_effects(plan, seqs)
    pc = an.per_config(seqs)
    checks = an.data_checks(plan, log, seqs)
    sens = sensitivity(plan)
    figures(plan, seqs, pc)

    k = len(seqs)
    n = min(len(d) for d in seqs.values())
    alpha = plan["alpha_total"] / (k * (k - 1) / 2)
    times = {v: d.loc[d.success == 1, "time_to_success_s"].dropna().astype(float) for v, d in seqs.items()}
    t_success = pd.DataFrame([{
        "variant": v, "policy / executed": f"{r['policy']} / {r['exec steps']}", "successes": r["successes"],
        "failures": r["failures"], "trials": r["trials"], "success rate": f"{r['success rate']:.2f}",
        "90% Wilson CI": r["90% Wilson CI"],
    } for v, r in summary.iterrows()]).set_index("variant")
    t_time_ok = pd.DataFrame([{
        "variant": v, "successful trials": int(seqs[v].success.sum()), "timed": len(t),
        "median (s)": f"{summary.loc[v, 'time median (s)']:.1f}", "IQR (s)": summary.loc[v, "time IQR (s)"],
    } for v, t in times.items()]).set_index("variant")

    def stop_times(d: pd.DataFrame) -> str:
        x = d.loc[(d.success == 0) & (d.end_reason == "stopped"), "duration_s"]
        return f"{len(x)} (median {x.median():.1f} s, range {x.min():.1f}–{x.max():.1f} s)" if len(x) else "0"
    t_time_fail = pd.DataFrame([{
        "variant": v, "failed trials": int((d.success == 0).sum()),
        f"timeout at {plan['timeout_s']} s": int(((d.success == 0) & (d.end_reason == "timeout")).sum()),
        "stopped by operator (time)": stop_times(d),
    } for v, d in seqs.items()]).set_index("variant")
    t_progress = pd.DataFrame([{
        "variant": v, "mean progress": f"{d.progress.mean():.2f}",
        **{lab: int((d.stages_completed == k).sum()) for k, lab in enumerate(["0", "0.25", "0.5", "0.75", "1 (success)"])},
    } for v, d in seqs.items()]).set_index("variant")
    t_where = fs.rename(index={"reaching the cube": "reaching the cube (score 0)", "grasping": "grasping (0.25)",
                               "carrying": "carrying (0.5)", "placing / releasing": "placing / releasing (0.75)"})
    t_where.index.name = "failed while"
    t_step = pd.DataFrame([{
        "A (first)": r["A"], "B (second)": r["B"], "successes A vs B": f"{r['success A'].split('/')[0]} vs {r['success B'].split('/')[0]} (of {r['pairs used']})",
        "decision": r["decision"],
        "favoured": f"**{r['favoured']}**" if r["favoured"] != "inconclusive" else "inconclusive",
        "trials per policy at first decision": "" if pd.isna(r["trials per policy at decision"]) else int(r["trials per policy at decision"]),
    } for r in pairs.to_dict("records")])

    md = {"success": save(t_success, "1_successes"), "ok": save(t_time_ok, "2_completion_time_success"),
          "fail": save(t_time_fail, "2_completion_time_failures"), "progress": save(t_progress, "3_progress"),
          "where": save(t_where, "3_failure_stages"), "step": save(t_step, "4_step", index=False)}
    for name, df in {"factors": factors, "data_checks": checks, "sensitivity": sens, "per_config": pc, "step_full": pairs}.items():
        save(df, name, index=name != "step_full")
    d_all = pd.concat(seqs.values())
    invalid = log[log.status == "invalid"]
    missing = man.loc[(man.video_scene == "missing") & (man.category != "pilot"), "rollout_id"].tolist()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    none_cfg = int((pc["variants succeeding"] == 0).sum())
    all_cfg = int((pc["variants succeeding"] == k).sum())
    pooled = {g: f"{r['success rate']:.0%}" for g, r in factors.iterrows()}
    sens_same = sens["pairs separated"].nunique() == 1 and sens["CLD"].nunique() == 1
    reach = {v: int(fs.loc["reaching the cube", v]) for v in seqs}

    ana = man[man.category == "analysed"]
    picks = {"F40 success": ana[(ana.variant == "F40") & (ana.success == 1)],
             "grasp failure": ana[ana.failure_stage == "grasping"],
             "half data, never reached the cube": ana[(ana.failure_stage == "reaching the cube") & ana.variant.str.startswith("H")]}
    examples = ", ".join(f"[{name} ({d.iloc[0].rollout_id})]({d.iloc[0].url_scene})" for name, d in picks.items() if len(d))

    text = f"""# Task 1 results: cube placement with ACT

**Question.** Does ACT succeed more often when trained on all 147 demonstrations instead of a random half (73), and
when it executes 40 instead of 15 of the 50 actions it predicts per chunk?

**Variants.** Two ACT policies (chunk 50, no VAE, step 45 000; their training configs differ only in the episodes),
each run with two execution horizons:

| Variant | Full form | Training data | Actions executed per predicted chunk of 50 |
|---|---|---|---|
| **F40** | Full data, execute 40 | all 147 demonstrations | 40 |
| **F15** | Full data, execute 15 | all 147 demonstrations | 15 |
| **H40** | Half data, execute 40 | random half, 73 demonstrations | 40 |
| **H15** | Half data, execute 15 | random half, 73 demonstrations | 15 |

**Training data.** All 147 demonstrations: [faizalkarim/hrl_A2_t1_v3](https://huggingface.co/datasets/faizalkarim/hrl_A2_t1_v3)
(LeRobot v3 dataset; the half-data policy's 73 episodes are listed in `plan/half_episodes.json`).

**Online.** This folder with every rollout video (wrist + scene, plays in the browser):
[{publish.RESULTS_REPO}](https://huggingface.co/datasets/{publish.RESULTS_REPO}), under `rollouts/<id>/`.
`logs/manifest.csv` gives each rollout's variant, configuration and outcome with direct video links (`url_wrist`, `url_scene`).

**Protocol.** {n} matched configurations (cube + tray pose), each run once per variant ({n * k} rollouts), in randomised
configuration and variant order, labelled blind. {plan['fps']} Hz, timeout {plan['timeout_s']} s, arm commands limited to
{plan['max_step_deg']} deg/step (gripper unlimited), same for all variants. **Success:** cube in the tray, gripper released,
arm lifted.

## 1. Successes and failures

{md['success']}

Binary success; 90% Wilson score interval per variant. The intervals describe each variant's own uncertainty;
their overlap is not the comparison (section 4).

![Success rate with 90% Wilson interval](figures/success_rates.png)

## 2. Completion time

Successful trials, from the start of policy execution until the success criterion was met:

{md['ok']}

Unsuccessful trials: ended at the timeout, or stopped earlier by the operator once the rollout had clearly failed
(gripper stuck at the cube, cube knocked away or dropped):

{md['fail']}

One H15 success (r0065) has no recorded time: it was ended with the failure key by mistake and relabelled (section 6).

![Completion times and failure end times](figures/completion_times.png)

## 3. Task progress and failures

Partial score, 0.25 per completed stage (reach the cube, grasp + lift, carry over the tray, place + release); reported
separately from binary success and not used in STEP. Number of trials per score:

{md['progress']}

Where the failed trials stopped progressing:

{md['where']}

- **Grasping is the main failure for every variant**: the gripper reaches the cube but does not secure it.
- **Data quantity:** the half-data policies more often failed to reach the cube at all ({reach['H40']} and {reach['H15']} trials vs
  {reach['F40']} and {reach['F15']}) and progressed less (mean 0.40 / 0.34 vs 0.72 / 0.61): with fewer demonstrations the cube is
  localised less reliably. Pooled over execution horizon: full {pooled['full data (F40+F15)']} vs half {pooled['half data (H40+H15)']} success.
- **Execution horizon:** 40 executed actions was ahead of 15 for both policies (pooled {pooled['execute 40 (F40+H40)']} vs {pooled['execute 15 (F15+H15)']}),
  with the same failure pattern; the difference is in how often the grasp succeeds, not where failures occur.

Example scene videos: {examples}.

![Task progress per trial](figures/progress.png)

## 4. STEP results

Settings: `MirroredStepTest`, `inference_mode="ranking"`, `alternative=Hypothesis.P0LessThanP1`,
`alpha` = {plan['alpha_total']} / {k*(k-1)//2} pairs = {alpha:.4f}, `n_max` = {plan['n_max']}, linear risk budget, outcomes in trial order
(no shuffling); sequentialized_barnard_tests {version('sequentialized_barnard_tests')}, commit 3ae1623.

{md['step']}

`AcceptNull` with A first favours A; `FailToDecide` is inconclusive. **F40 is favoured over H40 and H15.** The other four
pairs are inconclusive: no difference was established with {n} trials per policy, which does not mean the policies are equal.

![STEP pairwise decisions](figures/step_matrix.png)

## 5. Posterior success probability

![Posterior success probability with CLD letters](figures/violins.png)

Posterior Beta(s + 1, n − s + 1) of each variant's success probability (uniform prior), labelled with compact letters
from the STEP decisions: {', '.join(f'{v} `{x}`' for v, x in letters.items())}. Variants sharing a letter were not separated.

## 6. Notes

- **Matched design:** configurations are shared, so outcomes are correlated ({none_cfg} failed for all variants, {all_cfg} succeeded
  for all; `figures/per_config.png`); the two-sample test does not model this.
- **Corrections** (`logs/corrections.yaml`; raw log unchanged): r0065 (trial 61, H15) relabelled as a success (wrong key);
  block 30 (trials 117–120, one rollout per variant, 0/4 successes) excluded after an error during the run.
  {'STEP decisions and letters are identical without either correction' if sens_same else 'See'} (`tables/sensitivity.csv`).
- **Invalid rollouts** (repeated): {len(invalid)} ({', '.join(f'{r.rollout_id}: {r.notes}' for r in invalid.itertuples())}).
  **Setup:** {int(d_all.setup_override.sum())}/{len(d_all)} rollouts started outside the {plan['configurations']['tolerance_px']} px tolerance; median placement error
  {d_all.cube_err_px.median():.0f} px (≈ 0.4 cm), similar across variants.
- **Limitations:** the scene camera was ≈ {d_all.camera_shift_px.median():.0f} px from its data-collection position throughout
  (shared distribution shift; `figures/setup_quality.png`); one training run per data size; the plan was fixed before the
  first trial but not committed to git first; {len(missing)} rollout ({', '.join(missing)}) has no video; 4 pilot trials not analysed.
- **Files:** `logs/manifest.csv` (one row per rollout: trial, configuration, variant, policy, success, partial score,
  times, video paths), `logs/eval_log.csv` (raw), `logs/demos_batch1-3.csv`, `rollouts/<id>/` (wrist + scene video,
  joint data), `plan/`, `checkpoints/`, `tables/` (every table as CSV), `figures/`. Regenerate with
  `eval/03_export_results.sh`. lerobot {version('lerobot')}, torch {version('torch')}, commit {commit}.
"""
    (OUT / "README.md").write_text(text)
    print(f"Wrote {OUT}: README.md, {len(list((OUT / 'tables').glob('*.csv')))} tables, "
          f"{len(list((OUT / 'figures').glob('*.png')))} figures, manifest with {len(man)} rollouts")


if __name__ == "__main__":
    main()
