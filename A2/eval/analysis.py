"""Task 1 analysis: success table, completion times, partial progress, STEP pairwise tests, CLD, violins.

Uses only the valid rollouts of eval_log.csv, in chronological (schedule) order, after the documented
post-hoc corrections in corrections.yaml (relabels, excluded trials; the raw log is never modified). Each variant's sequence
is ordered by trial, so the i-th outcome of every variant comes from the same block = the same config.
STEP: TRI's sequentialized_barnard_tests (pip install git+https://github.com/TRI-ML/sequentialized_barnard_tests),
MirroredStepTest in "ranking" mode, alpha split over all k(k-1)/2 pairs.
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "task1"


def load(plan_path: Path = HERE / "plan" / "eval_plan.yaml", log_path: Path = RESULTS / "logs" / "eval_log.csv",
         corrections_path: Path = RESULTS / "logs" / "corrections.yaml"):
    """Plan, the full log (corrections applied, `included` column), the analysed rollouts per variant in
    trial order, and a list of the corrections applied (for the report)."""
    plan = yaml.safe_load(plan_path.read_text())
    log = pd.read_csv(log_path)
    log["included"] = log.status == "valid"
    applied = []
    if corrections_path.exists():
        corr = yaml.safe_load(corrections_path.read_text()) or {}
        for r in corr.get("relabel") or []:
            idx = log.index[log.rollout_id == r["rollout_id"]]
            if len(idx) != 1:
                raise ValueError(f"Relabel: rollout {r['rollout_id']} not found once in the log")
            before = log.loc[idx[0], list(r["set"])].to_dict()
            for k, v in r["set"].items():
                log.loc[idx[0], k] = v
            change = ", ".join(f"{k} {before[k] if pd.notna(before[k]) else '-'} -> {v if v != '' else '-'}"
                               for k, v in r["set"].items() if k in ("success", "stages_completed", "end_reason"))
            applied.append(f"Relabelled {r['rollout_id']} (trial {r['trial']}, {r['variant']}): {change}. "
                           + " ".join(r["reason"].split()))
        ex = corr.get("exclude_trials") or {}
        if ex.get("trials"):
            mask = log.trial.isin(ex["trials"])
            log.loc[mask, "included"] = False
            applied.append(f"Excluded trials {ex['trials']} ({int((mask & (log.status == 'valid')).sum())} valid rollouts). "
                           + " ".join(ex["reason"].split()))
    valid = log[log.included].sort_values("trial")
    seqs = {v: valid[valid.variant == v] for v in plan["variants"]}
    return plan, log, seqs, applied


def wilson(s: int, n: int, conf: float = 0.90) -> tuple[float, float]:
    """Wilson score interval for s successes in n trials."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(0.5 + conf / 2)
    p = s / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def summary_table(plan: dict, seqs: dict) -> pd.DataFrame:
    rows = []
    for v, d in seqs.items():
        s, n = int(d.success.sum()), len(d)
        lo, hi = wilson(s, n)
        t = d.loc[d.success == 1, "time_to_success_s"].astype(float)
        fails = d[d.success == 0]
        rows.append({
            "variant": v, "policy": plan["variants"][v]["policy"], "exec steps": plan["variants"][v]["n_action_steps"],
            "successes": s, "failures": n - s, "trials": n,
            "success rate": round(s / n, 3) if n else np.nan, "90% Wilson CI": f"[{lo:.2f}, {hi:.2f}]",
            "timed successes": int(t.notna().sum()),
            "time median (s)": round(t.median(), 1) if len(t) else np.nan,
            "time IQR (s)": f"[{t.quantile(.25):.1f}, {t.quantile(.75):.1f}]" if len(t) else "",
            "mean progress": round(d.progress.mean(), 3) if n else np.nan,
            "timeouts": int((fails.end_reason == "timeout").sum()),
            "stopped": int((fails.end_reason == "stopped").sum()),
        })
    return pd.DataFrame(rows).set_index("variant")


def failure_stages(seqs: dict) -> pd.DataFrame:
    """Where the failed rollouts stopped progressing (counts per variant)."""
    order = ["reaching the cube", "grasping", "carrying", "placing / releasing"]
    t = pd.DataFrame({v: d[d.success == 0].failure_stage.value_counts() for v, d in seqs.items()}).reindex(order)
    return t.fillna(0).astype(int)


def step_pairwise(plan: dict, seqs: dict) -> pd.DataFrame:
    """MirroredStepTest (ranking) for every pair, outcomes fed in trial order; first decision recorded."""
    from sequentialized_barnard_tests import Decision, Hypothesis, MirroredStepTest

    names = list(seqs)
    k = len(names)
    alpha = plan["alpha_total"] / (k * (k - 1) / 2)
    rows = []
    for a, b in itertools.combinations(names, 2):
        xa, xb = seqs[a].success.to_numpy(int), seqs[b].success.to_numpy(int)
        n = min(len(xa), len(xb))
        test = MirroredStepTest(Hypothesis.P0LessThanP1, plan["n_max"], alpha, inference_mode="ranking")
        decision, at = Decision.FailToDecide, None
        for t in range(n):  # one matched pair at a time, so the first decision and its trial count are exact
            res = test.step(xa[t], xb[t])
            if res.decision != Decision.FailToDecide:
                decision, at = res.decision, t + 1
                break
        favoured = {Decision.AcceptAlternative: b, Decision.AcceptNull: a}.get(decision, "inconclusive")
        rows.append({"A": a, "B": b, "pairs used": n, "decision": decision.name,
                     "favoured": favoured, "trials per policy at decision": at, "alpha_pair": round(alpha, 4),
                     "success A": f"{xa[:n].sum()}/{n}", "success B": f"{xb[:n].sum()}/{n}"})
    return pd.DataFrame(rows)


def cld_letters(names: list[str], rates: dict, pairs: pd.DataFrame) -> dict:
    """Compact letter display: one letter per maximal group of variants with no separated pair among them.

    Groups are lettered from the highest success rate down, so 'a' goes to the best group.
    """
    separated = {frozenset((r.A, r.B)) for r in pairs.itertuples() if r.favoured != "inconclusive"}
    together = lambda g: all(frozenset(p) not in separated for p in itertools.combinations(g, 2))  # noqa: E731
    groups = [set(g) for r in range(len(names), 0, -1) for g in itertools.combinations(names, r) if together(g)]
    maximal = [g for g in groups if not any(g < h for h in groups)]
    maximal.sort(key=lambda g: -max(rates[v] for v in g))
    letters = {v: "" for v in names}
    for i, g in enumerate(maximal):
        for v in sorted(g, key=names.index):
            letters[v] += "abcdefgh"[i]
    return letters


def violin_plot(seqs: dict, letters: dict, path: Path | None = None, seed: int = 0):
    """Posterior Beta(s+1, n-s+1) of each variant's success probability, labelled with CLD letters."""
    rng = np.random.default_rng(seed)
    names = list(seqs)
    samples = [rng.beta(d.success.sum() + 1, len(d) - d.success.sum() + 1, 20000) for d in seqs.values()]
    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 1, 4))
    parts = ax.violinplot(samples, showmeans=False, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.6)
    for i, (v, d) in enumerate(seqs.items(), 1):
        ax.text(i, 1.03, letters[v], ha="center", fontsize=13, fontweight="bold")
        ax.scatter(i, d.success.mean(), color="k", zorder=3, s=15)
    ax.set_xticks(range(1, len(names) + 1), [f"{v}\n{int(d.success.sum())}/{len(d)}" for v, d in seqs.items()])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("success probability (posterior)")
    ax.set_title("Posterior success probability, Beta(s+1, n-s+1); letters: STEP CLD (alpha 0.10)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def data_checks(plan: dict, log: pd.DataFrame, seqs: dict) -> pd.DataFrame:
    """Per variant: rollouts, invalid repeats, setup quality, camera shift, control-loop health."""
    rows = []
    for v, d in seqs.items():
        allv = log[log.variant == v]
        rows.append({
            "variant": v, "analysed trials": len(d), "invalid (repeated)": int((allv.status == "invalid").sum()),
            "setup overrides": int(d.setup_override.sum()),
            "cube setup error px (median)": round(d.cube_err_px.median(), 1),
            "tray setup error px (median)": round(d.tray_err_px.median(), 1),
            "camera shift px (median)": round(d.camera_shift_px.median(), 1),
            "loop Hz (mean)": round(d.mean_hz.mean(), 1), "slow frames (mean)": round(d.slow_frames.mean(), 1),
            "limited cmds / rollout": round(d.limited_cmds.mean()),
        })
    return pd.DataFrame(rows).set_index("variant")


def main_effects(plan: dict, seqs: dict) -> pd.DataFrame:
    """Descriptive pooled rates for the two factors (each pooled over the other factor's two levels)."""
    groups = {"full data (F40+F15)": ["F40", "F15"], "half data (H40+H15)": ["H40", "H15"],
              "execute 40 (F40+H40)": ["F40", "H40"], "execute 15 (F15+H15)": ["F15", "H15"]}
    rows = []
    for name, vs in groups.items():
        d = pd.concat([seqs[v] for v in vs if v in seqs])
        s, n = int(d.success.sum()), len(d)
        lo, hi = wilson(s, n)
        rows.append({"group": name, "successes": s, "trials": n, "success rate": round(s / n, 3),
                     "90% Wilson CI": f"[{lo:.2f}, {hi:.2f}]", "mean progress": round(d.progress.mean(), 3)})
    return pd.DataFrame(rows).set_index("group")


def per_config(seqs: dict) -> pd.DataFrame:
    """Success (1/0) of every variant on every configuration (rows in block order)."""
    t = pd.concat([d.set_index("config_id").success.rename(v) for v, d in seqs.items()], axis=1)
    order = pd.concat(seqs.values()).groupby("config_id").block.min().sort_values().index
    t = t.loc[order].astype(int)
    t["variants succeeding"] = t.sum(axis=1)
    return t


def progress_plot(seqs: dict, path: Path | None = None):
    """Stacked bars: how far each variant got (0-4 stages completed) over its trials."""
    labels = ["0: did not reach", "1: reached cube", "2: grasped + lifted", "3: carried to tray", "4: success"]
    counts = pd.DataFrame({v: d.stages_completed.astype(int).value_counts().reindex(range(5), fill_value=0)
                           for v, d in seqs.items()}).T
    fig, ax = plt.subplots(figsize=(7, 3.5))
    left = np.zeros(len(counts))
    colours = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 5))
    for k in range(5):
        ax.barh(counts.index, counts[k], left=left, color=colours[k], label=labels[k])
        left += counts[k].to_numpy()
    ax.invert_yaxis()
    ax.set_xlabel("trials")
    ax.set_title("Task progress per trial (stages completed)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table (no tabulate dependency), for pasting into the report."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])
