"""One-PDF visual report of a hybrid (predictor+VLA) closed-loop eval run.

VLABench adaptation of the action_predictor eval report: a single env writes one
``chunk_decisions.jsonl`` (one row per replan decision, with ``episode_index`` and a
per-episode ``replan_index``), and per-episode success is read from the saved video
filenames (``<task>/videos/<idx>_success_<bool>_progress_<p>.mp4``). No extra logging is
needed -- everything is already in the run dir.

Pages:
    1  summary + headline: success rate, overall skip (substitution) rate, mean skip by outcome
    2  per-episode decision timeline raster (where each plan went VLA vs skip), grouped by outcome
    3  skip-rate-vs-progress (success vs failure) + reject-reason breakdown + gate-score histogram

``make_report(run_dir)`` is called automatically at the end of live_eval; it is also a CLI:
    python -m r3m_action_predictor.visualize_run --run-dir <save_dir>/<track> [--out report.pdf]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

C_NONE, C_STRUCT, C_GATE, C_SUB = "#e8e8e8", "#4c72b0", "#dd8452", "#55a868"
C_OK, C_FAIL = "#2ca02c", "#d62728"

# A non-substitute decision is a "gate said no" (orange) if rejected for a score/safety reason,
# else "structural" (blue: warmup / history / cooldown / budget).
GATE_REASONS = {
    "envelope_above_threshold", "gripper_change", "risk_below_threshold", "below_threshold",
    "confidence_below_threshold", "step_confidence_below_threshold", "prefix_risk_below_threshold",
    "action_safety_gate", "risk_score_unavailable", "gate_score_unavailable",
}


def _success_by_episode(run_dir: Path) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for v in glob.glob(str(run_dir / "*" / "videos" / "*.mp4")):
        m = re.match(r"(\d+)_success_(True|False)", os.path.basename(v))
        if m:
            out[int(m.group(1))] = m.group(2) == "True"
    return out


def _load_decisions(run_dir: Path) -> list[dict]:
    path = run_dir / "hybrid_logs" / "chunk_decisions.jsonl"
    return [json.loads(l) for l in open(path) if json.loads(l).get("event") == "chunk_decision"]


def _episode_rows(decisions: list[dict], success: dict[int, bool]) -> list[dict]:
    by_ep = defaultdict(list)
    for d in decisions:
        by_ep[int(d["episode_index"])].append(d)
    rows = []
    for ep, ds in sorted(by_ep.items()):
        ds.sort(key=lambda x: x.get("replan_index", 0))
        sub = np.array([bool(x.get("use_substitute")) for x in ds])
        rows.append({
            "episode_index": ep,
            "success": success.get(ep),
            "n_plans": len(ds),
            "skip_rate": float(sub.mean()) if len(sub) else 0.0,
            "plan_idx": np.array([int(x.get("replan_index", i)) for i, x in enumerate(ds)]),
            "substitute": sub,
            "reasons": [x.get("reject_reasons") or [] for x in ds],
        })
    return rows


def _cell(sub: bool, reasons: list[str]) -> int:
    if sub:
        return 2  # substitute
    if any(r in GATE_REASONS for r in reasons):
        return 1  # VLA: gate said no
    return 0      # VLA: structural (warmup/cooldown/history/budget)


def page_summary(pdf, rows, decisions, meta):
    labeled = [r for r in rows if r["success"] is not None]
    succ = [r["skip_rate"] for r in labeled if r["success"]]
    fail = [r["skip_rate"] for r in labeled if not r["success"]]
    overall_skip = float(np.mean([bool(d.get("use_substitute")) for d in decisions])) if decisions else 0.0
    sr = np.mean([r["success"] for r in labeled]) if labeled else float("nan")
    metric = decisions[0].get("decision_metric", "?") if decisions else "?"
    thrs = [d.get("decision_threshold") for d in decisions if d.get("decision_threshold") is not None]
    thr = thrs[-1] if thrs else float("nan")

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"Hybrid eval report\n{meta.get('run_name','')}", fontsize=13, fontweight="bold")
    ax = fig.add_axes([0.06, 0.55, 0.52, 0.33]); ax.axis("off")
    lines = [
        f"task: {meta.get('task','?')}    decision_metric: {metric}",
        f"episodes: {len(rows)}   (success {sum(1 for r in labeled if r['success'])}, "
        f"failure {sum(1 for r in labeled if not r['success'])}, unlabeled {len(rows)-len(labeled)})",
        f"success rate: {sr:.3f}",
        f"overall skip (substitution) rate: {overall_skip:.3f}",
        f"gate threshold: {thr:.4f}",
        f"prev/pred horizon: {meta.get('prev_horizon','?')}/{meta.get('pred_horizon','?')}   "
        f"replan_steps: {meta.get('replan_steps','?')}",
        "",
        "Mean skip rate by outcome:",
        f"   success: {np.mean(succ) if succ else float('nan'):.3f} (median {np.median(succ) if succ else float('nan'):.3f}, n={len(succ)})",
        f"   failure: {np.mean(fail) if fail else float('nan'):.3f} (median {np.median(fail) if fail else float('nan'):.3f}, n={len(fail)})",
    ]
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")

    ax1 = fig.add_axes([0.64, 0.55, 0.30, 0.33])
    means = [np.mean(succ) if succ else 0, np.mean(fail) if fail else 0]
    ax1.bar(["success", "failure"], means, color=[C_OK, C_FAIL])
    ax1.set_ylabel("mean skip rate"); ax1.set_title("Skip rate by outcome")
    ax1.set_ylim(0, max(0.01, max(means) * 1.3))

    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.34])
    data = [succ or [0], fail or [0]]
    bp = ax2.boxplot(data, showmeans=True, patch_artist=True)
    ax2.set_xticks([1, 2]); ax2.set_xticklabels([f"success (n={len(succ)})", f"failure (n={len(fail)})"])
    for patch, c in zip(bp["boxes"], [C_OK, C_FAIL]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    for i, vals in enumerate(data, start=1):
        x = np.random.default_rng(0).normal(i, 0.04, size=len(vals))
        ax2.scatter(x, vals, s=12, color="k", alpha=0.5, zorder=3)
    ax2.set_ylabel("per-episode skip rate"); ax2.set_title("Per-episode skip rate distribution (success vs failure)")
    pdf.savefig(fig); plt.close(fig)


def page_timeline(pdf, rows):
    rows = sorted(rows, key=lambda r: (r["success"] is True, r["episode_index"]))
    max_plan = max((int(r["plan_idx"].max()) + 1 if r["n_plans"] else 1) for r in rows)
    grid = np.full((len(rows), max_plan), -1, dtype=np.int8)
    for i, r in enumerate(rows):
        for j, pidx in enumerate(r["plan_idx"]):
            grid[i, int(pidx)] = _cell(bool(r["substitute"][j]), r["reasons"][j])
    cmap = ListedColormap([C_NONE, C_STRUCT, C_GATE, C_SUB])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    # pcolormesh (vector quads) instead of imshow (embedded raster image): the raster renders
    # blank in some PDF viewers (e.g. VSCode), whereas vector quads display everywhere.
    ax.pcolormesh(np.arange(max_plan + 1), np.arange(len(rows) + 1), grid, cmap=cmap, norm=norm)
    ax.invert_yaxis()  # episode 0 at top, like imshow
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([f"ep {r['episode_index']}" for r in rows], fontsize=5)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(C_OK if r["success"] else (C_FAIL if r["success"] is False else "k"))
    n_fail = sum(1 for r in rows if r["success"] is False)
    if 0 < n_fail < len(rows):
        ax.axhline(n_fail, color="k", lw=1.0, ls="--")
    ax.set_xlabel("plan index within episode (each = one VLA-or-skip decision)")
    ax.set_ylabel("episode (label colour: green=success, red=failure)")
    ax.set_title("Per-episode decision timeline -- where the VLA was skipped (substituted)")
    ax.legend(handles=[
        Patch(color=C_NONE, label="no plan"), Patch(color=C_STRUCT, label="VLA (warmup/cooldown/history)"),
        Patch(color=C_GATE, label="VLA (gate rejected)"), Patch(color=C_SUB, label="substitute (skip VLA)"),
    ], loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_extra(pdf, rows, decisions, nbins=10):
    fig = plt.figure(figsize=(11, 8.5))
    # (a) skip rate vs normalized progress, by outcome
    ax = fig.add_axes([0.08, 0.58, 0.4, 0.33])
    for ok, color, lab in [(True, C_OK, "success"), (False, C_FAIL, "failure")]:
        sub_rows = [r for r in rows if r["success"] is ok]
        if not sub_rows:
            continue
        progs, subs = [], []
        for r in sub_rows:
            denom = max(1, int(r["plan_idx"].max()))
            progs.append(r["plan_idx"] / denom); subs.append(r["substitute"].astype(float))
        progs = np.concatenate(progs); subs = np.concatenate(subs)
        bins = np.linspace(0, 1, nbins + 1)
        idx = np.clip(np.digitize(progs, bins) - 1, 0, nbins - 1)
        rate = [subs[idx == b].mean() if (idx == b).any() else np.nan for b in range(nbins)]
        ax.plot((bins[:-1] + bins[1:]) / 2, rate, "-o", color=color, label=lab)
    ax.set_xlabel("normalized episode progress"); ax.set_ylabel("skip rate")
    ax.set_ylim(0, 1); ax.set_title("Skip rate vs episode progress"); ax.legend(fontsize=8)

    # (b) reject-reason breakdown
    ax2 = fig.add_axes([0.57, 0.58, 0.4, 0.33])
    cnt = defaultdict(int)
    for d in decisions:
        if d.get("use_substitute"):
            cnt["substitute"] += 1
        for r in (d.get("reject_reasons") or []):
            cnt[r] += 1
    items = sorted(cnt.items(), key=lambda kv: -kv[1])
    ax2.bar([k for k, _ in items], [v for _, v in items],
            color=[C_SUB if k == "substitute" else (C_GATE if k in GATE_REASONS else C_STRUCT) for k, _ in items])
    ax2.set_ylabel("decisions"); ax2.set_title("Decision / reject-reason breakdown")
    plt.setp(ax2.get_xticklabels(), rotation=35, ha="right", fontsize=7)

    # (c) gate-score histogram (decision_score) with threshold
    ax3 = fig.add_axes([0.08, 0.1, 0.88, 0.36])
    metric = decisions[0].get("decision_metric", "?") if decisions else "?"
    scores = np.array([d["decision_score"] for d in decisions if d.get("decision_score") is not None], dtype=float)
    thrs = [d.get("decision_threshold") for d in decisions if d.get("decision_threshold") is not None]
    if scores.size:
        ax3.hist(scores, bins=40, color=C_SUB, alpha=0.85)
        if thrs:
            ax3.axvline(thrs[-1], color="red", ls="--", label=f"threshold={thrs[-1]:.3f}"); ax3.legend()
        direction = "lower=skip" if metric in ("envelope", "aleatoric") else "higher=skip"
        ax3.set_xlabel(f"online gate score ({metric}, {direction}; eligible decisions)")
        ax3.set_ylabel("count"); ax3.set_title(f"Online gate-score distribution ({metric})")
    else:
        ax3.axis("off"); ax3.text(0.5, 0.5, "no candidate gate scores logged", ha="center")
    pdf.savefig(fig); plt.close(fig)


def make_report(run_dir: str | Path, out: str | Path | None = None, meta: dict | None = None) -> Path:
    """Build the PDF report for one eval run dir (the per-track dir with hybrid_logs/ + <task>/videos/)."""
    run_dir = Path(run_dir)
    decisions = _load_decisions(run_dir)
    if not decisions:
        raise FileNotFoundError(f"no chunk_decisions in {run_dir}")
    rows = _episode_rows(decisions, _success_by_episode(run_dir))
    meta = dict(meta or {})
    meta.setdefault("run_name", run_dir.parent.name + "/" + run_dir.name)
    out = Path(out) if out else run_dir / "analysis" / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out) as pdf:
        page_summary(pdf, rows, decisions, meta)
        page_timeline(pdf, rows)
        page_extra(pdf, rows, decisions)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="per-track run dir (contains hybrid_logs/ and <task>/videos/)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = make_report(args.run_dir, args.out)
    print(f"[visualize_run] wrote {out}")


if __name__ == "__main__":
    main()
