"""Pick the risk threshold that yields a target VLA-skip (substitution) rate.

Reads a hybrid run's chunk_decisions.jsonl and estimates, for a grid of thresholds T,
the fraction of replans that would substitute = (replans whose candidate score >= T AND
whose only barrier was the risk threshold) / total replans. Gates other than the risk
threshold (min_vla_calls, cooldown, consecutive limit, no-candidate) are treated as fixed
from the reference run, so the estimate is approximate (lowering T triggers more
substitutions, hence slightly more cooldowns) -- the final run is what we verify against.

Usage: python pick_threshold.py <chunk_decisions.jsonl> [target_fraction=0.20]
"""
import json
import sys

# reject reasons that are about the decision score/threshold (removable by lowering T)
THRESHOLD_REASONS = {
    "risk_below_threshold",
    "confidence_below_threshold",
    "step_confidence_below_threshold",
    "prefix_risk_below_threshold",
}

def main():
    path = sys.argv[1]
    target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20

    total = 0
    actual_sub = 0
    ref_thr = None
    # eligible[i] = score of a replan that would substitute at a low enough threshold
    eligible_scores = []
    for line in open(path):
        d = json.loads(line)
        if d.get("event") != "chunk_decision":
            continue
        total += 1
        ref_thr = d.get("risk_threshold", ref_thr)
        score = d.get("candidate_risk_safe_probability")
        if d.get("use_substitute"):
            actual_sub += 1
            if score is not None:
                eligible_scores.append(score)
            continue
        reasons = set(d.get("reject_reasons") or [])
        # eligible iff a candidate score exists and the ONLY barriers are threshold-type
        if score is not None and reasons and reasons <= THRESHOLD_REASONS:
            eligible_scores.append(score)

    eligible_scores.sort()
    def frac_at(t):
        return sum(1 for s in eligible_scores if s >= t) / total if total else 0.0

    # search threshold over candidate score values for the closest to target
    grid = sorted(set(eligible_scores))
    best_t, best_err = ref_thr or 0.9, 1e9
    for t in grid:
        err = abs(frac_at(t) - target)
        if err < best_err:
            best_err, best_t = err, t
    # nudge slightly below the chosen score so that score>=t includes it
    chosen = best_t

    print(f"total_replans          = {total}")
    print(f"reference_threshold    = {ref_thr}")
    print(f"actual_sub_fraction    = {actual_sub/total:.4f} ({actual_sub}/{total})")
    print(f"gate-eligible replans  = {len(eligible_scores)} ({len(eligible_scores)/total:.3f} of total)")
    if eligible_scores:
        import statistics
        qs = [eligible_scores[int(p*(len(eligible_scores)-1))] for p in (0.0,0.25,0.5,0.75,1.0)]
        print(f"eligible score quantiles (min,q25,med,q75,max) = "
              + ", ".join(f"{q:.4f}" for q in qs))
    print(f"target_skip_fraction   = {target}")
    print(f"==> chosen_threshold   = {chosen:.4f}  (predicted skip = {frac_at(chosen):.4f})")
    # also show the curve near the target for context
    print("threshold -> predicted_skip:")
    for t in [0.5,0.6,0.7,0.8,0.85,0.9,0.92,0.94,0.95,0.96,0.97,0.98,0.99]:
        print(f"  {t:.2f} -> {frac_at(t):.3f}")

if __name__ == "__main__":
    main()
