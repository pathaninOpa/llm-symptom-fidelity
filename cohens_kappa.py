"""
Intra-Rater Reliability — Cohen's Kappa
=========================================
Study: "Testing the Simulated Patient: PHQ-9 and GAD-7 Fidelity in LLM-Driven Mental Health Personas"
Author: Tae (JAIST KnowLab)

Protocol:
  1. Fill in SESSION_1 scores below after your first scoring session
  2. Wait 48 hours — do NOT look at session 1 scores
  3. Fill in SESSION_2 scores independently
  4. Run: python kappa.py

Output:
  - Cohen's kappa (overall and per-instrument)
  - Flagged items where |session1 - session2| > 1
  - Recommended ground truth table (mean of sessions where agreement, flagged for review otherwise)
"""


from sklearn.metrics import cohen_kappa_score
from collections import defaultdict

# ── Enter your scores here ────────────────────────────────────────────────────
# Format: (vignette_id, instrument, item_number): score (0–3)
# Use None for items not applicable to a vignette (e.g. GAD-7 for pure MDD cases)

SESSION_1 = {
    # V1 — Mr. A (45M) — MDD moderate-severe — PHQ-9 only
    ("V1", "PHQ9", 1): 3,
    ("V1", "PHQ9", 2): 3,
    ("V1", "PHQ9", 3): 3,
    ("V1", "PHQ9", 4): 3,
    ("V1", "PHQ9", 5): 3,
    ("V1", "PHQ9", 6): 3,
    ("V1", "PHQ9", 7): 3,
    ("V1", "PHQ9", 8): 3,
    ("V1", "PHQ9", 9): 0,

    # V2 — Ms. B (32F) — MDD mild-moderate — PHQ-9 only
    ("V2", "PHQ9", 1): 3,
    ("V2", "PHQ9", 2): 3,
    ("V2", "PHQ9", 3): 2,
    ("V2", "PHQ9", 4): 3,
    ("V2", "PHQ9", 5): 0,
    ("V2", "PHQ9", 6): 1,
    ("V2", "PHQ9", 7): 2,
    ("V2", "PHQ9", 8): 0,
    ("V2", "PHQ9", 9): 0,

    # V3 — Mr. C (34M) — GAD moderate — GAD-7 only
    ("V3", "GAD7", 1): 2,
    ("V3", "GAD7", 2): 2,
    ("V3", "GAD7", 3): 2,
    ("V3", "GAD7", 4): 2,
    ("V3", "GAD7", 5): 2,
    ("V3", "GAD7", 6): 0,
    ("V3", "GAD7", 7): 0,

    # V4 — Ms. D (28F) — GAD severe — GAD-7 only
    ("V4", "GAD7", 1): 3,
    ("V4", "GAD7", 2): 2,
    ("V4", "GAD7", 3): 2,
    ("V4", "GAD7", 4): 3,
    ("V4", "GAD7", 5): 3,
    ("V4", "GAD7", 6): 3,
    ("V4", "GAD7", 7): 3,

    # V5 — Mr. E (50M) — MDD with anxious distress, moderate — PHQ-9 + GAD-7
    ("V5", "PHQ9", 1): 2,
    ("V5", "PHQ9", 2): 2,
    ("V5", "PHQ9", 3): 2,
    ("V5", "PHQ9", 4): 2,
    ("V5", "PHQ9", 5): 0,
    ("V5", "PHQ9", 6): 0,
    ("V5", "PHQ9", 7): 2,
    ("V5", "PHQ9", 8): 0,
    ("V5", "PHQ9", 9): 0,
    ("V5", "GAD7", 1): 2,
    ("V5", "GAD7", 2): 2,
    ("V5", "GAD7", 3): 2,
    ("V5", "GAD7", 4): 2,
    ("V5", "GAD7", 5): 0,
    ("V5", "GAD7", 6): 2,
    ("V5", "GAD7", 7): 0,

    # V6 — Ms. F (39F) — Comorbid MDD + GAD, both severe — PHQ-9 + GAD-7
    ("V6", "PHQ9", 1): 3,
    ("V6", "PHQ9", 2): 3,
    ("V6", "PHQ9", 3): 0, ##
    ("V6", "PHQ9", 4): 3,
    ("V6", "PHQ9", 5): 3,
    ("V6", "PHQ9", 6): 3,
    ("V6", "PHQ9", 7): 0,
    ("V6", "PHQ9", 8): 0,
    ("V6", "PHQ9", 9): 0,
    ("V6", "GAD7", 1): 3,
    ("V6", "GAD7", 2): 2,
    ("V6", "GAD7", 3): 2,
    ("V6", "GAD7", 4): 3, ##
    ("V6", "GAD7", 5): 3,
    ("V6", "GAD7", 6): 3,
    ("V6", "GAD7", 7): 3, ##

    # V7 — Mr. G (28M) — GAD mild (contrast) — GAD-7 only
    ("V7", "GAD7", 1): 0,
    ("V7", "GAD7", 2): 2,
    ("V7", "GAD7", 3): 2,
    ("V7", "GAD7", 4): 0,
    ("V7", "GAD7", 5): 1,
    ("V7", "GAD7", 6): 0,
    ("V7", "GAD7", 7): 0,

    # V8 — Ms. H (40F) — MDD mild (contrast) — PHQ-9 only
    ("V8", "PHQ9", 1): 2,
    ("V8", "PHQ9", 2): 2,
    ("V8", "PHQ9", 3): 1,
    ("V8", "PHQ9", 4): 2,
    ("V8", "PHQ9", 5): 0,
    ("V8", "PHQ9", 6): 1,
    ("V8", "PHQ9", 7): 1,
    ("V8", "PHQ9", 8): 0,
    ("V8", "PHQ9", 9): 0,
}

# Copy SESSION_1 structure and fill in independently after 48 hours
SESSION_2 = {
    ("V1", "PHQ9", 1): 3,
    ("V1", "PHQ9", 2): 3,
    ("V1", "PHQ9", 3): 3,
    ("V1", "PHQ9", 4): 3,
    ("V1", "PHQ9", 5): 3,
    ("V1", "PHQ9", 6): 3,
    ("V1", "PHQ9", 7): 3,
    ("V1", "PHQ9", 8): 3,
    ("V1", "PHQ9", 9): 0,

    ("V2", "PHQ9", 1): 3,
    ("V2", "PHQ9", 2): 3,
    ("V2", "PHQ9", 3): 2,
    ("V2", "PHQ9", 4): 3,
    ("V2", "PHQ9", 5): 0,
    ("V2", "PHQ9", 6): 1,
    ("V2", "PHQ9", 7): 2,
    ("V2", "PHQ9", 8): 0,
    ("V2", "PHQ9", 9): 0,

    ("V3", "GAD7", 1): 2,
    ("V3", "GAD7", 2): 2,
    ("V3", "GAD7", 3): 2,
    ("V3", "GAD7", 4): 2,
    ("V3", "GAD7", 5): 2,
    ("V3", "GAD7", 6): 0,
    ("V3", "GAD7", 7): 0,

    ("V4", "GAD7", 1): 3,
    ("V4", "GAD7", 2): 2,
    ("V4", "GAD7", 3): 2,
    ("V4", "GAD7", 4): 3,
    ("V4", "GAD7", 5): 3,
    ("V4", "GAD7", 6): 3,
    ("V4", "GAD7", 7): 3,

    ("V5", "PHQ9", 1): 2,
    ("V5", "PHQ9", 2): 2,
    ("V5", "PHQ9", 3): 2,
    ("V5", "PHQ9", 4): 2,
    ("V5", "PHQ9", 5): 0,
    ("V5", "PHQ9", 6): 0,
    ("V5", "PHQ9", 7): 2,
    ("V5", "PHQ9", 8): 0,
    ("V5", "PHQ9", 9): 0,
    ("V5", "GAD7", 1): 2,
    ("V5", "GAD7", 2): 2,
    ("V5", "GAD7", 3): 2,
    ("V5", "GAD7", 4): 2,
    ("V5", "GAD7", 5): 0,
    ("V5", "GAD7", 6): 2,
    ("V5", "GAD7", 7): 0,

    ("V6", "PHQ9", 1): 3,
    ("V6", "PHQ9", 2): 3,
    ("V6", "PHQ9", 3): 0, ##
    ("V6", "PHQ9", 4): 3,
    ("V6", "PHQ9", 5): 3,
    ("V6", "PHQ9", 6): 3,
    ("V6", "PHQ9", 7): 0,
    ("V6", "PHQ9", 8): 0,
    ("V6", "PHQ9", 9): 0,
    ("V6", "GAD7", 1): 3,
    ("V6", "GAD7", 2): 2,
    ("V6", "GAD7", 3): 2,
    ("V6", "GAD7", 4): 3, ##
    ("V6", "GAD7", 5): 3,
    ("V6", "GAD7", 6): 3,
    ("V6", "GAD7", 7): 3, ##

    ("V7", "GAD7", 1): 0,
    ("V7", "GAD7", 2): 2,
    ("V7", "GAD7", 3): 2,
    ("V7", "GAD7", 4): 0,
    ("V7", "GAD7", 5): 1,
    ("V7", "GAD7", 6): 0,
    ("V7", "GAD7", 7): 0,

    ("V8", "PHQ9", 1): 2,
    ("V8", "PHQ9", 2): 2,
    ("V8", "PHQ9", 3): 1,
    ("V8", "PHQ9", 4): 2,
    ("V8", "PHQ9", 5): 0,
    ("V8", "PHQ9", 6): 1,
    ("V8", "PHQ9", 7): 1,
    ("V8", "PHQ9", 8): 0,
    ("V8", "PHQ9", 9): 0,
}


# ── Analysis ──────────────────────────────────────────────────────────────────
def compute_kappa():
    # Filter to keys present in both sessions with non-None values
    all_keys = sorted(SESSION_1.keys())
    valid_keys = [
        k for k in all_keys
        if SESSION_1.get(k) is not None and SESSION_2.get(k) is not None
    ]

    if not valid_keys:
        print("No scores found. Fill in SESSION_1 and SESSION_2 before running.")
        return

    s1 = [SESSION_1[k] for k in valid_keys]
    s2 = [SESSION_2[k] for k in valid_keys]

    # ── Overall kappa ─────────────────────────────────────────────────────────
    kappa_overall = cohen_kappa_score(s1, s2, weights='quadratic')

    print("=" * 60)
    print("INTRA-RATER RELIABILITY — Cohen's Kappa")
    print("=" * 60)
    print(f"\nTotal items scored: {len(valid_keys)}")
    print(f"Overall kappa:      {kappa_overall:.3f}  {interpret(kappa_overall)}")

    # ── Per-instrument kappa ──────────────────────────────────────────────────
    for inst in ["PHQ9", "GAD7"]:
        inst_keys = [k for k in valid_keys if k[1] == inst]
        if len(inst_keys) < 2:
            continue
        i1 = [SESSION_1[k] for k in inst_keys]
        i2 = [SESSION_2[k] for k in inst_keys]
        k  = cohen_kappa_score(i1, i2, weights='quadratic')
        print(f"{inst} kappa ({len(inst_keys)} items):  {k:.3f}  {interpret(k)}")

    # ── Per-vignette kappa ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Per-vignette breakdown:")
    vignettes = sorted(set(k[0] for k in valid_keys))
    for v in vignettes:
        v_keys = [k for k in valid_keys if k[0] == v]
        v1 = [SESSION_1[k] for k in v_keys]
        v2 = [SESSION_2[k] for k in v_keys]
        # Need at least 2 unique values for kappa; if all identical, kappa=1
        if len(set(v1 + v2)) < 2:
            print(f"  {v}: kappa = 1.000 (perfect — all items identical)")
        else:
            k = cohen_kappa_score(v1, v2, weights='quadratic')
            print(f"  {v}: kappa = {k:.3f}  {interpret(k)}")

    # ── Flagged items (|s1 - s2| > 1) ────────────────────────────────────────
    flagged = [k for k in valid_keys if abs(SESSION_1[k] - SESSION_2[k]) > 1]
    print("\n" + "-" * 60)
    if flagged:
        print(f"⚠️  Flagged items (|session1 - session2| > 1): {len(flagged)}")
        for k in flagged:
            v, inst, item = k
            print(f"  {v} {inst} item {item}: "
                  f"session1={SESSION_1[k]}  session2={SESSION_2[k]}  "
                  f"diff={abs(SESSION_1[k]-SESSION_2[k])}")
        print("  → Review these items before locking ground truth.")
    else:
        print("✓ No items flagged (all disagreements ≤ 1 point)")

    # ── Ground truth recommendation ───────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Recommended ground truth scores:")
    print("(mean of sessions for agreed items; flagged items marked for review)\n")

    gt = {}
    for k in valid_keys:
        s1_score = SESSION_1[k]
        s2_score = SESSION_2[k]
        diff = abs(s1_score - s2_score)
        if diff <= 1:
            # Round mean — conservative approach
            gt[k] = round((s1_score + s2_score) / 2)
        else:
            gt[k] = "REVIEW"

    # Print as a table grouped by vignette
    current_v = None
    for k in valid_keys:
        v, inst, item = k
        if v != current_v:
            print(f"  {v}:")
            current_v = v
        marker = " ← REVIEW" if gt[k] == "REVIEW" else ""
        print(f"    {inst} item {item}: {gt[k]}{marker}")

    # Print copy-paste format for experiment.py ground_truth dict
    print("\n" + "=" * 60)
    print("Copy-paste format for experiment.py ground_truth dict:")
    print("(REVIEW items shown as None — resolve before pasting)\n")

    by_vignette = defaultdict(lambda: defaultdict(dict))
    for k in valid_keys:
        v, inst, item = k
        by_vignette[v][inst][item] = gt[k]

    for v in sorted(by_vignette.keys()):
        print(f'    # {v}')
        for inst in sorted(by_vignette[v].keys()):
            scores = by_vignette[v][inst]
            score_str = ", ".join(
                f"{i}: {scores[i] if scores[i] != 'REVIEW' else 'None'}"
                for i in sorted(scores.keys())
            )
            print(f'    "{inst}": {{{score_str}}},')
        print()

    # ── Pass/fail summary ─────────────────────────────────────────────────────
    print("=" * 60)
    print("RELIABILITY VERDICT")
    print("=" * 60)
    if kappa_overall >= 0.80:
        print(f"✓ PASS — kappa={kappa_overall:.3f} meets the ≥0.80 threshold.")
        print("  Ground truth is sufficiently reliable to proceed.")
    elif kappa_overall >= 0.61:
        print(f"⚠ MARGINAL — kappa={kappa_overall:.3f}. Acceptable for feasibility study")
        print("  but should be reported as a limitation. Review flagged items.")
    else:
        print(f"✗ FAIL — kappa={kappa_overall:.3f}. Ground truth unreliable.")
        print("  Revisit scoring rules before proceeding with the experiment.")


def interpret(k: float) -> str:
    if k > 0.80:  return "(Almost perfect)"
    if k > 0.60:  return "(Substantial)"
    if k > 0.40:  return "(Moderate)"
    if k > 0.20:  return "(Fair)"
    return "(Poor)"


if __name__ == "__main__":
    compute_kappa()