============================================================
INTRA-RATER RELIABILITY — Cohen's Kappa
============================================================

Total items scored: 80
Overall kappa:      1.000  (Almost perfect)
PHQ9 kappa (45 items):  1.000  (Almost perfect)
GAD7 kappa (35 items):  1.000  (Almost perfect)

------------------------------------------------------------
Per-vignette breakdown:
  V1: kappa = 1.000  (Almost perfect)
  V2: kappa = 1.000  (Almost perfect)
  V3: kappa = 1.000  (Almost perfect)
  V4: kappa = 1.000  (Almost perfect)
  V5: kappa = 1.000  (Almost perfect)
  V6: kappa = 1.000  (Almost perfect)
  V7: kappa = 1.000  (Almost perfect)
  V8: kappa = 1.000  (Almost perfect)

------------------------------------------------------------
✓ No items flagged (all disagreements ≤ 1 point)

------------------------------------------------------------
Recommended ground truth scores:
(mean of sessions for agreed items; flagged items marked for review)

  V1:
    PHQ9 item 1: 3
    PHQ9 item 2: 3
    PHQ9 item 3: 3
    PHQ9 item 4: 3
    PHQ9 item 5: 3
    PHQ9 item 6: 3
    PHQ9 item 7: 3
    PHQ9 item 8: 3
    PHQ9 item 9: 0
  V2:
    PHQ9 item 1: 3
    PHQ9 item 2: 3
    PHQ9 item 3: 2
    PHQ9 item 4: 3
    PHQ9 item 5: 0
    PHQ9 item 6: 1
    PHQ9 item 7: 2
    PHQ9 item 8: 0
    PHQ9 item 9: 0
  V3:
    GAD7 item 1: 2
    GAD7 item 2: 2
    GAD7 item 3: 2
    GAD7 item 4: 2
    GAD7 item 5: 2
    GAD7 item 6: 0
    GAD7 item 7: 0
  V4:
    GAD7 item 1: 3
    GAD7 item 2: 2
    GAD7 item 3: 2
    GAD7 item 4: 3
    GAD7 item 5: 3
    GAD7 item 6: 3
    GAD7 item 7: 3
  V5:
    GAD7 item 1: 2
    GAD7 item 2: 2
    GAD7 item 3: 2
    GAD7 item 4: 2
    GAD7 item 5: 0
    GAD7 item 6: 2
    GAD7 item 7: 0
    PHQ9 item 1: 2
    PHQ9 item 2: 2
    PHQ9 item 3: 2
    PHQ9 item 4: 2
    PHQ9 item 5: 0
    PHQ9 item 6: 0
    PHQ9 item 7: 2
    PHQ9 item 8: 0
    PHQ9 item 9: 0
  V6:
    GAD7 item 1: 3
    GAD7 item 2: 2
    GAD7 item 3: 2
    GAD7 item 4: 3
    GAD7 item 5: 3
    GAD7 item 6: 3
    GAD7 item 7: 3
    PHQ9 item 1: 3
    PHQ9 item 2: 3
    PHQ9 item 3: 0
    PHQ9 item 4: 3
    PHQ9 item 5: 3
    PHQ9 item 6: 3
    PHQ9 item 7: 0
    PHQ9 item 8: 0
    PHQ9 item 9: 0
  V7:
    GAD7 item 1: 0
    GAD7 item 2: 2
    GAD7 item 3: 2
    GAD7 item 4: 0
    GAD7 item 5: 1
    GAD7 item 6: 0
    GAD7 item 7: 0
  V8:
    PHQ9 item 1: 2
    PHQ9 item 2: 2
    PHQ9 item 3: 1
    PHQ9 item 4: 2
    PHQ9 item 5: 0
    PHQ9 item 6: 1
    PHQ9 item 7: 1
    PHQ9 item 8: 0
    PHQ9 item 9: 0

============================================================
Copy-paste format for experiment.py ground_truth dict:
(REVIEW items shown as None — resolve before pasting)

    # V1
    "PHQ9": {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 0},

    # V2
    "PHQ9": {1: 3, 2: 3, 3: 2, 4: 3, 5: 0, 6: 1, 7: 2, 8: 0, 9: 0},

    # V3
    "GAD7": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 0, 7: 0},

    # V4
    "GAD7": {1: 3, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3},

    # V5
    "GAD7": {1: 2, 2: 2, 3: 2, 4: 2, 5: 0, 6: 2, 7: 0},
    "PHQ9": {1: 2, 2: 2, 3: 2, 4: 2, 5: 0, 6: 0, 7: 2, 8: 0, 9: 0},

    # V6
    "GAD7": {1: 3, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3},
    "PHQ9": {1: 3, 2: 3, 3: 0, 4: 3, 5: 3, 6: 3, 7: 0, 8: 0, 9: 0},

    # V7
    "GAD7": {1: 0, 2: 2, 3: 2, 4: 0, 5: 1, 6: 0, 7: 0},

    # V8
    "PHQ9": {1: 2, 2: 2, 3: 1, 4: 2, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0},

============================================================
RELIABILITY VERDICT
============================================================
✓ PASS — kappa=1.000 meets the ≥0.80 threshold.
  Ground truth is sufficiently reliable to proceed.