"""
LLM Symptom Fidelity Experiment
================================
Study: "Testing the Simulated Patient: PHQ-9 and GAD-7 Fidelity in LLM-Driven Mental Health Personas"
Author: 
- Pathanin Opachalearn, Mahidol University, pathanin.opa@student.mahidol.ac.th
- Konlakorn Wongpatikaseree, Mahidol University, konlakorn.won@mahidol.ac.th
- Natthawut Kertkeidkachorn, Japan Advanced Institute of Science and Technology, natt@jaist.ac.jp

Experiment parameters:
  - Models: GPT-4o (OpenAI), Llama 3.3 - 70b versatile (Groq)
  - Vignettes: 8 (V1–V8)
  - Prompt conditions: standard, label_removed
  - Runs per cell: 3
  - Temperature: 0, seed=42 (GPT-4o)
  - Total trials: 8 × 2 × 2 × 3 = 96 maximum

Usage:
  pip install openai groq
  export OPENAI_API_KEY="sk-..."
  export GROQ_API_KEY="gsk_..."
  python experiment.py

Output:
  results.csv  — one row per item response (96 trials × 10 items = 960 rows)
"""

import os
import csv
import time
import re
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── API clients ──────────────────────────────────────────────────────────────
from openai import OpenAI
from groq import Groq

def _openai_client():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _groq_client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])

# ── Instrument definitions ────────────────────────────────────────────────────
PHQ9_ITEMS = [
    (1,  "Little interest or pleasure in doing things"),
    (2,  "Feeling down, depressed, or hopeless"),
    (3,  "Trouble falling or staying asleep, or sleeping too much"),
    (4,  "Feeling tired or having little energy"),
    (5,  "Poor appetite or overeating"),
    (6,  "Feeling bad about yourself — or that you are a failure or have let yourself or your family down"),
    (7,  "Trouble concentrating on things, such as reading the newspaper or watching television"),
    (8,  "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual"),
    (9,  "Thoughts that you would be better off dead, or of hurting yourself in some way"),
]

GAD7_ITEMS = [
    (1,  "Feeling nervous, anxious, or on edge"),
    (2,  "Not being able to stop or control worrying"),
    (3,  "Worrying too much about different things"),
    (4,  "Trouble relaxing"),
    (5,  "Being so restless that it's hard to sit still"),
    (6,  "Becoming easily annoyed or irritable"),
    (7,  "Feeling afraid, as if something awful might happen"),
]

ITEM_SCALE = "0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day"

# ── Vignettes ─────────────────────────────────────────────────────────────────
# Each vignette has:
#   text_standard     : full vignette including diagnosis label
#   text_label_removed: identical symptom detail, diagnosis/severity labels stripped
#   instruments       : list of "PHQ9" and/or "GAD7"
#   ground_truth      : dict mapping instrument → {item_number: expected_score}
#   expected_band     : dict mapping instrument → severity band string

VIGNETTES = {
    "V1": {
        "patient": "Mr. A (45M)",
        "condition": "MDD, moderate-severe",
        "text_standard": (
            "Mr. A is a 45-year-old man who presents to the clinic reporting a 4-week history of feeling "
            "\"down in the dumps\" and empty most of the day, nearly every day. He has markedly diminished "
            "interest in his usual hobbies, including golf and reading, nearly every day, stating he derives "
            "no pleasure from them anymore. He reports a 10-pound weight loss over the past month because he "
            "experiences a decrease in appetite nearly every day and has to force himself to eat. He struggles "
            "with terminal insomnia nearly every day, waking up at 3:00 AM unable to return to sleep, and "
            "experiences fatigue or loss of energy nearly every day. He is visibly slowed down in his speech "
            "and movements (psychomotor retardation) nearly every day and expresses excessive or inappropriate "
            "guilt nearly every day about \"failing\" his family despite being a reliable provider. His symptoms "
            "result in a diminished ability to concentrate at work nearly every day. "
            "[Diagnosis: Major Depressive Disorder, moderate to severe]"
        ),
        "text_label_removed": (
            "Mr. A is a 45-year-old man who presents to the clinic reporting a 4-week history of feeling "
            "\"down in the dumps\" and empty most of the day, nearly every day. He has markedly diminished "
            "interest in his usual hobbies, including golf and reading, nearly every day, stating he derives "
            "no pleasure from them anymore. He reports a 10-pound weight loss over the past month because he "
            "experiences a decrease in appetite nearly every day and has to force himself to eat. He struggles "
            "with terminal insomnia nearly every day, waking up at 3:00 AM unable to return to sleep, and "
            "experiences fatigue or loss of energy nearly every day. He is visibly slowed down in his speech "
            "and movements (psychomotor retardation) nearly every day and expresses excessive or inappropriate "
            "guilt nearly every day about \"failing\" his family despite being a reliable provider. His symptoms "
            "result in a diminished ability to concentrate at work nearly every day."
        ),
        "instruments": ["PHQ9"],
        "ground_truth": {
            "PHQ9": {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 0}
        },
        "expected_band": {"PHQ9": "Moderately severe–severe (15–27)"},
    },

    "V2": {
        "patient": "Ms. B (32F)",
        "condition": "MDD, mild-moderate",
        "text_standard": (
            "Ms. B is a 32-year-old woman who presents to her primary care physician reporting that she has "
            "felt sad, empty, and discouraged nearly every day for the past three weeks. She notes a marked loss "
            "of interest in her usual activities nearly every day, stating she no longer enjoys her weekly book "
            "club or gardening. During this same three-week period, she has experienced insomnia more than half "
            "the days, daytime fatigue nearly every day, and a diminished ability to concentrate on tasks at her "
            "accounting job more than half the days. She also ruminates on feelings of excessive guilt regarding "
            "her recent lack of productivity on several days. While her symptoms cause her significant distress, "
            "she is still managing to meet her basic role obligations at work and home, albeit with substantial "
            "effort and minor impairment. She has never experienced a manic or hypomanic episode. "
            "[Diagnosis: Major Depressive Disorder, mild to moderate]"
        ),
        "text_label_removed": (
            "Ms. B is a 32-year-old woman who presents to her primary care physician reporting that she has "
            "felt sad, empty, and discouraged nearly every day for the past three weeks. She notes a marked loss "
            "of interest in her usual activities nearly every day, stating she no longer enjoys her weekly book "
            "club or gardening. During this same three-week period, she has experienced insomnia more than half "
            "the days, daytime fatigue nearly every day, and a diminished ability to concentrate on tasks at her "
            "accounting job more than half the days. She also ruminates on feelings of excessive guilt regarding "
            "her recent lack of productivity on several days. While her symptoms cause her significant distress, "
            "she is still managing to meet her basic role obligations at work and home, albeit with substantial "
            "effort and minor impairment. She has never experienced a manic or hypomanic episode."
        ),
        "instruments": ["PHQ9"],
        "ground_truth": {
            "PHQ9": {1: 3, 2: 3, 3: 2, 4: 3, 5: 0, 6: 1, 7: 2, 8: 0, 9: 0}
        },
        "expected_band": {"PHQ9": "Mild–moderate (5–14)"},
    },

    "V3": {
        "patient": "Mr. C (34M)",
        "condition": "GAD, moderate",
        "text_standard": (
            "Mr. C is a 34-year-old man who reports an 8-month history of excessive worry about his job "
            "performance, his finances, and his children's safety occurring more days than not. He finds it "
            "difficult to control the worry, stating it \"takes over my brain\". Associated with this apprehensive "
            "expectation, he experiences muscle tension more days than not, feels \"keyed up\" on more than half "
            "the days, and is easily fatigued more days than not. He specifically reports having trouble relaxing "
            "on more than half the days and being so restless that it is hard to sit still on more than half the "
            "days. He also has difficulty falling asleep at night more days than not because his mind is racing "
            "with \"what ifs\". His anxiety has caused clinically significant distress, creating friction in his "
            "marriage and distracting him at work, though he still manages to maintain his overall job performance "
            "with effort. [Diagnosis: Generalized Anxiety Disorder, moderate]"
        ),
        "text_label_removed": (
            "Mr. C is a 34-year-old man who reports an 8-month history of excessive worry about his job "
            "performance, his finances, and his children's safety occurring more days than not. He finds it "
            "difficult to control the worry, stating it \"takes over my brain\". Associated with this apprehensive "
            "expectation, he experiences muscle tension more days than not, feels \"keyed up\" on more than half "
            "the days, and is easily fatigued more days than not. He specifically reports having trouble relaxing "
            "on more than half the days and being so restless that it is hard to sit still on more than half the "
            "days. He also has difficulty falling asleep at night more days than not because his mind is racing "
            "with \"what ifs\". His anxiety has caused clinically significant distress, creating friction in his "
            "marriage and distracting him at work, though he still manages to maintain his overall job performance "
            "with effort."
        ),
        "instruments": ["GAD7"],
        "ground_truth": {
            "GAD7": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 0, 7: 0}
        },
        "expected_band": {"GAD7": "Moderate (10–14)"},
    },

    "V4": {
        "patient": "Ms. D (28F)",
        "condition": "GAD, severe",
        "text_standard": (
            "Ms. D is a 28-year-old woman who presents to her primary care clinic reporting a 12-month history "
            "of excessive anxiety and worry occurring more days than not, focusing on a wide range of topics "
            "including her job security, her mounting credit card debt, her parents' health, and the possibility "
            "of getting into a car accident. She finds it difficult to control the worry. Accompanying this "
            "apprehensive expectation, she reports experiencing several physical and cognitive symptoms: she feels "
            "\"keyed up\" or on edge nearly every day, experiences muscle tension more days than not, and has sleep "
            "disturbance nearly every day, managing only a few hours of restless sleep each night. Consequently, she "
            "is easily fatigued nearly every day and complains of difficulty concentrating or that her mind goes blank "
            "nearly every day. She specifically notes having trouble relaxing nearly every day, being so restless that "
            "it is hard to sit still nearly every day, becoming easily annoyed or irritable nearly every day, and feeling "
            "afraid, as if something awful might happen nearly every day. Her symptoms cause clinically significant "
            "impairment in occupational functioning; her difficulty concentrating and fatigue have made it impossible "
            "for her to do her job efficiently, resulting in her recently being placed on a short-term disability leave. "
            "[Diagnosis: Generalized Anxiety Disorder, severe]"
        ),
        "text_label_removed": (
            "Ms. D is a 28-year-old woman who presents to her primary care clinic reporting a 12-month history "
            "of excessive anxiety and worry occurring more days than not, focusing on a wide range of topics "
            "including her job security, her mounting credit card debt, her parents' health, and the possibility "
            "of getting into a car accident. She finds it difficult to control the worry. Accompanying this "
            "apprehensive expectation, she reports experiencing several physical and cognitive symptoms: she feels "
            "\"keyed up\" or on edge nearly every day, experiences muscle tension more days than not, and has sleep "
            "disturbance nearly every day, managing only a few hours of restless sleep each night. Consequently, she "
            "is easily fatigued nearly every day and complains of difficulty concentrating or that her mind goes blank "
            "nearly every day. She specifically notes having trouble relaxing nearly every day, being so restless that "
            "it is hard to sit still nearly every day, becoming easily annoyed or irritable nearly every day, and feeling "
            "afraid, as if something awful might happen nearly every day. Her symptoms cause clinically significant "
            "impairment in occupational functioning; her difficulty concentrating and fatigue have made it impossible "
            "for her to do her job efficiently, resulting in her recently being placed on a short-term disability leave."
        ),
        "instruments": ["GAD7"],
        "ground_truth": {
            "GAD7": {1: 3, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3}
        },
        "expected_band": {"GAD7": "Severe (15–21)"},
    },

    "V5": {
        "patient": "Mr. E (50M)",
        "condition": "Comorbid MDD + GAD, moderate",
        "text_standard": (
            "Mr. E is a 50-year-old man who presents with a 6-week history of a major depressive episode, reporting "
            "depressed mood on more than half the days, a loss of interest in his usual activities on more than half "
            "the days, hypersomnia on more than half the days, fatigue on more than half the days, and a diminished "
            "ability to concentrate on more than half the days. Concurrently, he reports a 6-month history of excessive "
            "worry about his health and finances, occurring more days than not. He finds it difficult to control the "
            "worry. His anxiety is accompanied by three physical and cognitive symptoms: he feels keyed up or on edge on "
            "more than half the days, becomes easily annoyed or irritable on more than half the days, and experiences "
            "muscle tension more days than not. While his symptoms cause clinically significant distress and affect his "
            "work performance, his functional impairment is intermediate between mild and severe, allowing him to still "
            "manage his daily obligations with some effort. His generalized anxiety is severe enough to warrant clinical "
            "attention independently of his depressive symptoms. "
            "[Diagnosis: Comorbid Major Depressive Disorder and Generalized Anxiety Disorder, moderate]"
        ),
        "text_label_removed": (
            "Mr. E is a 50-year-old man who presents with a 6-week history of a major depressive episode, reporting "
            "depressed mood on more than half the days, a loss of interest in his usual activities on more than half "
            "the days, hypersomnia on more than half the days, fatigue on more than half the days, and a diminished "
            "ability to concentrate on more than half the days. Concurrently, he reports a 6-month history of excessive "
            "worry about his health and finances, occurring more days than not. He finds it difficult to control the "
            "worry. His anxiety is accompanied by three physical and cognitive symptoms: he feels keyed up or on edge on "
            "more than half the days, becomes easily annoyed or irritable on more than half the days, and experiences "
            "muscle tension more days than not. While his symptoms cause clinically significant distress and affect his "
            "work performance, his functional impairment is intermediate between mild and severe, allowing him to still "
            "manage his daily obligations with some effort. His generalized anxiety is severe enough to warrant clinical "
            "attention independently of his depressive symptoms."
        ),
        "instruments": ["PHQ9", "GAD7"],
        "ground_truth": {
            "PHQ9": {1: 2, 2: 2, 3: 2, 4: 2, 5: 0, 6: 0, 7: 2, 8: 0, 9: 0},
            "GAD7": {1: 2, 2: 2, 3: 2, 4: 2, 5: 0, 6: 2, 7: 0}
        },
        "expected_band": {
            "PHQ9": "Moderate (10–14)",
            "GAD7": "Moderate (10-14)"
        },
    },

    "V6": {
        "patient": "Ms. F (39F)",
        "condition": "Comorbid MDD + GAD, both severe",
        "text_standard": (
            "Ms. F is a 39-year-old woman who presents with a lifelong history of excessive, uncontrollable worry "
            "(apprehensive expectation) regarding everyday events that occurs more days than not for at least 6 months. "
            "Her anxiety is accompanied by all six of the DSM-5-TR associated physical and cognitive symptoms: she feels "
            "restless and keyed up or on edge nearly every day, is easily fatigued nearly every day, struggles with "
            "difficulty concentrating (her mind going blank) nearly every day, becomes easily annoyed or irritable "
            "nearly every day, experiences muscle tension nearly every day, and reports sleep disturbance nearly every day. "
            "This worry heavily impacts her daily functioning. Over the past three weeks, she has additionally developed "
            "exactly five symptoms characteristic of a major depressive episode. She reports feeling down, depressed, or "
            "hopeless most of the day, nearly every day; having little interest or pleasure in doing things nearly every day; "
            "a decrease in appetite resulting in weight loss nearly every day; feeling tired or having little energy nearly "
            "every day; and feeling bad about herself—or that she is a failure—nearly every day. Her generalized anxiety is "
            "severe enough to warrant clinical attention independently of her new depressive symptoms, distinguishing it "
            "from the anxiety that often temporarily accompanies depression. "
            "[Diagnosis: Comorbid Major Depressive Disorder and Generalized Anxiety Disorder, severe]"
        ),
        "text_label_removed": (
            "Ms. F is a 39-year-old woman who presents with a lifelong history of excessive, uncontrollable worry "
            "(apprehensive expectation) regarding everyday events that occurs more days than not for at least 6 months. "
            "Her anxiety is accompanied by all six of the DSM-5-TR associated physical and cognitive symptoms: she feels "
            "restless and keyed up or on edge nearly every day, is easily fatigued nearly every day, struggles with "
            "difficulty concentrating (her mind going blank) nearly every day, becomes easily annoyed or irritable "
            "nearly every day, experiences muscle tension nearly every day, and reports sleep disturbance nearly every day. "
            "This worry heavily impacts her daily functioning. Over the past three weeks, she has additionally developed "
            "exactly five symptoms characteristic of a major depressive episode. She reports feeling down, depressed, or "
            "hopeless most of the day, nearly every day; having little interest or pleasure in doing things nearly every day; "
            "a decrease in appetite resulting in weight loss nearly every day; feeling tired or having little energy nearly "
            "every day; and feeling bad about herself—or that she is a failure—nearly every day. Her generalized anxiety is "
            "severe enough to warrant clinical attention independently of her new depressive symptoms, distinguishing it "
            "from the anxiety that often temporarily accompanies depression."
        ),
        "instruments": ["PHQ9", "GAD7"],
        "ground_truth": {
            "PHQ9": {1: 3, 2: 3, 3: 0, 4: 3, 5: 3, 6: 3, 7: 0, 8: 0, 9: 0},
            "GAD7": {1: 3, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3}
        },
        "expected_band": {
            "PHQ9": "Moderate–severe (15+)",
            "GAD7": "Severe (15–21)"
        },
    },

    "V7": {
        "patient": "Mr. G (28M)",
        "condition": "GAD, mild (contrast case)",
        "text_standard": (
            "Mr. G is a 28-year-old man who presents with excessive worry about his work performance and his finances, "
            "occurring more days than not for the past 6 months. He finds it difficult to control the worry, which occurs "
            "without specific precipitants. His psychological distress is accompanied by three physical symptoms: "
            "restlessness on several days, muscle tension more days than not, and sleep disturbance more days than not. "
            "His symptoms cause him clinically significant distress, but he is still able to exert enough effort to "
            "maintain his job performance, demonstrating a mild, manageable level of impairment. "
            "He is diagnosed with Generalized Anxiety Disorder."
        ),
        "text_label_removed": (
            "Mr. G is a 28-year-old man who presents with excessive worry about his work performance and his finances, "
            "occurring more days than not for the past 6 months. He finds it difficult to control the worry, which occurs "
            "without specific precipitants. His psychological distress is accompanied by three physical symptoms: "
            "restlessness on several days, muscle tension more days than not, and sleep disturbance more days than not. "
            "His symptoms cause him clinically significant distress, but he is still able to exert enough effort to "
            "maintain his job performance, demonstrating a mild, manageable level of impairment."
        ),
        "instruments": ["GAD7"],
        "ground_truth": {
            "GAD7": {1: 0, 2: 2, 3: 2, 4: 0, 5: 1, 6: 0, 7: 0}
        },
        "expected_band": {"GAD7": "Mild (5–9)"},
    },

    "V8": {
        "patient": "Ms. H (40F)",
        "condition": "MDD, mild (contrast case)",
        "text_standard": (
            "Ms. H is a 40-year-old woman who presents to the clinic reporting that over the last 2 weeks, she has been "
            "bothered by feeling down, depressed, or hopeless on more than half the days. Rather than an inability to "
            "anticipate happiness, she notes having little interest or pleasure in doing things on more than half the days. "
            "She also reports feeling tired or having little energy on more than half the days. She experiences trouble "
            "falling or staying asleep on several days, and has trouble concentrating on things on several days. Instead "
            "of being consumed by self-loathing, she experiences rumination, feeling bad about herself—or that she is a "
            "failure—on several days. Her symptoms cause her clinically significant distress, but she only finds it "
            "\"somewhat difficult\" to do her work and take care of things at home, resulting in minor impairment. "
            "Because her symptoms result in only minor impairment and her overall symptom burden is low, her diagnosis "
            "is specified clinically as Major Depressive Disorder, single episode, mild."
        ),
        "text_label_removed": (
            "Ms. H is a 40-year-old woman who presents to the clinic reporting that over the last 2 weeks, she has been "
            "bothered by feeling down, depressed, or hopeless on more than half the days. Rather than an inability to "
            "anticipate happiness, she notes having little interest or pleasure in doing things on more than half the days. "
            "She also reports feeling tired or having little energy on more than half the days. She experiences trouble "
            "falling or staying asleep on several days, and has trouble concentrating on things on several days. Instead "
            "of being consumed by self-loathing, she experiences rumination, feeling bad about herself—or that she is a "
            "failure—on several days. Her symptoms cause her clinically significant distress, but she only finds it "
            "\"somewhat difficult\" to do her work and take care of things at home, resulting in minor impairment."
        ),
        "instruments": ["PHQ9"],
        "ground_truth": {
            "PHQ9": {1: 2, 2: 2, 3: 1, 4: 2, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0}
        },
        "expected_band": {"PHQ9": "Mild (5–9)"},
    },
}

# ── Severity band helpers ─────────────────────────────────────────────────────
PHQ9_BANDS = [
    (0,  4,  "Minimal"),
    (5,  9,  "Mild"),
    (10, 14, "Moderate"),
    (15, 19, "Moderately severe"),
    (20, 27, "Severe"),
]

GAD7_BANDS = [
    (0,  4,  "Minimal"),
    (5,  9,  "Mild"),
    (10, 14, "Moderate"),
    (15, 21, "Severe"),
]

def score_to_band(total: int, instrument: str) -> str:
    bands = PHQ9_BANDS if instrument == "PHQ9" else GAD7_BANDS
    for lo, hi, label in bands:
        if lo <= total <= hi:
            return label
    return "Out of range"

def bands_within_one(band_a: str, band_b: str, instrument: str) -> bool:
    """Returns True if two band labels are within one severity step of each other."""
    order = [b[2] for b in (PHQ9_BANDS if instrument == "PHQ9" else GAD7_BANDS)]
    if band_a not in order or band_b not in order:
        return False
    return abs(order.index(band_a) - order.index(band_b)) <= 1

# ── Prompt builders ───────────────────────────────────────────────────────────
SYSTEM_TEMPLATE = """You are roleplaying as the following patient. Stay in character throughout. 
Do not add clinical commentary, disclaimers, or meta-observations about your role.
Respond only as this patient would respond when asked questions about their symptoms.

PATIENT DESCRIPTION:
{vignette_text}"""

ITEM_TEMPLATE = """Over the last two weeks, how often have you been bothered by the following problem?

"{item_text}"

Respond with a single integer using this scale:
{scale}

Reply with the number only. Do not explain."""

# ── API call wrappers ─────────────────────────────────────────────────────────
def call_gpt4o(system_prompt: str, item_prompt: str) -> str:
    response = _openai_client().chat.completions.create(
        model="gpt-4o",
        temperature=0,
        seed=42,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": item_prompt},
        ],
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()

def call_llama33(system_prompt: str, item_prompt: str) -> str:
    response = _groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",   # adjust if using a different Llama 3.3 variant
        temperature=0,
        seed=42,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": item_prompt},
        ],
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()

MODEL_CALLERS = {
    "gpt4o":   call_gpt4o,
    "llama33": call_llama33,
}

# ── Response parser ───────────────────────────────────────────────────────────
def parse_score(raw: str) -> int | None:
    """Extract first integer 0–3 from raw model response. Returns None if unparseable."""
    match = re.search(r"\b([0-3])\b", raw)
    return int(match.group(1)) if match else None

# ── CSV setup ─────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "trial_id", "vignette_id", "patient", "condition",
    "model", "prompt_condition", "run_number",
    "instrument", "item_number", "item_text",
    "raw_response", "parsed_score", "ground_truth_score", "deviation",
]

def make_trial_id(vignette_id, model, prompt_condition, run, instrument, item_num):
    return f"{vignette_id}_{model}_{prompt_condition}_r{run}_{instrument}_i{item_num}"

# ── Main experiment loop ──────────────────────────────────────────────────────
def run_experiment(
    output_path: str = "results.csv",
    n_runs: int = 3,
    models: list = None,
    vignette_ids: list = None,
    dry_run: bool = False,
):
    """
    Args:
        output_path  : path to output CSV
        n_runs       : number of runs per cell (default 3)
        models       : subset of ["gpt4o", "llama33"] to run; None = both
        vignette_ids : subset of vignette keys to run; None = all
        dry_run      : if True, print prompts without calling APIs
    """
    models        = models        or ["gpt4o", "llama33"]
    vignette_ids  = vignette_ids  or list(VIGNETTES.keys())
    conditions    = ["standard", "label_removed"]

    instrument_items = {"PHQ9": PHQ9_ITEMS, "GAD7": GAD7_ITEMS}

    total_calls = (
        len(vignette_ids) * len(models) * len(conditions) * n_runs
        * sum(
            sum(len(instrument_items[inst]) for inst in VIGNETTES[v]["instruments"])
            for v in vignette_ids
        ) // len(vignette_ids)
    )
    print(f"[INFO] Starting experiment | vignettes={vignette_ids} | models={models} "
          f"| runs={n_runs} | estimated_api_calls≈{total_calls}")

    file_exists = os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        for vignette_id in vignette_ids:
            vignette = VIGNETTES[vignette_id]
            print(f"\n{'='*60}")
            print(f"Vignette: {vignette_id} | {vignette['patient']} | {vignette['condition']}")

            for model_name in models:
                caller = MODEL_CALLERS[model_name]

                for condition in conditions:
                    vignette_text = (
                        vignette["text_standard"]
                        if condition == "standard"
                        else vignette["text_label_removed"]
                    )
                    system_prompt = SYSTEM_TEMPLATE.format(vignette_text=vignette_text)

                    for run in range(1, n_runs + 1):
                        print(f"  [{model_name}] [{condition}] run {run}/{n_runs}")

                        for instrument in vignette["instruments"]:
                            items = instrument_items[instrument]
                            gt_scores = vignette["ground_truth"].get(instrument, {})

                            for item_num, item_text in items:
                                item_prompt = ITEM_TEMPLATE.format(
                                    item_text=item_text,
                                    scale=ITEM_SCALE,
                                )

                                if dry_run:
                                    print(f"    [DRY RUN] {instrument} item {item_num}: {item_text[:50]}...")
                                    raw_response  = "DRY_RUN"
                                    parsed_score  = None
                                else:
                                    try:
                                        raw_response = caller(system_prompt, item_prompt)
                                        time.sleep(2.1)  # gentle rate limiting
                                    except Exception as e:
                                        print(f"    [ERROR] {e}")
                                        raw_response = f"ERROR: {e}"

                                    parsed_score = parse_score(raw_response)

                                gt_score  = gt_scores.get(item_num)
                                deviation = (
                                    abs(parsed_score - gt_score)
                                    if parsed_score is not None and gt_score is not None
                                    else None
                                )

                                row = {
                                    "trial_id":        make_trial_id(
                                                           vignette_id, model_name,
                                                           condition, run, instrument, item_num),
                                    "vignette_id":     vignette_id,
                                    "patient":         vignette["patient"],
                                    "condition":       vignette["condition"],
                                    "model":           model_name,
                                    "prompt_condition": condition,
                                    "run_number":      run,
                                    "instrument":      instrument,
                                    "item_number":     item_num,
                                    "item_text":       item_text,
                                    "raw_response":    raw_response,
                                    "parsed_score":    parsed_score,
                                    "ground_truth_score": gt_score,
                                    "deviation":       deviation,
                                }
                                writer.writerow(row)
                                csvfile.flush()  # write immediately; safe to interrupt

    print(f"\n[DONE] Results saved to: {output_path}")


# ── Quick analysis after experiment ──────────────────────────────────────────
def analyze_results(csv_path: str = "results.csv"):
    """
    Reads results.csv and prints:
      - Per-vignette, per-model, per-condition MAD
      - Instrument-level severity band + pass/fail
      - Bias flag (standard vs label_removed score divergence)
    """
    import csv as csv_mod
    from collections import defaultdict

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("No data found.")
        return

    # Group: (vignette_id, model, prompt_condition, instrument, run_number) → list of (parsed, gt)
    # Keyed by run so we can sum per-run totals correctly
    run_groups = defaultdict(list)
    for r in rows:
        key = (r["vignette_id"], r["model"], r["prompt_condition"],
               r["instrument"], r["run_number"])
        ps = int(r["parsed_score"])        if r["parsed_score"]        not in ("", "None", None) else None
        gt = int(r["ground_truth_score"])  if r["ground_truth_score"]  not in ("", "None", None) else None
        run_groups[key].append((ps, gt))

    # Collapse to (vid, model, cond, inst) → list of per-run (llm_total, gt_total)
    # Each run total = sum of item scores for that run
    cell_runs = defaultdict(list)   # (vid, model, cond, inst) → [(llm_run_total, gt_run_total), ...]
    for (vid, model, cond, inst, run), pairs in run_groups.items():
        valid = [(ps, gt) for ps, gt in pairs if ps is not None and gt is not None]
        if not valid:
            continue
        llm_run_total = sum(ps for ps, _ in valid)
        gt_run_total  = sum(gt for _, gt in valid)
        cell_runs[(vid, model, cond, inst)].append((llm_run_total, gt_run_total))

    print("\n" + "="*70)
    print("FIDELITY ANALYSIS")
    print("="*70)

    # MAD and severity band — averaged across runs
    for (vid, model, cond, inst), run_totals in sorted(cell_runs.items()):
        # MAD: average item-level deviation across all runs
        all_pairs = []
        for (vid2, model2, cond2, inst2, run), pairs in run_groups.items():
            if (vid2, model2, cond2, inst2) == (vid, model, cond, inst):
                all_pairs.extend([(ps, gt) for ps, gt in pairs
                                  if ps is not None and gt is not None])
        mad = sum(abs(ps - gt) for ps, gt in all_pairs) / len(all_pairs) if all_pairs else None

        # Mean total score across runs for band assignment
        mean_llm_total = sum(t[0] for t in run_totals) / len(run_totals)
        mean_gt_total  = sum(t[1] for t in run_totals) / len(run_totals)
        llm_band = score_to_band(round(mean_llm_total), inst)
        gt_band  = score_to_band(round(mean_gt_total),  inst)
        within_1 = bands_within_one(llm_band, gt_band, inst)

        print(f"\n{vid} | {model} | {cond} | {inst}")
        print(f"  MAD: {mad:.3f}" if mad is not None else "  MAD: N/A")
        print(f"  LLM mean total: {mean_llm_total:.1f} ({llm_band})  |  "
              f"GT mean total: {mean_gt_total:.1f} ({gt_band})")
        print(f"  Within-one-band: {'PASS ✓' if within_1 else 'FAIL ✗'}")

    # ── Bias detection ────────────────────────────────────────────────────────
    # For each (vid, model, inst): compare mean run-total under standard vs label_removed
    # Delta is on the instrument total scale (PHQ-9: 0–27, GAD-7: 0–21)
    print("\n" + "-"*70)
    print("BIAS DETECTION (standard vs label_removed divergence)")
    print("Threshold: |Δ| ≥ 3 total-score points flags potential label bias")
    print("-"*70)

    bias_data = defaultdict(lambda: {"standard": [], "label_removed": []})
    for (vid, model, cond, inst), run_totals in cell_runs.items():
        llm_run_totals = [t[0] for t in run_totals]
        bias_data[(vid, model, inst)][cond].extend(llm_run_totals)

    for (vid, model, inst), cond_totals in sorted(bias_data.items()):
        std_totals = cond_totals.get("standard", [])
        lbl_totals = cond_totals.get("label_removed", [])
        if not std_totals or not lbl_totals:
            continue
        std_mean = sum(std_totals) / len(std_totals)
        lbl_mean = sum(lbl_totals) / len(lbl_totals)
        delta = std_mean - lbl_mean
        # Threshold of 3 points on instrument total is meaningful:
        # ~1 band boundary on PHQ-9 (bands span 5pts), proportionally similar on GAD-7
        flag = "⚠️  LABEL BIAS" if abs(delta) >= 3 else "OK"
        print(f"  {vid} | {model} | {inst}: "
              f"std={std_mean:.1f}  lbl={lbl_mean:.1f}  Δ={delta:+.1f}  {flag}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Symptom Fidelity Experiment")
    parser.add_argument("--output",    default="results.csv",    help="Output CSV path")
    parser.add_argument("--runs",      type=int, default=3,      help="Runs per cell")
    parser.add_argument("--models",    nargs="+", default=None,  help="gpt4o llama33")
    parser.add_argument("--vignettes", nargs="+", default=None,  help="V1 V2 ... V8")
    parser.add_argument("--dry-run",   action="store_true",      help="Print without calling APIs")
    parser.add_argument("--analyze",   action="store_true",      help="Run analysis on existing CSV")
    args = parser.parse_args()

    if args.analyze:
        analyze_results(args.output)
    else:
        run_experiment(
            output_path=args.output,
            n_runs=args.runs,
            models=args.models,
            vignette_ids=args.vignettes,
            dry_run=args.dry_run,
        )
