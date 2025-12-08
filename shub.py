# shub_fixed.py
import re
from datasets import load_dataset
from transformers import pipeline
import torch

# ------------------------------
# Device selection (GPU if available, otherwise CPU)
# ------------------------------
def choose_device():
    if torch.cuda.is_available():
        print("Device set to use cuda")
        return 0
    else:
        print("Device set to use cpu")
        return -1

device = choose_device()

# ------------------------------
# Model & pipeline (text2text-generation)
# ------------------------------
MODEL_NAME = "google/flan-t5-base"   # keep/change as you prefer
print("Loading model:", MODEL_NAME)
flan = pipeline("text2text-generation", model=MODEL_NAME, max_new_tokens=256, device=device)

# ------------------------------
# Load GSM8K dataset (inspect to be robust)
# ------------------------------
ds = load_dataset("gsm8k", "main")
print("Loaded dataset splits:", list(ds.keys()))
print("Dataset columns (test):", ds["test"].column_names)
print("Example element (test)[0]:\n", ds["test"][0])

# Small sample for debugging / quick runs (change n_samples as needed)
n_samples = 20
n_samples = min(n_samples, len(ds["test"]))
samples = ds["test"].select(range(n_samples))

# ------------------------------
# Few-shot examples and prompt templates
# ------------------------------
few_shot_examples = [
    {"q": "If a pen costs 2 dollars and I buy 3, how much do I spend?", "a": "6"},
    {"q": "A train has 10 compartments, each with 20 seats. How many seats in total?", "a": "200"}
]

def generate_prompts(question):
    normal = f"Answer the following math word problem:\n\n{question}\nAnswer:"
    few_shot = "Solve math problems step by step:\n"
    for ex in few_shot_examples:
        few_shot += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    few_shot += f"Q: {question}\nA:"
    cot = f"Solve step by step. Show reasoning, then final answer.\n\nQuestion: {question}\nAnswer:"
    few_shot_cot = "Solve math problems step by step with reasoning:\n"
    for ex in few_shot_examples:
        few_shot_cot += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    few_shot_cot += f"Q: {question}\nA:"
    return normal, few_shot, cot, few_shot_cot

def get_answer(prompt):
    out = flan(prompt, do_sample=False)
    # pipeline returns a list of outputs; take the text of the first
    return out[0].get("generated_text", str(out[0]))

# ------------------------------
# Utilities: extract numeric answer & normalize text
# ------------------------------
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')

def extract_number(text):
    """Return last numeric token as int or float, or None if no number found."""
    if text is None:
        return None
    s = str(text).replace(",", "")  # remove commas in numbers like "1,000"
    found = num_re.findall(s)
    if not found:
        return None
    last = found[-1]
    try:
        if "." in last:
            return float(last)
        else:
            return int(last)
    except Exception:
        try:
            return float(last)
        except Exception:
            return None

def normalize_text(s):
    return re.sub(r'\s+', ' ', str(s)).strip().lower()

# ------------------------------
# Run generation on samples and store results
# ------------------------------
results = {"normal": [], "few_shot": [], "cot": [], "few_shot_cot": []}

for idx, item in enumerate(samples):
    # Robust extraction of question and answer from sample element
    if isinstance(item, dict):
        # Common keys: 'question' and 'answer' for GSM8K
        question = item.get("question") or item.get("Q") or item.get("input") or next(iter(item.values()))
        correct_answer = item.get("answer") or item.get("answer_text") or item.get("output") or item.get("target") or None
    else:
        s = str(item)
        if "\t" in s:
            parts = s.split("\t")
        elif "|" in s:
            parts = s.split("|")
        else:
            parts = [s]
        question = parts[0]
        correct_answer = parts[1] if len(parts) > 1 else None

    normal, few_shot, cot, few_shot_cot = generate_prompts(question)

    pred_normal = get_answer(normal)
    pred_few = get_answer(few_shot)
    pred_cot = get_answer(cot)
    pred_few_cot = get_answer(few_shot_cot)

    results["normal"].append({"q": question, "gt": correct_answer, "pred": pred_normal})
    results["few_shot"].append({"q": question, "gt": correct_answer, "pred": pred_few})
    results["cot"].append({"q": question, "gt": correct_answer, "pred": pred_cot})
    results["few_shot_cot"].append({"q": question, "gt": correct_answer, "pred": pred_few_cot})

    # optional: small progress print
    if (idx + 1) % 5 == 0 or (idx + 1) == n_samples:
        print(f"Generated {idx+1}/{n_samples} samples")

# ------------------------------
# Scoring: numeric match (primary) or substring fallback
# ------------------------------
def is_match(pred_text, gt_text, float_tol=1e-3):
    # try numeric comparison first
    pnum = extract_number(pred_text)
    gnum = extract_number(gt_text)
    if (pnum is not None) and (gnum is not None):
        # compare numerically
        try:
            return abs(float(pnum) - float(gnum)) <= float_tol
        except Exception:
            return str(pnum) == str(gnum)
    # fallback: normalized substring match (looser)
    if gt_text is None:
        return False
    return normalize_text(str(gt_text)) in normalize_text(str(pred_text))

def evaluate_results(results_list):
    total = len(results_list)
    if total == 0:
        return 0.0, []
    correct = 0
    mismatches = []
    for r in results_list:
        if is_match(r["pred"], r["gt"]):
            correct += 1
        else:
            mismatches.append(r)
    return correct / total, mismatches

# ------------------------------
# Print accuracies and examples
# ------------------------------
for method in results:
    acc, mismatches = evaluate_results(results[method])
    print(f"{method} accuracy: {acc:.2f} ({len(results[method])} examples)")
    print("Up to 5 mismatches for this method:")
    for m in mismatches[:5]:
        print("Q:", m["q"])
        print("GT:", m["gt"])
        print("Pred:", m["pred"])
        print("-----")
    print()
