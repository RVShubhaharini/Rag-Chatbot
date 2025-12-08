'''
import google.generativeai as genai

# Put your Gemini API key here
genai.configure(api_key="AIzaSyAJmrtPwOX6_QVgb_YVoX_3R2BFndDF1to")

# Use a lightweight model for quick testing
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Hello Gemini! Can you confirm you're working?")
print("Model response:", response.text)


import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ollama   # <-- swapped google.generativeai with ollama

# --------------------
# CONFIG
# --------------------

# Load Flan model (example: Flan-T5 Base)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# --------------------
# Function: Flan generates answer
# --------------------
def generate_flan_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --------------------
# Function: Gemma evaluates semantic similarity (via Ollama)
# --------------------
def evaluate_with_gemma(expected, predicted):
    # Prompt Gemma to evaluate overlap
    prompt = f"""
    Compare the following two answers semantically. 
    Expected Answer: {expected}
    Predicted Answer: {predicted}

    Task: 
    - Check if predicted answer contains the key meaning/keywords of expected answer.
    - Give a similarity score between 0 and 1 (1 = perfect match, 0 = no similarity).
    - Also explain briefly why.

    Return JSON in format:
    {{
        "score": float,
        "explanation": "string"
    }}
    """
    response = ollama.chat(
        model="gemma:2b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


# --------------------
# Example Run (single QA pair)
# --------------------
if __name__ == "__main__":
    # Example question + expected answer from GSM8K dataset
    question = "What is the capital of France?"
    expected_answer = "Paris"

    # Step 1: Flan generates
    predicted_answer = generate_flan_answer(question)
    print("Flan Predicted:", predicted_answer)

    # Step 2: Gemma evaluates
    evaluation = evaluate_with_gemma(expected_answer, predicted_answer)
    print("Gemma Evaluation:", evaluation)
'''

import re
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ollama

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate

# -------------------------------
# 1. Load GSM8K dataset
# -------------------------------
dataset = load_dataset("gsm8k", "main", split="test").select(range(1))
# You can uncomment these if you want to debug dataset shapes
# dataset11 = load_dataset("gsm8k", "main", split="test")
# dataset12 = load_dataset("gsm8k", "main")
# print("Dataset11 length:", len(dataset11))
# print("Column11 names:", dataset11.column_names)
# print("Dataset12 length:", len(dataset12))
# print("Column12 names:", dataset12.column_names)

# -------------------------------
# 2. Load Flan model
# -------------------------------
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_flan_answer(prompt: str) -> str:
    inputs = flan_tokenizer(prompt, return_tensors="pt")
    outputs = flan_model.generate(**inputs, max_new_tokens=50)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# 3. Custom Gemma Evaluator for DeepEval
# -------------------------------

class CustomResponse:
    """Minimal wrapper so DeepEval sees .output_text, .statements, and .verdicts"""
    def __init__(self, text: str):
        self.output_text = text
        self.statements = [text]
        self.verdicts = [text]  # ✅ Added to satisfy AnswerRelevancyMetric


class GemmaEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name="gemma:2b-instruct"):
        self.model_name = model_name

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self):
        return None  # Ollama handles model serving

    def generate(self, prompt: str, **kwargs):
        resp = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp["message"]["content"]
        return CustomResponse(text)  # ✅ wrap response

    async def a_generate(self, prompt: str, **kwargs):
        return self.generate(prompt, **kwargs)

# -------------------------------
# 4. Evaluation Logic (numeric + semantic fallback)
# -------------------------------
def evaluate_math_answer(question: str, predicted: str, expected: str):
    # 4a: Direct numeric check
    num_pred = re.findall(r"-?\d+", predicted)
    num_exp = re.findall(r"-?\d+", expected)
    if num_pred and num_exp and num_pred[-1] == num_exp[-1]:
        return {"score": 1.0, "explanation": "Exact numerical match."}

    # 4b: Semantic similarity via Gemma
    prompt = (
        f"Question: {question}\n"
        f"Expected Answer: {expected}\n"
        f"Predicted Answer: {predicted}\n\n"
        "Task:\n"
        "- Score similarity between predicted and expected answer (0.0–1.0)\n"
        "- Explain briefly why\n\n"
        "Return JSON only in format:\n"
        "{\"score\": float, \"explanation\": \"...\"}"
    )
    response = GemmaEvaluator().generate(prompt)
    try:
        return json.loads(response.output_text)   # ✅ fixed: use .output_text
    except Exception:
        return {"score": 0.0, "explanation": "Gemma response not in JSON format."}

# -------------------------------
# 5. Run Evaluation
# -------------------------------
test_cases = []
for ex in dataset:
    q, expected = ex["question"], ex["answer"]
    pred = generate_flan_answer(q)

    tc = LLMTestCase(input=q, actual_output=pred, expected_output=expected)
    test_cases.append((tc, q, expected))

# -------------------------------
# 6. DeepEval metrics (using Gemma instead of OpenAI)
# -------------------------------
gemma_model = GemmaEvaluator()
metrics = [AnswerRelevancyMetric(model=gemma_model)]

results = evaluate([tc for tc, _, _ in test_cases], metrics)

for idx, ((tc, q, expected), res) in enumerate(zip(test_cases, results)):
    print(f"\n=== Example {idx+1} ===")
    print("Q:", q)
    print("Flan Answer:", tc.actual_output)
    print("Expected:", expected)

    # DeepEval results are dicts
    for metric_result in res:
        print(f"DeepEval ({metric_result['metric']}) Score:", metric_result['score'])

    # Post-hoc numeric/semantic evaluation
    semantic = evaluate_math_answer(q, tc.actual_output, expected)
    print("Post-hoc Numeric/Semantic Score:", semantic["score"])
    print("Reason:", semantic["explanation"])
