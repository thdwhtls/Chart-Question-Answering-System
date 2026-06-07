import json
import os
import re
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"
FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "merged_model")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test.json")


def deep_find_value(obj):
    """Recursively find 'value' key in dictionary/list"""
    if isinstance(obj, dict):
        if "value" in obj:
            return obj["value"]
        for v in obj.values():
            res = deep_find_value(v)
            if res is not None:
                return res
    elif isinstance(obj, list) and len(obj) > 0:
        for item in obj:
            res = deep_find_value(item)
            if res is not None:
                return res
    return None


def normalize_answer(answer_str):
    s = answer_str.strip()
    s = re.sub(r'^answer:\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[\[\]"]', '', s)
    s = s.strip('.,;:!?，。；：！？\'"')
    try:
        num = float(s)
        if '.' in s:
            return str(round(num, 2))
        return str(int(num))
    except ValueError:
        s_lower = s.lower()
        if s_lower in ("yes", "true"):
            return "yes"
        if s_lower in ("no", "false"):
            return "no"
    return s.lower().strip()


def extract_number(text):
    text = text.strip().replace(",", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
        if match:
            return float(match.group())
        return None


def is_likely_year(text):
    return bool(re.fullmatch(r'(19|20)\d{2}', text))


def answers_match(true_answer, predicted_answer):
    true_norm = true_answer.strip().lower().replace(",", "")
    pred_norm = predicted_answer.strip().lower().replace(",", "")
    true_norm = re.sub(r'^answer:\s*', '', true_norm)
    pred_norm = re.sub(r'^answer:\s*', '', pred_norm)

    if true_norm == pred_norm:
        return True

    if is_likely_year(true_norm):
        return False

    true_num = extract_number(true_norm)
    pred_num = extract_number(pred_norm)

    if true_num is not None and pred_num is not None:
        if true_num == 0:
            return pred_num == 0
        return abs(true_num - pred_num) / abs(true_num) <= 0.05

    return False


def evaluate(model, processor, test_data_path):
    correct = 0
    total = 0
    mismatch_count = 0
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    for item in test_data:
        # Read relative path from JSON
        image_rel_path = item["images"][0]
        # Convert to absolute path for image loading
        image_abs_path = os.path.join(PROJECT_ROOT, image_rel_path)

        raw_content = item["messages"][0]["content"]
        if isinstance(raw_content, list):
            question = "".join([c["text"] for c in raw_content if c["type"] == "text"])
        else:
            question = raw_content
        question = re.sub(r'<image>', '', question).strip()
        true_answer = item["messages"][1]["content"].strip()

        # Use absolute path here
        image = Image.open(image_abs_path).convert("RGB")
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        predicted_answer = output.strip()
        pred_normalized = normalize_answer(predicted_answer)
        true_normalized = normalize_answer(true_answer)

        if answers_match(true_normalized, pred_normalized):
            correct += 1
        else:
            mismatch_count += 1
        total += 1

    return correct / total


# ================= Evaluate the Base model =================
print("Loading base model...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
base_processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
print("\n========== Evaluating base model ==========")
base_acc = evaluate(base_model, base_processor, TEST_DATA_PATH)
print(f"\nBase model accuracy: {base_acc:.2%}")

del base_model
torch.cuda.empty_cache()
print("\nBase model memory released, starting to load fine-tuned model...")


# ================= Evaluate the Fine-tuned model =================
finetuned_model = Qwen2VLForConditionalGeneration.from_pretrained(
    FINETUNED_MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
finetuned_processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_PATH)
print("\n========== Evaluating fine-tuned model ==========")
finetuned_acc = evaluate(finetuned_model, finetuned_processor, TEST_DATA_PATH)
print(f"\nFine-tuned model accuracy: {finetuned_acc:.2%}")
