import json
import os
import re
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Absolute path for saving images locally
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def is_valid_answer(answer):
    answer = answer.strip()
    # Only intercept empty values
    return bool(answer)


def clean_answer(question, answer):
    answer = answer.strip()

    # 1. Remove brackets/quotes
    answer = re.sub(r'[\[\]"\']', '', answer)

    # 2. Standardize and intercept boolean values
    if answer.lower() in ("true", "false"):
        return "Yes" if answer.lower() == "true" else "No"
    if answer.lower() in ("yes", "no"):
        return answer.capitalize()

    # 3. Numerical cleaning
    original_had_percent = '%' in answer
    try:
        cleaned = answer.replace(",", "").replace("%", "")
        num = float(cleaned)

        # Convergence fix: Standardize percentage representation
        has_percent_context = bool(re.search(r'percent|percentage|%', question, re.I))
        if has_percent_context and not original_had_percent and 0 < num < 1:
            num *= 100

        # Never truncate decimal precision
        if num.is_integer():
            return str(int(num))
        else:
            return str(num)

    except ValueError:
        # If still cannot be converted to float after removing symbols, it's pure text
        pass

    return answer


def format_instruction(question):
    return f"<image>Answer the following question based on the chart: {question} Keep your answer short (a single word, number, or short phrase only). Output in the format: Answer: <answer>"


def generate_dataset(split_name, output_path, max_samples):
    """Generate dataset from specified split"""
    print(f"Loading ChartQA {split_name} dataset...")
    dataset = load_dataset("HuggingFaceM4/ChartQA", split=split_name)
    processed_data = []
    skipped_count = 0
    img_count = 0

    print(f"Starting filtering and reconstruction of {split_name} data...")
    for item in dataset:
        if len(processed_data) >= max_samples:
            break
        question = item['query']
        original_answer = item['label'][0]
        image = item['image']

        if not is_valid_answer(original_answer):
            skipped_count += 1
            continue

        img_name = f"{split_name}_chart_{img_count}.png"
        abs_img_path = os.path.join(IMG_DIR, img_name)

        # Save image using absolute path
        image.save(abs_img_path)
        img_count += 1

        instruction = format_instruction(question)
        clean_ans = clean_answer(question, original_answer)
        final_answer = f"Answer: {clean_ans}"

        # 🔥 KEY FIX: Save relative path in JSON for cross-platform compatibility
        rel_img_path = f"data/images/{img_name}"

        sample = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": final_answer}
            ],
            "images": [rel_img_path]  # Use relative path here
        }

        processed_data.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(
        f"✅ Completed! Total {len(processed_data)} {split_name} samples "
        f"Skipped {skipped_count} samples, saved to {output_path}")


if __name__ == "__main__":
    train_output_path = os.path.join(PROJECT_ROOT, "data", "train.json")
    test_output_path = os.path.join(PROJECT_ROOT, "data", "test.json")

    generate_dataset("train", train_output_path, 5000)
    generate_dataset("test", test_output_path, 500)
