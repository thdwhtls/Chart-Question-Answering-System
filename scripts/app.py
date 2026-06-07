import os
import gradio as gr
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Dynamically get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "merged_model")

print("Loading fine-tuned model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded successfully!")


def predict(image, question):
    """Perform question answering on the uploaded chart"""
    if image is None:
        return "Please upload a chart image first"
    if not question.strip():
        return "Please enter a question"

    image = Image.open(image).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output.strip()


# Preset example questions
EXAMPLE_QUESTIONS = [
    "What is the value of the highest bar?",
    "What is the title of this chart?",
    "Which category has the lowest value?",
    "Extract all the numerical values from this chart in JSON format.",
    "What is the difference between the maximum and minimum values?",
]

with gr.Blocks(
        title="Chart QA System",
        theme=gr.themes.Soft()
) as demo:
    gr.Markdown("""
    # 📊 Chart Numerical Extraction and Reasoning System
    Based on fine-tuned Qwen2-VL-2B-Instruct, optimized for numerical extraction and reasoning tasks on complex charts.

    **How to use:** Upload a chart image and enter a question to get the answer.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="📊 Upload Chart Image")
            question_input = gr.Textbox(
                label="❓ Enter Question",
                placeholder="Example: What is the value of the highest bar?",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=question_input,
                label="💡 Example Questions (Click to auto-fill)"
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="📝 Model Answer",
                lines=8,
                show_copy_button=True
            )

    submit_btn.click(
        fn=predict,
        inputs=[image_input, question_input],
        outputs=output_text
    )
    clear_btn.click(
        fn=lambda: (None, "", ""),
        inputs=[],
        outputs=[image_input, question_input, output_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
