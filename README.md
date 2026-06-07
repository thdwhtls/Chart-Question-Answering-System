# Chart-Question-Answering-System
Chart Question Answering System - A vertical domain fine-tuning project for Qwen2-VL to alleviate hallucination issues in chart-question answering. Achieved 61.4% accuracy improvement (55%→61.4%) on ChartQA dataset through structured task design and LoRA fine-tuning. End-to-end deployment with Gradio WebUI for interactive chart analysis.


## 📊 Project Overview
This project focuses on improving the accuracy of answering the question from charts using Qwen2-VL through:
- **Data Reconstruction**: 5,000 high-quality instruction pairs from ChartQA dataset
- **Structured Task Design**: Converting open-ended QA to structured tasks
- **LoRA Fine-tuning**: Lightweight adaptation of Qwen2-VL-2B-Instruct
- **WebUI Deployment**: Interactive chart QA interface with Gradio
## 🚀 Key Achievements
- **Accuracy Improvement**: 55% → 61.4% (Exact Match) on test set
- **Resource Efficiency**: Fine-tuned on 2B model with limited computational resources
- **Hallucination Reduction**: Significant improvement in numerical accuracy
- **End-to-End Solution**: From data preparation to deployment
## 🛠️ Tech Stack
- **Model**: Qwen2-VL-2B-Instruct
- **Fine-tuning**: LoRA (Parameter-Efficient Fine-tuning)
- **Framework**: PyTorch, Hugging Face Transformers
- **Dataset**: ChartQA (HuggingFaceM4/ChartQA)
- **Deployment**: Gradio WebUI
- **Language**: Python

## 📁 Project Structure
chartqa-qa-system/
├── data/
│   ├── images/          # Chart images
│   ├── train.json      # 5,000 training samples
│   └── test.json       # 500 test samples
├── scripts/
│   ├── data_preprocessing.py  # Data cleaning and formatting
│   ├── evaluation.py         # Model evaluation
│   └── merge_model.py        # LoRA weight merging
│   └── app.py                # Gradio interface
│   └── export.yaml           # LoRA weights merging and model export config
# Gradio interface
├── output/
│   └── merged_model/         # Fine-tuned model weights
└── README.md


🚀 Quick Start
1. Clone the repository
git clone https://github.com/yourusername/chartqa-qa-system.git
cd chartqa-qa-system
2. Install dependencies
pip install -r requirements.txt
3. Run the WebUI
python webui/WebUI.py
4. Access the interface
Open your browser and navigate to: http://localhost:7861
📊 Usage Example
Upload a chart image (pie chart, bar chart, line chart, etc.)
Enter your question about the chart
Get the model's answer 
Verify the accuracy
📈 Results
Metric	  Base Model	 Fine-tuned Model	
Accuracy	55.0%	       61.4%	
Hallucination Rate	High	Significantly Reduced	
🔧 Fine-tuning Details
Target Layers: q_proj, k_proj, v_proj, o_proj, visual.patch_embed.proj, visual.merger.mlp, visual.blocks.*.attn, visual.blocks.*.mlp
Training Data: 5,000 structured instruction pairs
Evaluation: 500 test samples with numerical matching algorithm (5% tolerance)
