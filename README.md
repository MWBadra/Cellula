#  Cellula Technologies - NLP Internship

Welcome to my repository for the **Cellula Technologies NLP Internship**. This repo contains all weekly tasks, research, and projects completed during the program, focusing on Large Language Models (LLMs), Model Optimization, and Full-Stack AI Deployment.

## 📂 Repository Structure

```text
├── Cellula_1week_[MohamedBadra]/    # Week 1: Transformers & Architectures
│   ├── BERT_Family/                 # Research on DistilBERT, ALBERT, LoRA, QLoRA
│   └── LSTM/                        # LSTM Model implementation & Report
│
└── Cellula_2week_[MohamedBadra]/    # Week 2: Optimization & Deployment
    ├── Quantization_techniques_Research/  # Task 0: Research on Model Quantization
    └── Toxic_content_classification_project/ # Task 1: End-to-End Toxic Content Classifier
```
## 📅 Week 1: Transformers & Architectures
**Focus:** Deep dive into the BERT family, Parameter-Efficient Fine-Tuning (PEFT), and Recurrent Neural Networks.

### **BERT Family Research**
* **Architecture Analysis:** Analyzed architectural differences between **DistilBERT**, **ALBERT**, and standard **BERT**.
* **Efficiency:** Explored **LoRA** (Low-Rank Adaptation) and **QLoRA** for memory-efficient fine-tuning.

### **LSTM Implementation**
* **Sequence Classification:** Built and trained a custom **LSTM** model from scratch to handle sequential data tasks.

---

## 📅 Week 2: Model Optimization & Deployment
**Focus:** Quantization techniques and building a full-stack AI application.

### 🔹 Task 0: Quantization Research
* **Goal:** Address the memory bottleneck of Large Language Models (LLMs).
* **Outcome:** Demonstrated how **Dynamic Quantization** reduces model size by **~2.8x** (from 255MB to 91MB) using PyTorch.
* **Key Concepts:** * Affine Quantization
    * Scale Factors & Zero Points
    * Int8 vs. FP32 Precision

### 🔹 Task 1: Toxic Content Classifier (Full-Stack App)
An end-to-end system that detects toxic content in images.
#### 🚀 Live Deployment
The code and the application are already deployed on **Hugging Face Spaces** and ready to be used.
* **Live Demo:** [Toxic Content Detector](https://huggingface.co/spaces/MWBadra/Toxic-content-detector)
  
#### ✨ Features
* **Image Analysis:** Uses **BLIP** (Bootstrapping Language-Image Pre-training) for image captioning to detect unsafe visual content.
* **Text Analysis:** Uses a fine-tuned **DistilBERT** model to classify text toxicity.
* **Database:** Logs all scan history (User input, Prediction, Confidence) to **MongoDB**.
* **UI/API:** Interactive web interface with history tracking and a scalable **FastAPI** backend.

#### 🛠️ Tech Stack
| Category | Tools |
| :--- | :--- |
| **AI / ML** | PyTorch, Transformers, PEFT |
| **Backend** | FastAPI, Python |
| **Frontend** | HTML, JavaScript |
| **Database** | MongoDB (Motor) |
| **DevOps** | Docker |
