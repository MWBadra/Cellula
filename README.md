#  Cellula Technologies - NLP Internship

Welcome to my repository for the **Cellula Technologies NLP Internship**. This repo contains all weekly tasks, research, and projects completed during the program, focusing on Large Language Models (LLMs), Model Optimization, and Full-Stack AI Deployment.


```
## 📅 Week 5: API Deployment & Containerization (MLOps)
**Focus:** Converting a local LangGraph AI agent into a production-ready web service using FastAPI and Docker.

### 🔹 Enterprise Backend Architecture
Upgraded the autonomous Self-Learning RAG system into a modular, containerized API.
* **FastAPI Integration:** Built a highly concurrent web server with a `/query` POST endpoint that returns both the AI's response and the internal routing path (Explain vs. Generate vs. Teach).
* **Modular Codebase:** Refactored a single monolithic script into a professional multi-directory architecture (`api/`, `graph/`, `rag/`).
* **Docker Containerization:** Packaged the entire application, including the ChromaDB vector database and lightweight PyTorch dependencies, into a standalone Docker image for seamless deployment across any OS.

---


## 📅 Week 4: Generative AI & Tooling
**Focus:** Building interactive Generative AI applications and streamlining developer workflows.

### 🔹 Self-Learning Code Generator
A specialized application designed to assist in generating and optimizing code snippets using LLMs.
* **Live Demo:** [Self-Learning Code Generator](https://self-learning-code-generator.streamlit.app/)
* **Interface:** Built with **Streamlit** for a seamless, interactive user experience.

---

## 📅 Week 3: Advanced LLMs & RAG
**Focus:** Exploring Retrieval-Augmented Generation (RAG) and scaling Large Language Model capabilities.

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


---

## 📅 Week 1: Transformers & Architectures
**Focus:** Deep dive into the BERT family, Parameter-Efficient Fine-Tuning (PEFT), and Recurrent Neural Networks.

### **BERT Family Research**
* **Architecture Analysis:** Analyzed architectural differences between **DistilBERT**, **ALBERT**, and standard **BERT**.
* **Efficiency:** Explored **LoRA** (Low-Rank Adaptation) and **QLoRA** for memory-efficient fine-tuning.

### **LSTM Implementation**
* **Sequence Classification:** Built and trained a custom **LSTM** model from scratch to handle sequential data tasks.

