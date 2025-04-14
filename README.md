# 🧠 AI & ML Projects – Venkata Karthik Patralapati

Welcome to my portfolio of applied AI/ML projects across deep learning, natural language processing, and generative AI. This includes work on GANs (SRGAN, LSGAN), LSTM-based text classification, and Retrieval-Augmented Generation (RAG) pipelines using state-of-the-art LLMs like **Mistral** and **LLaMA3**.

---

## 📌 Table of Contents

- [1. SRGAN & LSGAN - Super-Resolution GANs](#1-srgan--lsgan---super-resolution-gans)
- [2. Toxic Comment Classification with LSTM](#2-toxic-comment-classification-with-lstm)
- [3. Retrieval-Augmented Generation (RAG) with Mistral & LLaMA3](#3-retrieval-augmented-generation-rag-with-mistral--llama3)
- [4. Challenges & Learnings](#4-challenges--learnings)
- [5. Contact](#5-contact)

---

## 1. SRGAN & LSGAN – Super-Resolution GANs

**Tech Stack**: `PyTorch`, `ImageNet`, `ResNet`, `BCE`, `MSE`, `Adam`, `Matplotlib`

- Implemented a Super-Resolution GAN (SRGAN) from scratch in PyTorch to upscale low-resolution images.
- **Generator**: 16 residual blocks, trained with content loss (MSE) and adversarial loss (BCE).
- **Discriminator**: Convolutional network using LeakyReLU to classify real vs. generated HR images.
- Visual comparisons of input vs. output done via `matplotlib`.
- Experimented with different epochs and noted resolution clarity improvements up to 1000 epochs.

---

## 2. Toxic Comment Classification with LSTM

**Tech Stack**: `PyTorch`, `LSTM`, `Jigsaw Dataset`, `NLTK`, `BCEWithLogitsLoss`, `Adam`

- Built an LSTM-based multi-label classifier for classifying toxicity in online comments.
- Preprocessed data using tokenization, stopword removal, and padding.
- Model: `Embedding ➝ LSTM ➝ Fully Connected ➝ Sigmoid`
- Achieved **91.7% accuracy** across six toxicity labels.
- Generated Kaggle-style submission files and evaluated using predicted probabilities.

---

## 3. Retrieval-Augmented Generation (RAG) with Mistral & LLaMA3

**Tech Stack**: `HuggingFace Transformers`, `LangChain`, `QLoRA`, `ChromaDB`, `SentenceTransformers`

- Developed custom RAG pipelines using **Mistral-7B** and **LLaMA3**, both base and **QLoRA-finetuned**.
- Used `LangChain` and `ChromaDB` for embedding storage and retrieval, simulating real-world Q&A.
- Applied QLoRA for efficient fine-tuning on resource-constrained setups.
- Evaluated model responses using **BLEU**, **ROUGE**, and latency metrics.
- Built agents to query structured knowledge and generate context-rich answers.

---

## 4. Challenges & Learnings

- **SRGAN**: GPU memory constraints limited high-epoch training; future improvement via distributed training.
- **Jigsaw NLP**: Cleaned mislabeled data (labels with `-1`) to improve performance.
- **LLMs**: QLoRA dramatically reduced training cost, but needed prompt tuning for long-context retrieval.

---

## 5. Contact

📍 San Jose, CA  
📧 venkatakarthik.patralapati@sjsu.edu  
🔗 [LinkedIn](https://www.linkedin.com/in/pvk04/)  
💻 [GitHub](https://github.com/KarthikPatralapati01)

---

