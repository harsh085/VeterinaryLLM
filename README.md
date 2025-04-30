# 🐾 AniSafeLM: A Safe Small Language Model for Animal Science

**VeterinaryLLM** is a domain-specific Language Model focused on **veterinary** and **animal science**, designed to improve the **safety**, **accuracy**, and **efficiency** of AI-generated responses in this critical field. Built using a robust fine-tuning pipeline on **LLaMA-3.1**, it demonstrates how smaller, specialized models can outperform large general-purpose LLMs when properly aligned.

---

## 📌 Motivation

Natural Language Processing has advanced rapidly with transformer models like ChatGPT and LLaMA. However, general-purpose LLMs often fall short in specialized domains like **animal science**, where incorrect or unsafe outputs can have real-world consequences.

This project addresses that gap by developing a **safe**, **domain-aligned**, and **efficient** small language model for veterinary use cases.

---

## 🎯 Objectives

- Build a robust pipeline for training domain-specific LLMs.
- Curate a high-quality, large-scale veterinary dataset.
- Perform continual pretraining and fine-tuning.
- Emphasize safety, preference alignment, and ethical behavior.
- Benchmark against leading open-source models.

---

## 🛠️ Development Pipeline

Data Collection → Pretraining → Fine-tuning → Preference Alignment → Safety Alignment


### ✅ Data Collection

- Collected **1.3B tokens** from 400k+ sources:
  - Veterinary research articles
  - Books and academic journals
  - Blogs and trusted websites
- Final dataset: **3.3M samples** with one column `"text"`

### ✅ Model Training

- Trained custom models with **1B** and **3B** parameters
- Focused on high domain relevance and safety

### ✅ Fine-tuning & Alignment

- Instruction tuning for better response handling
- Preference alignment using curated prompts
- Safety alignment to reduce hallucinations and risky outputs

---

## 📊 Benchmark Results (ROUGE-L)

| Model                |RougeL Score|
|----------------------|------------|
| LLaMA-3.1-70B        | 0.121      |
| Qwen-2-72B           | 0.170      |
| Phi-3.5-mini         | 0.218      |
| Gemma-2-2B           | 0.139      |
| Gemma-2-27B          | 0.158      |
| Qwen-2-1.5B          | 0.153      |
| LLaMA-3.2-3B         | 0.162      |
| **VeterinaryLLM-1B** | **0.265**  |
| **VeterinaryLLM-3B** | **0.291**  |

> 🥇 Our 3B model outperformed all other LLMs tested, including much larger models.

---

## 🧰 Tools & Technologies

- PyTorch
- Hugging Face Transformers
- Pandas
- Prompt Engineering
- Custom Web Scrapers & APIs

---

## ⚠️ Challenges

- **Data Scarcity**: Lack of clean, high-quality veterinary data
- **Compute Constraints**: Optimized for smaller, efficient models
- **Safety Risks**: Required careful alignment to avoid misinformation

---

## 🔮 Future Work

- Expand multilingual veterinary datasets
- Open-source model weights and tokenizer
- Develop VeterinaryLLM-powered APIs and applications

---

## 🙌 Acknowledgments

Inspired by the need for safer, specialized AI tools in critical domains like animal healthcare. This project showcases how **small models**, when trained well, can make a **big impact**.

---

> *“Towards Building a Safe Small Language Model for Animal Science”*
