# Multimodal-Amazon-Chatbot

**A Retrieval-Augmented Generation (RAG) E-commerce Assistant**

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/jyLynn/AI_shopper)
[![Powered by OpenAI](https://img.shields.io/badge/LLM-GPT--4o-green)](https://openai.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-CLIP-orange)](https://openai.com/research/clip)

## 📖 Overview

The **Multimodal AI Shopper** is an intelligent conversational agent designed to bridge the gap between static e-commerce catalogs and natural user intent. Unlike traditional keyword search, this system uses **Vector Search** and **Visual Recognition** to understand what a user wants, whether they type a vague description or upload a photo of a product.

Built on the **Amazon Product Dataset 2020**, the system processes approximately 9,500 curated product entries, employing a sophisticated RAG pipeline to retrieve relevant items and generate helpful, context-aware responses using GPT-4o.

## 🚀 Live Demo

Try the application live on Hugging Face Spaces:

<img width="1635" height="849" alt="Screenshot 2025-12-13 at 9 36 41 PM" src="https://github.com/user-attachments/assets/0dc236b7-3287-4aa2-a1c0-eed057720f5a" />

👉 **[Launch AI Shopper](https://mulitmodal-amazon-chatbot-dq5fvxecykrzwy68zltzdh.streamlit.app)**

---

## ✨ Key Features

* **Multimodal Inputs:** Accepts both **Text** ("I need headphones") and **Images** (upload a photo of a shoe) to find products.
* **Dual-Path Retrieval:** Dynamically routes queries to either a visual search path (CLIP embeddings) or a semantic text search path based on input modality.
* **Smart Query Expansion:** An intermediate LLM layer (GPT-3.5) cleans noisy queries, corrects typos, and expands brand names to improve retrieval recall.
* **Re-Ranking Engine:** Utilizes a Cross-Encoder to re-score retrieved candidates, significantly reducing semantic drift.
* **Intent-Based Responses:** The AI intelligently decides when to show images based on user intent (e.g., "Show me" vs. "How much is it?").

---

## 🛠️ System Architecture

### 1. Data Pipeline & Preprocessing
To ensure high-quality retrieval, the raw Amazon dataset underwent a rigorous normalization workflow:
* **Description Consolidation:** Attributes like *Product Description*, *About Product*, and *Technical Details* were concatenated into a unified `full_description` field to capture complete semantic context.
* **Text Segmentation:** Documents exceeding 1500 characters were segmented using a 300-character sliding window to maintain contextual coherence.
* **Data Cleaning:** Selling prices were normalized to floating-point values, and entries with invalid images or minimal content (<50 chars) were excluded.

### 2. Embedding & Vector Storage
* **Model:** `openai/clip-vit-base-patch32` (Pre-trained).
* **Text Processing:** Product descriptions were converted into numerical vectors in batches to optimize memory usage.
* **Image Processing:** Utilized parallel execution and streaming to embed ~10,000 images without memory overflow.
* **Vector DB:** **FAISS** stores separate indexes for Text and Images with a dimensionality of 512, ensuring fast and scalable retrieval.

### 3. Retrieval Logic & Re-Ranking
The system employs a `retrieve_products()` function that handles modality-specific logic:
* **Text Mode:** Embeds user queries and searches the FAISS text index.
* **Image Mode:** Embeds uploaded images and searches the FAISS image index.
* **Cross-Encoder Re-Ranker:** To improve precision, the top 20 retrieved candidates are re-scored using a cross-encoder. The top 5 highest-ranking items are then filtered and passed to the LLM, reducing irrelevant results.

### 4. LLM Generation (The "Two-Brain" Logic)
The generation phase uses **GPT-4o** to synthesize responses.
* **Query Rewriting (Brain 1):** Before search, `gpt-3.5-turbo` acts as a query optimizer. It removes conversational noise (e.g., "Can you show me..."), expands categories (e.g., "AirPods" → "Earbuds"), and fixes misspellings.
* **Response Generation (Brain 2):** The final prompt logic separates the "Search Query" from the "User Intent".
    * **Relevance Hierarchy:** Prioritizes exact matches first, then functional alternatives. Strictly forbids recommending irrelevant items.
    * **Conditional Formatting:** Only renders images if the user explicitly asks to "see" or "view" items; otherwise, provides text-only info to avoid clutter.

---

## 📊 Performance & Evaluation

We evaluated the system on a curated set of 50 representative QA pairs (15 Text, 35 Multimodal).

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Recall@1** | **94.00%** | The exact correct item was the top result. |
| **Recall@5** | **98.00%** | The correct item was in the top 5 suggestions. |
| **Alignment** | **~90%** | Accuracy of QA items alignment with target products. |

---

## 🔮 Future Development

* **Contextual Continuity:** Implementing a session state memory (e.g., LangChain) to handle multi-turn conversations and co-reference resolution (e.g., "Does *that one* come in black?").
* **Robustness Testing:** Constructing a "Hard Negative" test set with augmented images (blur, crop, rotation) to stress-test the visual encoder against low-quality user uploads.
* **Hybrid Search:** Integrating BM25 (Sparse Retrieval) with the existing Vector Search to improve exact keyword matching for specific model numbers.
