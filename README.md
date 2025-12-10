# Multimodal-Amazon-Chatbot

**A Retrieval-Augmented Generation (RAG) E-commerce Assistant**

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/jyLynn/AI_shopper)
[![Powered by OpenAI](https://img.shields.io/badge/LLM-GPT--4o-green)](https://openai.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-CLIP-orange)](https://openai.com/research/clip)

## 📖 Overview

[cite_start]The **Multimodal AI Shopper** is an intelligent conversational agent designed to bridge the gap between static e-commerce catalogs and natural user intent[cite: 192]. Unlike traditional keyword search, this system uses **Vector Search** and **Visual Recognition** to understand what a user wants, whether they type a vague description or upload a photo of a product.

[cite_start]Built on the **Amazon Product Dataset 2020**, the system processes approximately 9,500 curated product entries, employing a sophisticated RAG pipeline to retrieve relevant items and generate helpful, context-aware responses using GPT-4o[cite: 191, 193].

## 🚀 Live Demo

Try the application live on Hugging Face Spaces:
👉 **[Launch AI Shopper](https://huggingface.co/spaces/jyLynn/AI_shopper)**

---

## ✨ Key Features

* [cite_start]**Multimodal Inputs:** Accepts both **Text** ("I need headphones") and **Images** (upload a photo of a shoe) to find products[cite: 192].
* [cite_start]**Dual-Path Retrieval:** Dynamically routes queries to either a visual search path (CLIP embeddings) or a semantic text search path based on input modality[cite: 256].
* [cite_start]**Smart Query Expansion:** An intermediate LLM layer (GPT-3.5) cleans noisy queries, corrects typos, and expands brand names to improve retrieval recall[cite: 290].
* [cite_start]**Re-Ranking Engine:** Utilizes a Cross-Encoder to re-score retrieved candidates, significantly reducing semantic drift[cite: 266].
* [cite_start]**Intent-Based Responses:** The AI intelligently decides when to show images based on user intent (e.g., "Show me" vs. "How much is it?")[cite: 310].

---

## 🛠️ System Architecture

### 1. Data Pipeline & Preprocessing
[cite_start]To ensure high-quality retrieval, the raw Amazon dataset underwent a rigorous normalization workflow[cite: 195]:
* [cite_start]**Description Consolidation:** Attributes like *Product Description*, *About Product*, and *Technical Details* were concatenated into a unified `full_description` field to capture complete semantic context[cite: 200].
* [cite_start]**Text Segmentation:** Documents exceeding 1500 characters were segmented using a 300-character sliding window to maintain contextual coherence[cite: 212].
* [cite_start]**Data Cleaning:** Selling prices were normalized to floating-point values, and entries with invalid images or minimal content (<50 chars) were excluded[cite: 197, 202, 211].

### 2. Embedding & Vector Storage
* [cite_start]**Model:** `openai/clip-vit-base-patch32` (Pre-trained)[cite: 224].
* [cite_start]**Text Processing:** Product descriptions were converted into numerical vectors in batches to optimize memory usage[cite: 226].
* [cite_start]**Image Processing:** Utilized parallel execution and streaming to embed ~10,000 images without memory overflow[cite: 239].
* [cite_start]**Vector DB:** **FAISS** stores separate indexes for Text and Images with a dimensionality of 512, ensuring fast and scalable retrieval[cite: 248, 249].

### 3. Retrieval Logic & Re-Ranking
[cite_start]The system employs a `retrieve_products()` function that handles modality-specific logic[cite: 254]:
* [cite_start]**Text Mode:** Embeds user queries and searches the FAISS text index[cite: 257].
* [cite_start]**Image Mode:** Embeds uploaded images and searches the FAISS image index[cite: 258].
* **Cross-Encoder Re-Ranker:** To improve precision, the top 20 retrieved candidates are re-scored using a cross-encoder. [cite_start]The top 5 highest-ranking items are then filtered and passed to the LLM, reducing irrelevant results[cite: 266, 269].

### 4. LLM Generation (The "Two-Brain" Logic)
[cite_start]The generation phase uses **GPT-4o** to synthesize responses[cite: 274].
* **Query Rewriting (Brain 1):** Before search, `gpt-3.5-turbo` acts as a query optimizer. [cite_start]It removes conversational noise (e.g., "Can you show me..."), expands categories (e.g., "AirPods" $\to$ "Earbuds"), and fixes misspellings[cite: 290, 291, 292].
* [cite_start]**Response Generation (Brain 2):** The final prompt logic separates the "Search Query" from the "User Intent"[cite: 299].
    * **Relevance Hierarchy:** Prioritizes exact matches first, then functional alternatives. [cite_start]Strictly forbids recommending irrelevant items[cite: 304].
    * [cite_start]**Conditional Formatting:** Only renders images if the user explicitly asks to "see" or "view" items; otherwise, provides text-only info to avoid clutter[cite: 310].

---

## 📊 Performance & Evaluation

[cite_start]We evaluated the system on a curated set of 50 representative QA pairs (15 Text, 35 Multimodal)[cite: 216].

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Recall@1** | **94.00%** | [cite_start]The exact correct item was the top result[cite: 349]. |
| **Recall@5** | **98.00%** | The correct item was in the top 5 suggestions. |
| **Alignment** | **~90%** | [cite_start]Accuracy of QA items alignment with target products[cite: 221]. |

---

## 🔮 Future Development

* [cite_start]**Contextual Continuity:** Implementing a session state memory (e.g., LangChain) to handle multi-turn conversations and co-reference resolution (e.g., "Does *that one* come in black?")[cite: 354, 355].
* [cite_start]**Robustness Testing:** Constructing a "Hard Negative" test set with augmented images (blur, crop, rotation) to stress-test the visual encoder against low-quality user uploads[cite: 361, 362].
* [cite_start]**Hybrid Search:** Integrating BM25 (Sparse Retrieval) with the existing Vector Search to improve exact keyword matching for specific model numbers[cite: 366, 368].
