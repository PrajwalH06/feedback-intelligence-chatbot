# Cognitive Core: Feedback Intelligence — Technical Documentation & Project Report

## 1. Project Overview
Cognitive Core is a self-hosted, localized, end-to-end customer feedback intelligence platform. The primary goal of the system is to ingest raw customer feedback (like app reviews or survey responses), automatically analyze sentiment, categorize underlying issues, and allow interactive querying of this data through a Retrieval-Augmented Generation (RAG) AI Chatbot.

**Key Value Proposition:** Everything runs 100% locally. There are no external API calls to OpenAI or cloud databases, guaranteeing absolute data privacy and rapid local inference.

---

## 2. Technology Stack & Architecture

- **Backend:** Python + FastAPI
  - *Why FastAPI?* It provides high performance, asynchronous request handling, and auto-generated API documentation. We use it to serve the REST APIs (importing/exporting CSV, generating stats) and static frontend files.
- **Machine Learning / NLP Engine:** Scikit-Learn, Pandas, NumPy
  - *Why Scikit-Learn?* It is the industry standard for classical machine learning, providing robust vectorization and classification tools.
- **AI Chatbot (RAG System):** FAISS (Facebook AI Similarity Search) + Ollama (LLaMA 3)
  - *Why FAISS and Ollama?* FAISS handles blazing-fast semantic similarity searches across thousands of feedbacks directly in RAM. Ollama acts as the local LLM host, running a quantized version of Meta's LLaMA 3 to generate human-like answers strictly derived from local data.
- **Frontend UI:** HTML5, modern Javascript, CSS (Tailwind/Vanilla)
  - A highly responsive, single-page application dashboard designed to visualize the data metrics seamlessly.

---

## 3. The Logic & Data Flow (How It Works)

1. **Data Ingestion:** The user uploads a CSV file containing unstructured text feedback via the UI.
2. **Preprocessing:** Text data is cleaned (stopword removal, stemming, lowercasing) to ensure the AI doesn't get confused by meaningless grammatical variations.
3. **Training & Prediction:**
   - The system vectorizes the text using **TF-IDF**.
   - The vectors are passed through a trained **Logistic Regression** model.
   - The model predicts the **Sentiment** (Positive, Negative, Neutral) and assigns a confidence score.
   - A secondary mechanism categorizes the **Issue Type** (e.g., Pricing, Technical, Performance).
4. **Data Visualization:** The processed data is flushed to the frontend where charts are rendered and statistics are calculated.
5. **Interactive Querying (Chatbot):** A user asks the chatbot a question (e.g., "What are the common technical issues?"). 
   - A layer of **Smart Intent Detection** understands what the user wants.
   - It retrieves only relevant feedback rows.
   - It injects this specific data into the LLaMA 3 prompt.
   - The LLM synthesizes an intelligent, concise answer.

---

## 4. Deep Dive: Machine Learning Concepts

If your mentor asks about the "AI Engine", here is exactly how it works under the hood.

### A. TF-IDF Vectorization (How the computer reads words)
Machine learning models cannot read English text; they only understand numbers. We use a technique called **Term Frequency-Inverse Document Frequency (TF-IDF)** to convert sentences into mathematical vectors.
- **Term Frequency (TF):** Measures how frequently a word appears in a specific feedback sentence.
- **Inverse Document Frequency (IDF):** Penalizes very common words (like "the", "and") and gives high weight to rare, meaningful words (like "crash", "expensive").
- **Result:** Each feedback is transformed into an array of floats (a vector) where important semantic keywords have the highest values.

### B. Logistic Regression (How the computer decides sentiment)
Once we have our vectorized numbers, we feed them into a **Logistic Regression** classifier. 
- *What is it?* Despite the name "regression", it is a *classification* algorithm. It applies a mathematical mathematical function (the Sigmoid function) to the input values to output a probability between 0.0 and 1.0.
- *How it works here:* For a given review vector, the model calculates the probability of it belonging to each class (Positive, Negative, Neutral). The class with the highest probability "wins", and we store that probability as our `confidence` metric. 
- *Why Logistic Regression?* It is extremely fast, highly interpretable, and computationally lightweight, making it vastly superior to massive neural networks for simple text classification tasks with limited data.

---

## 5. Defense Questions (Mentor Q&A)

Here are the questions a mentor is likely to ask, and exactly how you should answer them:

**Q: On what basis exactly is the feedback analyzed?**
**A:** Feedback is analyzed on two primary axes: Sentiment and Issue Category. 
1. **Sentiment** is determined by our Logistic Regression ML model, which predicts if words used carry a mathematically positive or negative weight (learned during training on the TF-IDF feature space). 
2. **Issue Category** is analyzed using targeted keyword matching pipelines (e.g., linking words like "crash", "bug", "freeze" to "Technical Issues", and "cost", "expensive", "billing" to "Pricing").

**Q: What is a RAG pipeline and why did you use it instead of just asking ChatGPT?**
**A:** RAG stands for Retrieval-Augmented Generation. We used FAISS to build an index of our local feedback dataset. When a user asks a question, we *retrieve* identical documents from FAISS and pass them to LLaMA 3 as "context". 
We did this instead of using ChatGPT for two reasons: 
1. **Security/Privacy:** Corporate feedback data should not be sent to external cloud servers. 
2. **Accuracy:** Standard LLMs hallucinate. By forcing our local LLM to answer *only* based on the retrieved context, we guarantee 100% factual answers mapped to actual user feedback.

**Q: How does your Smart Intent Detection in the chatbot work?**
**A:** Earlier iterations blindly ran a similarity search (FAISS) based on the user's prompt. This failed on analytical queries like "Show me critical issues" because the feedback itself doesn't contain the phrase "critical issues". 
I built a heuristic intent router that detects the user's goal via regex and keywords. If the user asks about "crashes", the system forcefully slices the Pandas dataframe for "Technical" issues and injects *that* into the LLM, bypassing the naive FAISS word-search.

**Q: Do you use a real database like PostgreSQL?**
**A:** Currently, the system uses an in-memory Pandas dataframe backed by a persistent CSV flat-file (`data/feedback.csv`). For the scale of a few thousand feedback items, Pandas offers microsecond slice-and-dice times that outpace traditional SQL. If the dataset scales to millions of rows, the architecture allows for a seamless swap to PostgreSQL.

**Q: What happens if I upload completely new data? does the ML model know about it?**
**A:** Yes. The backend exposes an `/api/import` endpoint. When a new CSV is appended, the backend dynamically triggers a `train_model()` function that recalibrates the TF-IDF vectorizer and refits the Logistic Regression weights in real-time, meaning the entire system continuously learns.
