# Comprehensive Project Report: Feedback Intelligence Ecosystem

## 1. Project Overview
**Cognitive Architect: Feedback Intelligence** is a complete, real-time analytics platform and AI assistant designed to automatically analyze, categorize, and extract actionable insights from raw customer feedback. The system features a modern, responsive web dashboard seamlessly integrated with a powerful Python backend that handles Machine Learning (ML) classification and local Large Language Model (LLM) inference.

What makes this project unique is its **100% local, privacy-first architecture**. It does not rely on paid external APIs (like OpenAI); instead, it utilizes locally-hosted Machine Learning models and a locally-running LLM (Ollama) to achieve enterprise-grade Retrieval-Augmented Generation (RAG).

---

## 2. System Architecture & Tech Stack

### Frontend (User Interface)
*   **Technologies:** Vanilla HTML5, CSS3, JavaScript, Tailwind CSS (for rapid styling).
*   **Structure:** Multi-page architecture (`dashboard.html`, `dataset.html`, `reports.html`).
*   **Role:** Handles data visualization, CSV import/export flow, dynamic DOM manipulation for category filtering, and real-time chatting with the AI Neural Assistant.

### Backend (API & Routing)
*   **Technology:** FastAPI (Python), Uvicorn.
*   **Role:** Serves the frontend pages and acts as the brain behind the REST endpoints (`/api/predict`, `/api/chat`, `/api/import`). It connects the frontend to the ML pipeline.

### AI & Machine Learning Layer (The Core)
*   **Classical ML:** `scikit-learn` (Logistic Regression, Multinomial Naive Bayes), `nltk` (Text preprocessing), `pandas` (Data manipulation).
*   **Vector Database:** `faiss-cpu` (Facebook AI Similarity Search) for semantic search.
*   **Generative AI:** `Requests` library communicating with a local **Ollama instance (Llama 3 model)**.

---

## 3. How the Machine Learning Pipeline Works
When raw text ("*The app crashes on the login screen*") enters the system, it passes through a strict pipeline before showing up on the dashboard.

### Phase 1: Preprocessing
Raw text is messy. The pipeline first lowers the text, removes special characters, and strips out **Stopwords** (common words like "and", "the", "is" that carry no analytical meaning).

### Phase 2: Feature Extraction (TF-IDF Vectorization)
*   **What is it?** Machine Learning models cannot read English text; they require numbers. **TF-IDF (Term Frequency-Inverse Document Frequency)** is the mathematical technique we used to convert our text into a matrix of numbers.
*   **How it works:** 
    *   *Term Frequency (TF):* How often a word appears in a specific feedback statement.
    *   *Inverse Document Frequency (IDF):* Penalizes words that appear too frequently across the *entire* dataset (e.g., if every feedback says "app", the word "app" becomes mathematically less important).
*   **Result:** Unique, highly-weighted keywords (like "crash" or "expensive") stand out, mapping the sentence into a 5,000-dimensional mathematical vector.

### Phase 3: Sentiment Classification (Logistic Regression)
*   **What is Logistic Regression?** Despite its name, Logistic Regression is a fundamental statistical *classification* algorithm. It takes the mathematical TF-IDF vector, multiplies it by optimized weights, and passes the result through a Sigmoid (or Softmax) function to output a probability between 0 and 1.
*   **Why we used it:** During training, we evaluated both Logistic Regression and Multinomial Naive Bayes. Logistic Regression achieved a **98.9% accuracy score** on our test dataset, proving highly effective at drawing mathematical boundaries between "Positive", "Negative", and "Neutral" feedback.

### Phase 4: Intent & Issue Detection (Rule-Based & Keyword Heuristics)
Aside from sentiment, the backend (`issue_detector.py`) scans the text against curated heuristic dictionaries to tag specific pain points (e.g., if words like "slow", "lag", or "freeze" are present, it tags the issue as **Performance**).

---

## 4. The Neural Chatbot (Local RAG Architecture)
Our chatbot is not just a standard LLM wrapper; it utilizes an advanced **Retrieval-Augmented Generation (RAG)** pipeline optimized with Smart Intent Detection.

1.  **Intent Routing:** When a user asks a question (e.g., *"What are our critical issues?"*), the backend analyzes the question's intent. Instead of blindly passing it to AI, it proactively slices the Pandas DataFrame to grab the most relevant records (e.g., isolating all Negative feedback tagged as "Technical").
2.  **Vector Similarity Search (FAISS):** If the user asks a highly specific question, the system uses **FAISS**. It converts the user's question into a TF-IDF vector and mathematically calculates the L2-Distance to find the most conceptually similar pieces of feedback in the database.
3.  **Prompt Injection:** The retrieved data is injected into a strict system prompt.
4.  **Local LLM Generation:** The prompt is sent to the local **Ollama** daemon running `Llama 3`. We strictly prompt the model to *"answer ONLY using the provided data, and keep it under 80 words."* This prevents AI "hallucinations" and ensures enterprise reliability.

---

## 5. Potential Mentor Q&A (Defense Guide)

Here is a list of extremely likely questions your mentor or panel might ask during a presentation, and exactly how you should answer them:

**Q: Why did you use Logistic Regression instead of a Deep Learning model (like an LSTM or Transformer) for Sentiment Analysis?**
*   **A:** Deep learning models require massive amounts of data and compute power to train. For short-text sentiment classification (like casual user reviews on an app), classical TF-IDF paired with Logistic Regression provides incredible accuracy (nearly 99% on our dataset) at a fraction of the computational cost and latency. It allows the pipeline to retrain locally in under 2 seconds when new CSV data is imported.

**Q: If you are using classical ML for sentiment, why do you have Llama 3 (Ollama) in the project?**
*   **A:** We divide the labor. Classical ML (Logistic Regression) is incredibly fast and highly determinist, which is perfect for processing and aggregating thousands of rows of data into Dashboard charts instantaneously. However, classical ML cannot converse with users, summarize concepts, or answer specific analytical queries. We use the heavy Generative AI (Llama 3) strictly as a conversational analytical agent through our RAG chatbot.

**Q: Explain how the FAISS vector database works in your chatbot.**
*   **A:** FAISS stands for Facebook AI Similarity Search. When our system boots, it converts every piece of text feedback into a high-dimensional vector and maps it into a 3D semantic space using FAISS format. When a user asks a niche question, we convert their query into a vector, and FAISS calculates the "Euclidean (L2) distance" to instantly find the closest, most mathematically relevant feedback records without having to linearly loop through the dataset.

**Q: What is TF-IDF and why did you use it over Word2Vec or Embeddings?**
*   **A:** TF-IDF stands for Term Frequency-Inverse Document Frequency. It scores words based on how unique they are to a specific sentence compared to the whole dataset. While Neural Embeddings (like Word2Vec/SentenceTransformers) capture deeper context, TF-IDF vectors are drastically faster to generate and perfectly viable for an MVP local sentiment classifier.

**Q: How do you prevent your LLM Chatbot from hallucinating (making things up)?**
*   **A:** We use a strict RAG implementation. The LLM has zero underlying knowledge about our company. Before we send the user request to the LLM, we retrieve the exact DataFrame rows necessary, inject them into the prompt under a "Data" header, and use a strict System Prompt: *"Rules: Answer ONLY using the data below. Never invent data."*

**Q: How does the CSV Import functionally update the system?**
*   **A:** When a new CSV is uploaded via the Dashboard, the `server.py` endpoint appends the text to `feedback.csv`. It immediately triggers `train_model()` which recalculates the TF-IDF vocabulary, retrains the Logistic Regression model on the expanded dataset, overwrites the `.pkl` artifacts, and then rebuilds the FAISS singleton memory so the chatbot knows about the new data instantly. All of this happens seamlessly in the background.
