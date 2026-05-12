# 🎓 Project Presentation: Feedback Intelligence+ AI Chatbot

**Good morning/afternoon, Examiner. I will be presenting my project: Feedback Intelligence+ with AI Chatbot.**

My presentation will follow the chronological flow of data through the system, from the moment raw text is ingested to the moment the AI chatbot provides strategic business insights.

---

## 1. Project Introduction & Architecture Overview

**The Problem:** Modern businesses receive thousands of unstructured text reviews across various domains (e-commerce, SaaS, apps). Manually reading, categorizing, and finding insights in this data is impossible.

**The Solution:** Feedback Intelligence+ is a fully offline, local-first web application that automates this entire process using Machine Learning and Natural Language Processing (NLP).

Before we trace the data flow, it is crucial to understand that our system uses a **Zero-Dependency, Dual-Dataset Architecture**. We do not use a heavy SQL database. Instead, data lives in two distinct CSV files:
1. `data/training_data.csv`: A highly curated, manually labeled dataset of 400+ reviews used exclusively to teach the AI.
2. `data/analyzed.csv`: The live production dataset where newly uploaded, unlabeled user feedback is analyzed and stored.

Let us now walk through the chronological workflow of the system.

---

## Phase 1: The Machine Learning Training Flow

The entire system relies on its ability to understand language. This begins in our training pipeline. When the system initializes, it learns from `training_data.csv` in five rigorous steps:

### Step 1: NLP Preprocessing
Machine learning models only understand numbers, not words. First, we clean the raw text. We convert everything to lowercase, expand contractions (e.g., "don't" becomes "do not"), and remove punctuation. We then use **NLTK (Natural Language Toolkit)** to remove "stopwords" (filler words like "the", "is", "at"). 

*Crucially, we implemented custom logic to preserve negation words (like "not", "no") and intensifiers (like "very"), because removing them would destroy the sentiment meaning of short phrases.* Finally, we apply Porter Stemming to reduce words to their root form (e.g., "crashing" becomes "crash").

### Step 2: Feature Extraction (TF-IDF Vectorization)
The cleaned text is converted into a mathematical matrix using a **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer**. We configure it to capture Unigrams, Bigrams, and Trigrams. This means the algorithm learns to recognize not just individual words, but meaningful 3-word phrases like "not very good" or "app crashes frequently".

### Step 3: Stratified Splitting
We split our 400+ rows into two chunks: 80% for training and 20% for testing. We use **Stratified Splitting** to guarantee that the exact ratio of Positive/Negative/Neutral feedback is maintained in both chunks, preventing the model from becoming biased towards any one sentiment.

### Step 4: Algorithm Competition & Cross-Validation
Rather than guessing which algorithm works best, the script acts as an arena and trains three different algorithms simultaneously:
1. **Logistic Regression (with L2 Regularization):** A robust statistical baseline that calculates the probability of a text belonging to a class.
2. **Multinomial Naive Bayes:** A probabilistic model based on Bayes' Theorem.
3. **Linear Support Vector Machine (SVM):** A complex algorithm that plots text vectors in high-dimensional space and draws optimal mathematical hyperplanes between sentiments.

To ensure accuracy isn't a fluke, all three undergo **5-Fold Cross Validation**. The 80% training data is chopped into 5 chunks, rotating until every single piece has been used for testing exactly once. 

### Step 5: Selection & Serialization
The script dynamically selects the winner—almost always the **Linear SVM**, which achieves up to 85% accuracy on this highly dimensional data. It checks for overfitting by comparing the training score against the unseen 20% test score. Finally, the winning model is serialized into a `.pkl` file for instant memory loading.

---

## Phase 2: Live Ingestion & Inference Flow

Now that the system has a "brain", let's look at what happens when a business owner actually uses the application.

### Step 1: Unlabeled Data Import
The user navigates to the Dataset Management dashboard and uploads a raw, unlabeled CSV file full of new customer reviews. They select the "Analysis" target so it does not pollute the training data.

### Step 2: Auto-Prediction
The backend intercepts the upload. It instantly runs every single new review through the NLP preprocessor and feeds the resulting TF-IDF vectors into our trained SVM model. The model outputs a predicted label (Positive, Negative, or Neutral) and a mathematical confidence percentage.

### Step 3: Rule-Based Issue Detection
Parallel to the ML prediction, the text is scanned by a heuristic engine. It uses keyword arrays to categorize the business issue. For example, if it detects the words "server", "crash", or "bug", it tags the row as a **Technical** issue. If it detects "expensive" or "billing", it tags it as a **Pricing** issue.

### Step 4: Storage
These fully analyzed rows are appended to `analyzed.csv`, keeping the analysis data completely separate from the ground-truth training data.

---

## Phase 3: Dashboard & Analytics Flow

With the data fully analyzed, the backend calculates the high-level metrics and serves them to the frontend.

### Metric Calculations:
- **Net Sentiment Score:** Calculated by taking the ratio of purely positive feedback against the total count. If it exceeds 50%, the UI dynamically shifts to a "POSITIVE BASELINE" status.
- **Critical Issues:** The backend specifically sums the "Technical" and "Pricing" categories. These are highlighted in red on the dashboard as "Critical" because they directly correlate to immediate user churn and revenue loss.
- **Dynamic Visuals:** The frontend utilizes custom SVG-arc functions to render a real-time Donut Chart representing the sentiment distribution, completely avoiding heavy external charting libraries.

---

## Phase 4: Conversational AI Flow

The final piece of the workflow is how users explore this data. Rather than filtering through spreadsheets, users can ask questions in natural English. To ensure high speed and accuracy, the chatbot uses a **3-Tier Smart Routing Pipeline**:

### Tier 1: Direct Analytics Routing
If the user asks a mathematical question (e.g., "How many total feedbacks are there?" or "Show me the critical issues count"), the system intercepts it. It runs a direct DataFrame calculation and returns an instantaneous, 100% accurate mathematical answer without relying on an LLM.

### Tier 2: Semantic Topic Search (FAISS)
If the user asks for specific themes (e.g., "Show me complaints about battery drain"), the system utilizes **FAISS (Facebook AI Similarity Search)**. It converts the user's question into a vector and calculates the L2 Distance against every feedback vector in our database. It instantly returns the closest semantic matches.

### Tier 3: LLM Fallback (Ollama)
Only if the user asks a complex, open-ended question (e.g., "What strategic improvements should we make based on the negative reviews?") does the system fall back to a local LLM. It injects the FAISS semantic context into the prompt, forcing the LLM to generate answers based strictly on our localized data, eliminating hallucinations entirely.

---

**Conclusion:** 
By stringing together these four distinct flows—Machine Learning Training, Real-Time Inference, Metric Calculation, and Semantic Search—Feedback Intelligence+ transforms a raw, unstructured CSV file into an interactive, highly analytical business dashboard entirely offline.
