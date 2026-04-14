"""
Feedback Intelligence - Standalone FAISS & Ollama Chatbot Demo
This script isolates the Retrieval-Augmented Generation (RAG) system.
It can be run independently in Python, Google Colab, or Jupyter.

What it does:
1. Loads a dataset of feedback.
2. Converts feedback text into embeddings using TF-IDF.
3. Stores embeddings in a FAISS vector database.
4. Detects similar/duplicate feedback using vector distances.
5. Queries the local Ollama LLM (`llama3` model by default) to summarize or answer questions based on the retrieved context.

WARNING: You must have Ollama installed and the model downloaded.
Command: `ollama run llama3`
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss
except ImportError:
    print("FAISS not found. Install it carefully via: pip install faiss-cpu")

# -------------------------------------------------------------
# 0. CONFIGURATION
# -------------------------------------------------------------
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

# -------------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------------
DATA_PATH = "../data/feedback.csv"

if os.path.exists(DATA_PATH):
    print(f"=> Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
else:
    print("=> No local CSV found. Using hardcoded sample dataset.")
    data = {"text": [
        "The app is very slow and crashes randomly.",
        "Performance is terrible, crashes all the time.",
        "Pricing is way too expensive for what you get.",
        "I need a cheaper plan or I will unsubscribe.",
        "Great dark mode design, really easy on the eyes."
    ]}
    df = pd.DataFrame(data)

# -------------------------------------------------------------
# 2. VECTORIZATION (TF-IDF)
# -------------------------------------------------------------
print("=> Generating Vector Embeddings using TF-IDF...")
# We use dense TF-IDF arrays because FAISS requires dense numpy arrays.
vectorizer = TfidfVectorizer(max_features=100)
X_sparse = vectorizer.fit_transform(df['text'])
vectors = X_sparse.toarray().astype('float32')
dimension = vectors.shape[1]

# -------------------------------------------------------------
# 3. FAISS VECTOR DATABASE SETUP
# -------------------------------------------------------------
print("=> Initializing FAISS Vector Store...")
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
print(f"   Stored {index.ntotal} feedback vectors in FAISS memory.")

# -------------------------------------------------------------
# 4. DUPLICATE & SIMILARITY DETECTION
# -------------------------------------------------------------
print("\n=> Detecting Similar/Duplicate Feedbacks:")
# Let's search for the nearest neighbor for the first feedback entry
query_vec = vectors[0:1]
# k=2 because the first match is always the item itself (distance=0)
distances, indices = index.search(query_vec, k=2)

original_text = df.iloc[0]['text']
similar_text = df.iloc[indices[0][1]]['text']
similarity_score = 1 / (1 + distances[0][1])  # Inverse distance

print(f"   Query: '{original_text}'")
print(f"   Match: '{similar_text}'")
print(f"   Similarity Score: {similarity_score:.2f}")
if similarity_score > 0.5:
    print("   [!] Potential Duplicate Detected!")

# -------------------------------------------------------------
# 5. OLLAMA CHATBOT INTEGRATION
# -------------------------------------------------------------
def ask_ollama(prompt):
    """Send a prompt to the local Ollama LLM endpoint."""
    print(f"\n[Sending to Ollama '{OLLAMA_MODEL}'] -> Generating response... please wait.")
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error from Ollama API: {response.text}"
    except requests.exceptions.ConnectionError:
        return "FATAL ERROR: Could not connect to Ollama. Make sure Ollama is running on localhost:11434!"

print("\n==================================================")
print("             RAG SUMMARIZATION DEMO             ")
print("==================================================")

# Let's retrieve some technical feedbacks to summarize
search_query = "crash slow performance"
search_vec = vectorizer.transform([search_query]).toarray().astype('float32')
D, I = index.search(search_vec, k=3)

# Build context from FAISS results
retrieved_feedbacks = [df.iloc[idx]['text'] for idx in I[0] if idx < len(df) and idx != -1]

prompt_template = f"""
You are a brilliant business data analyst.
I have retrieved these user feedbacks from our database: {retrieved_feedbacks}

Summarize these feedbacks into a SINGLE powerful sentence explaining what users are complaining about.
"""

response = ask_ollama(prompt_template)

print("\n=> Ollama Summary Output:")
print(response)
print("\n=> Standalone Demo Complete! 🔥")
