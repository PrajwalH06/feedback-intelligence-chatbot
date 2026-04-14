"""
Local FAISS+Ollama AI Chatbot Engine.
Replaces the old rule-based system with a fully local RAG (Retrieval-Augmented Generation) pipeline.
Dependencies: faiss-cpu, requests, scikit-learn
"""

import json
import requests
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

class FeedbackVectorStore:
    """Singleton memory store to avoid rebuilding the FAISS index on every query."""
    _instance = None
    
    def __init__(self, df):
        if FeedbackVectorStore._instance is not None:
            raise Exception("Singleton class, use get_instance()")
            
        self.df = df = df.dropna(subset=['text']).copy()
        
        # Build TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_sparse = self.vectorizer.fit_transform(self.df['text'])
        
        # Build FAISS Index
        vectors = X_sparse.toarray().astype('float32')
        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors)
        
    @classmethod
    def get_instance(cls, df):
        if cls._instance is None:
            cls._instance = FeedbackVectorStore(df)
        return cls._instance
    
    def search(self, query, top_k=5):
        q_vec = self.vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = self.index.search(q_vec, k=top_k)
        results = []
        for d, i in zip(distances[0], indices[0]):
            if i != -1 and i < len(self.df):
                row = self.df.iloc[i]
                results.append(f"[{row.get('predicted', 'unknown').upper()}] {row.get('text', '')}")
        return results


def chatbot_response(query, df):
    """
    RAG-based response generation.
    1. Retrieve similar feedback context from FAISS.
    2. Prompt Llama 3 via Ollama to answer the question using the context.
    """
    q_lower = query.lower().strip()
    
    # Speed Optimization 1: Short-circuit conversational greetings.
    greetings = ["hi", "hello", "hey", "how are you", "who are you", "what's up", "good morning", "good evening", "hi there", "hello there"]
    if q_lower in greetings or (len(q_lower.split()) <= 2 and any(g in q_lower for g in ["hi", "hello", "hey"])):
        return "Hello! I am Cognitive Core, your local AI assistant. I've indexed your dataset using FAISS. Ask me anything about your customer feedback!"
        
    store = FeedbackVectorStore.get_instance(df)
    
    # Simple deduplication check shortcut
    if "duplicate" in q_lower or "similar" in q_lower:
        return "Duplicate detection is active via FAISS similarity search. Feedbacks closer than a threshold of 0.8 L2-distance are flagged automatically during pipeline ingest."

    # Speed Optimization 2: Reduce context token size from k=10 to k=4.
    retrieved_feedbacks = store.search(query, top_k=4)
    context_str = "\n".join([f" - {f}" for f in retrieved_feedbacks])
    
    # Speed Optimization 3: Minimal, direct prompt mapping
    prompt = f"""You are an intelligent business analyst assistant.
Use ONLY the following context to answer the user's question concisely.

Context:
{context_str}

Question: {query}
Answer:"""
    
    # 3. Query Ollama
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=45
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"**Ollama Engine Error:** LLM returned status {response.status_code}. Are you sure `{OLLAMA_MODEL}` is installed?"
    
    except requests.exceptions.ConnectionError:
        return "**Service Unavailable:** Could not connect to the local Ollama daemon on port 11434. Please ensure Ollama is running (`ollama run llama3`) before querying."
    except Exception as e:
        return f"**System Error:** {str(e)}"
