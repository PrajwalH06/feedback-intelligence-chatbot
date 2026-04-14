"""
Local FAISS+Ollama AI Chatbot Engine.
Replaces the old rule-based system with a fully local RAG (Retrieval-Augmented Generation) pipeline.
Dependencies: faiss-cpu, requests, scikit-learn
"""

import re
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


def _is_casual(query):
    """Detect casual/conversational messages that don't need RAG or LLM."""
    q = query.lower().strip()
    # Remove punctuation for matching
    q_clean = re.sub(r'[^a-z\s]', '', q).strip()
    
    # Exact greeting matches
    greetings = {
        "hi", "hello", "hey", "yo", "sup", "hola",
        "hi there", "hello there", "hey there",
        "how are you", "who are you", "what are you",
        "whats up", "what s up", "good morning", "good evening",
        "good afternoon", "good night", "thanks", "thank you",
        "bye", "goodbye", "see you", "ok", "okay",
        "hi chatbot", "hello chatbot", "hey chatbot",
        "hi bot", "hello bot", "hey bot",
        "hi neural", "hello neural",
    }
    if q_clean in greetings:
        return True
    
    # Short messages (1-3 words) that start with a greeting word
    words = q_clean.split()
    if len(words) <= 3 and words[0] in {"hi", "hello", "hey", "yo", "sup", "thanks", "bye", "ok", "okay"}:
        return True
    
    return False


def chatbot_response(query, df):
    """
    RAG-based response generation.
    1. Short-circuit casual greetings instantly (no LLM call).
    2. Retrieve similar feedback context from FAISS.
    3. Prompt Llama 3 via Ollama to answer ONLY what was asked.
    """
    # ── Instant responses for casual messages ──
    if _is_casual(query):
        return "Hello! I'm Cognitive Core, your local AI feedback analyst. Ask me anything about your customer feedback data — issues, sentiment, pricing complaints, or improvement ideas."
    
    store = FeedbackVectorStore.get_instance(df)
    
    # Deduplication shortcut
    q_lower = query.lower().strip()
    if "duplicate" in q_lower or "similar" in q_lower:
        return "Duplicate detection is active via FAISS similarity search. Feedbacks closer than a threshold of 0.8 L2-distance are flagged automatically during pipeline ingest."

    # ── Retrieve context (kept small for speed) ──
    retrieved_feedbacks = store.search(query, top_k=4)
    context_str = "\n".join([f" - {f}" for f in retrieved_feedbacks])
    
    # ── Strict, minimal prompt — prevents rambling ──
    prompt = f"""You are a concise data analyst. Rules:
1. Answer ONLY the user's question. Do NOT add extra commentary.
2. Keep your answer under 60 words.
3. Use ONLY the data below. Never invent data.

Data:
{context_str}

Question: {query}
Answer (be brief):"""
    
    # ── Query Ollama ──
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
