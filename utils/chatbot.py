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
    q_clean = re.sub(r'[^a-z\s]', '', q).strip()
    
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
    
    words = q_clean.split()
    if len(words) <= 3 and words[0] in {"hi", "hello", "hey", "yo", "sup", "thanks", "bye", "ok", "okay"}:
        return True
    
    return False


def _get_smart_context(query, df, store):
    """
    Instead of blindly FAISS-searching the user's question words,
    detect the INTENT of the query and pull the right data slice.
    """
    q = query.lower()
    
    samples = []
    
    # ── Intent: Negative / Critical / Complaints / Issues / Problems ──
    if any(w in q for w in ["critical", "negative", "bad", "worst", "complaint", "issue", "problem", "pain", "frustrat", "crash", "bug", "slow", "expensive"]):
        neg = df[df['predicted'] == 'negative']
        if 'issue' in df.columns:
            # Prioritize Technical and Pricing issues (most critical)
            tech = neg[neg['issue'] == 'Technical']
            price = neg[neg['issue'] == 'Pricing']
            perf = neg[neg['issue'] == 'Performance']
            critical = tech.head(3).tolist() if not tech.empty else []
            critical += price.head(2).tolist() if not price.empty else []
            critical += perf.head(2).tolist() if not perf.empty else []
            if not critical:
                critical = neg['text'].head(6).tolist()
            else:
                critical = []
                for subset in [tech, price, perf]:
                    if not subset.empty:
                        critical.extend(subset['text'].head(2).tolist())
                if not critical:
                    critical = neg['text'].head(6).tolist()
            samples = [f"[NEGATIVE] {t}" for t in critical]
        else:
            samples = [f"[NEGATIVE] {t}" for t in neg['text'].head(6).tolist()]
    
    # ── Intent: Positive / Good / Best / Praise ──
    elif any(w in q for w in ["positive", "good", "best", "love", "praise", "success", "happy", "great"]):
        pos = df[df['predicted'] == 'positive']
        samples = [f"[POSITIVE] {t}" for t in pos['text'].head(6).tolist()]
    
    # ── Intent: Summary / Overview / Report ──
    elif any(w in q for w in ["summary", "summarize", "overview", "report", "overall", "general"]):
        total = len(df)
        pos_c = len(df[df['predicted'] == 'positive'])
        neg_c = len(df[df['predicted'] == 'negative'])
        neu_c = len(df[df['predicted'] == 'neutral'])
        samples.append(f"[STATS] Total: {total}, Positive: {pos_c}, Negative: {neg_c}, Neutral: {neu_c}")
        samples.extend([f"[POSITIVE] {t}" for t in df[df['predicted'] == 'positive']['text'].head(2).tolist()])
        samples.extend([f"[NEGATIVE] {t}" for t in df[df['predicted'] == 'negative']['text'].head(2).tolist()])
        if 'issue' in df.columns:
            issues = df['issue'].value_counts().to_dict()
            samples.append(f"[ISSUES] {issues}")
    
    # ── Intent: Pricing ──
    elif any(w in q for w in ["price", "pricing", "cost", "expensive", "billing", "charge", "pay"]):
        if 'issue' in df.columns:
            pricing = df[df['issue'] == 'Pricing']
            samples = [f"[PRICING] {t}" for t in pricing['text'].head(6).tolist()]
        else:
            samples = store.search("pricing expensive cost billing", top_k=6)
    
    # ── Intent: Technical / Bugs / Crashes ──
    elif any(w in q for w in ["technical", "tech", "bug", "crash", "freeze", "error", "glitch"]):
        if 'issue' in df.columns:
            tech = df[df['issue'] == 'Technical']
            samples = [f"[TECHNICAL] {t}" for t in tech['text'].head(6).tolist()]
        else:
            samples = store.search("crash bug freeze error technical", top_k=6)
    
    # ── Intent: Performance ──
    elif any(w in q for w in ["performance", "speed", "slow", "fast", "loading", "lag"]):
        if 'issue' in df.columns:
            perf = df[df['issue'] == 'Performance']
            samples = [f"[PERFORMANCE] {t}" for t in perf['text'].head(6).tolist()]
        else:
            samples = store.search("slow loading speed performance lag", top_k=6)
    
    # ── Intent: Improvement / Suggestion ──
    elif any(w in q for w in ["improve", "suggestion", "recommend", "better", "should", "could", "enhancement"]):
        neg = df[df['predicted'] == 'negative']
        samples = [f"[NEEDS IMPROVEMENT] {t}" for t in neg['text'].head(6).tolist()]
    
    # ── Fallback: Use FAISS similarity search ──
    else:
        samples = store.search(query, top_k=6)
    
    return samples


def chatbot_response(query, df):
    """
    Smart RAG-based response generation.
    1. Short-circuit casual greetings instantly (no LLM call).
    2. Detect query intent and pull the RIGHT data slice.
    3. Prompt Llama 3 via Ollama to answer ONLY what was asked.
    """
    # ── Instant responses for casual messages ──
    if _is_casual(query):
        return "Hello! I'm Cognitive Core, your local AI feedback analyst. Ask me anything about your customer feedback — issues, sentiment, pricing complaints, or improvement ideas."
    
    store = FeedbackVectorStore.get_instance(df)
    
    # Deduplication shortcut
    q_lower = query.lower().strip()
    if "duplicate" in q_lower or "similar" in q_lower:
        return "Duplicate detection is active via FAISS similarity search. Feedbacks closer than a threshold of 0.8 L2-distance are flagged automatically during pipeline ingest."

    # ── Smart context retrieval based on intent ──
    context_items = _get_smart_context(query, df, store)
    context_str = "\n".join([f" - {f}" for f in context_items])
    
    # ── Strict, minimal prompt ──
    prompt = f"""You are a concise data analyst. Rules:
1. Answer ONLY the user's question using the data below.
2. Keep your answer under 80 words.
3. Reference specific feedback quotes when relevant.
4. Never say "there is no data" — the data below IS the answer.

Data:
{context_str}

Question: {query}
Answer:"""
    
    # ── Query Ollama ──
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"**Ollama Engine Error:** LLM returned status {response.status_code}. Are you sure `{OLLAMA_MODEL}` is installed?"
    
    except requests.exceptions.ConnectionError:
        return "**Service Unavailable:** Could not connect to the local Ollama daemon on port 11434. Please ensure Ollama is running (`ollama run llama3`) before querying."
    except Exception as e:
        return f"**System Error:** {str(e)}"
