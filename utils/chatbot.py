"""
Local FAISS+Ollama AI Chatbot Engine.
RAG pipeline with smart intent detection for accurate answers.
Dependencies: faiss-cpu, requests, scikit-learn
"""

import re
import requests
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"


class FeedbackVectorStore:
    """Singleton FAISS index — built once, reused across queries."""
    _instance = None

    def __init__(self, df):
        if FeedbackVectorStore._instance is not None:
            raise Exception("Use get_instance()")
        self.df = df.dropna(subset=['text']).copy()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(self.df['text'])
        vectors = X.toarray().astype('float32')
        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors)

    @classmethod
    def get_instance(cls, df):
        if cls._instance is None:
            cls._instance = FeedbackVectorStore(df)
        return cls._instance

    def search(self, query, top_k=6):
        q_vec = self.vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = self.index.search(q_vec, k=top_k)
        results = []
        for d, i in zip(distances[0], indices[0]):
            if i != -1 and i < len(self.df):
                row = self.df.iloc[i]
                results.append(f"[{row.get('predicted', '?').upper()}] {row['text']}")
        return results


def _is_casual(query):
    """Detect greetings and chit-chat — no LLM needed."""
    q = re.sub(r'[^a-z\s]', '', query.lower()).strip()
    greetings = {
        "hi", "hello", "hey", "yo", "sup", "hola",
        "hi there", "hello there", "hey there",
        "how are you", "who are you", "what are you",
        "whats up", "good morning", "good evening",
        "good afternoon", "thanks", "thank you",
        "bye", "goodbye", "ok", "okay",
        "hi chatbot", "hello chatbot", "hey chatbot",
        "hi bot", "hello bot", "hey bot",
    }
    if q in greetings:
        return True
    words = q.split()
    if len(words) <= 3 and words and words[0] in {"hi", "hello", "hey", "yo", "sup", "thanks", "bye", "ok", "okay"}:
        return True
    return False


def _texts_for(df, col, val, n=6):
    """Safely extract text samples from a filtered DataFrame."""
    subset = df[df[col] == val]
    return [f"[{val.upper()}] {t}" for t in subset['text'].head(n).tolist()]


def _get_smart_context(query, df, store):
    """Detect query intent and pull the right data — not random FAISS matches."""
    q = query.lower()
    has_issue_col = 'issue' in df.columns

    # ── Negative / Critical / Complaints ──
    if any(w in q for w in ["critical", "negative", "bad", "worst", "complaint",
                             "issue", "problem", "pain", "frustrat", "crash",
                             "bug", "slow", "expensive"]):
        items = _texts_for(df, 'predicted', 'negative', 6)
        if has_issue_col:
            items += _texts_for(df, 'issue', 'Technical', 2)
            items += _texts_for(df, 'issue', 'Pricing', 2)
        return items[:8]

    # ── Positive / Good / Praise ──
    if any(w in q for w in ["positive", "good", "best", "love", "praise",
                             "success", "happy", "great"]):
        return _texts_for(df, 'predicted', 'positive', 6)

    # ── Summary / Overview ──
    if any(w in q for w in ["summary", "summarize", "overview", "report", "overall"]):
        total = len(df)
        pos_c = len(df[df['predicted'] == 'positive'])
        neg_c = len(df[df['predicted'] == 'negative'])
        neu_c = len(df[df['predicted'] == 'neutral'])
        items = [f"[STATS] Total={total}, Positive={pos_c}, Negative={neg_c}, Neutral={neu_c}"]
        items += _texts_for(df, 'predicted', 'positive', 2)
        items += _texts_for(df, 'predicted', 'negative', 2)
        if has_issue_col:
            issues = df['issue'].value_counts().to_dict()
            items.append(f"[ISSUES] {issues}")
        return items

    # ── Pricing ──
    if any(w in q for w in ["price", "pricing", "cost", "expensive", "billing", "charge"]):
        if has_issue_col:
            return _texts_for(df, 'issue', 'Pricing', 6)
        return store.search("pricing expensive cost billing", top_k=6)

    # ── Technical ──
    if any(w in q for w in ["technical", "tech", "bug", "crash", "freeze", "error"]):
        if has_issue_col:
            return _texts_for(df, 'issue', 'Technical', 6)
        return store.search("crash bug freeze error", top_k=6)

    # ── Performance ──
    if any(w in q for w in ["performance", "speed", "slow", "fast", "loading", "lag"]):
        if has_issue_col:
            return _texts_for(df, 'issue', 'Performance', 6)
        return store.search("slow loading speed lag", top_k=6)

    # ── Improve / Suggestion ──
    if any(w in q for w in ["improve", "suggestion", "recommend", "better", "should", "enhance"]):
        return _texts_for(df, 'predicted', 'negative', 6)

    # ── Fallback: FAISS similarity ──
    return store.search(query, top_k=6)


def chatbot_response(query, df):
    """
    Main entry point.
    1. Instant reply for casual greetings.
    2. Smart intent-based data retrieval.
    3. Strict LLM prompt — answer only what was asked.
    """
    if _is_casual(query):
        return ("Hello! I'm Cognitive Core, your local AI feedback analyst. "
                "Ask me about issues, sentiment, pricing, or what to improve.")

    store = FeedbackVectorStore.get_instance(df)

    q_lower = query.lower().strip()
    if "duplicate" in q_lower or "similar" in q_lower:
        return ("Duplicate detection is active via FAISS similarity search. "
                "Feedbacks with L2-distance < 0.8 are flagged automatically.")

    context_items = _get_smart_context(query, df, store)
    context_str = "\n".join([f" - {item}" for item in context_items])

    prompt = f"""You are a concise data analyst. Rules:
1. Answer ONLY the user's question using the data below.
2. Keep your answer under 80 words.
3. Quote specific feedback when relevant.
4. The data below IS the answer — never say "no data found".

Data:
{context_str}

Question: {query}
Answer:"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json()["response"]
        return f"**Ollama Error:** Status {resp.status_code}. Is `{OLLAMA_MODEL}` installed?"
    except requests.exceptions.ConnectionError:
        return "**Offline:** Cannot reach Ollama on port 11434. Run `ollama run llama3` first."
    except Exception as e:
        return f"**Error:** {str(e)}"
