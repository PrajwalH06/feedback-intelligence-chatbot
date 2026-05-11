"""
Local AI Chatbot Engine — Full Data Access.
Smart query handling: text search > analytics > FAISS similarity > LLM fallback.
Dependencies: faiss-cpu, requests, scikit-learn
"""

import re
import requests
import faiss
from collections import Counter
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
                results.append({
                    "text": row['text'],
                    "predicted": row.get('predicted', '?'),
                    "issue": row.get('issue', ''),
                    "confidence": row.get('confidence', 0),
                    "distance": float(d),
                })
        return results


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _is_casual(query):
    q = re.sub(r'[^a-z\s]', '', query.lower()).strip()
    greetings = {
        "hi", "hello", "hey", "yo", "sup", "hola",
        "hi there", "hello there", "hey there",
        "how are you", "who are you", "what are you",
        "whats up", "good morning", "good evening",
        "good afternoon", "thanks", "thank you",
        "bye", "goodbye", "ok", "okay",
        "hi chatbot", "hello chatbot", "hey chatbot",
    }
    if q in greetings:
        return True
    words = q.split()
    if len(words) <= 3 and words and words[0] in {"hi", "hello", "hey", "yo", "sup", "thanks", "bye", "ok", "okay"}:
        return True
    return False


def _pct(count, total):
    return f"{count / total * 100:.1f}%" if total > 0 else "0%"


def _build_stats(df):
    total = len(df)
    has_predicted = 'predicted' in df.columns
    has_issue = 'issue' in df.columns
    has_confidence = 'confidence' in df.columns

    s = {"total": total}
    if has_predicted:
        vc = df['predicted'].value_counts().to_dict()
        s["positive"] = vc.get('positive', 0)
        s["negative"] = vc.get('negative', 0)
        s["neutral"] = vc.get('neutral', 0)
        s["net"] = round((s["positive"] - s["negative"]) / total * 100, 1) if total else 0
    if has_issue:
        s["issues"] = df['issue'].value_counts().to_dict()
        s["critical"] = s["issues"].get('Technical', 0) + s["issues"].get('Pricing', 0)
    if has_confidence:
        s["avg_conf"] = round(df['confidence'].mean(), 3)
    return s


def _text_search(df, keywords):
    """Search feedback text for rows containing ANY of the keywords."""
    mask = df['text'].str.lower().str.contains('|'.join(keywords), na=False)
    return df[mask]


def _extract_topic_words(query):
    """Extract meaningful topic words from a user query (strip filler words)."""
    q = re.sub(r'[^a-z\s]', '', query.lower())
    stop = {
        'what', 'which', 'how', 'many', 'much', 'any', 'are', 'is', 'was',
        'were', 'the', 'for', 'and', 'but', 'with', 'about', 'from', 'that',
        'this', 'there', 'their', 'they', 'have', 'has', 'had', 'can', 'could',
        'would', 'should', 'will', 'did', 'does', 'been', 'being', 'some',
        'all', 'our', 'your', 'my', 'me', 'you', 'we', 'us', 'its',
        'feedback', 'feedbacks', 'customer', 'customers', 'user', 'users',
        'review', 'reviews', 'show', 'tell', 'give', 'get', 'find', 'list',
        'related', 'regarding', 'about', 'on', 'of', 'in', 'to', 'by',
        'do', 'say', 'said', 'saying', 'think', 'opinion', 'opinions',
        'people', 'anyone', 'everyone', 'complaint', 'complaints',
        'comment', 'comments', 'mention', 'mentioned', 'mentioning',
    }
    words = [w for w in q.split() if w not in stop and len(w) > 2]
    return words


def _format_results(matches_df, topic="your query"):
    """Format matched feedbacks into a readable response."""
    total = len(matches_df)
    if total == 0:
        return None

    has_predicted = 'predicted' in matches_df.columns
    lines = [f"**{total} feedback(s) found** related to **{topic}**:\n"]

    if has_predicted:
        vc = matches_df['predicted'].value_counts().to_dict()
        parts = []
        for sent in ['positive', 'negative', 'neutral']:
            if vc.get(sent, 0) > 0:
                parts.append(f"{vc[sent]} {sent}")
        if parts:
            lines.append(f"Sentiment: {', '.join(parts)}\n")

    for _, row in matches_df.head(8).iterrows():
        sent = row.get('predicted', '?').upper()
        issue = row.get('issue', '')
        text = row['text']
        issue_tag = f" [{issue}]" if issue else ""
        lines.append(f"- **[{sent}]**{issue_tag} \"{text}\"")

    if total > 8:
        lines.append(f"\n... and {total - 8} more.")

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  Stats-based answers (for explicit analytics questions)
# ─────────────────────────────────────────────

def _try_stats_answer(query, df):
    """Handle explicit count/stats questions only."""
    q = query.lower().strip()
    stats = _build_stats(df)
    total = stats["total"]

    # ── How many / total / count ──
    is_count_q = any(w in q for w in ["how many", "total", "count", "number of"])

    if is_count_q:
        if any(w in q for w in ["positive"]):
            n = stats.get("positive", 0)
            samples = df[df['predicted'] == 'positive']['text'].head(3).tolist()
            return (f"**{n} positive feedbacks** ({_pct(n, total)}).\n\n"
                    + "\n".join(f'- "{s}"' for s in samples))
        if any(w in q for w in ["negative"]):
            n = stats.get("negative", 0)
            samples = df[df['predicted'] == 'negative']['text'].head(3).tolist()
            return (f"**{n} negative feedbacks** ({_pct(n, total)}).\n\n"
                    + "\n".join(f'- "{s}"' for s in samples))
        if any(w in q for w in ["neutral"]):
            n = stats.get("neutral", 0)
            samples = df[df['predicted'] == 'neutral']['text'].head(3).tolist()
            return (f"**{n} neutral feedbacks** ({_pct(n, total)}).\n\n"
                    + "\n".join(f'- "{s}"' for s in samples))
        if any(w in q for w in ["critical", "urgent", "priority"]):
            c = stats.get("critical", 0)
            tech = stats.get("issues", {}).get("Technical", 0)
            price = stats.get("issues", {}).get("Pricing", 0)
            return (f"**{c} critical feedbacks** (Technical: {tech}, Pricing: {price}).\n\n"
                    "These are high-priority issues requiring immediate attention.")
        if any(w in q for w in ["feedback", "record", "entri", "data", "total"]):
            return (f"**{total} total feedbacks** in the dataset.\n\n"
                    f"Breakdown: {stats.get('positive', 0)} positive, "
                    f"{stats.get('negative', 0)} negative, "
                    f"{stats.get('neutral', 0)} neutral.")

    # ── Sentiment distribution ──
    if any(w in q for w in ["sentiment", "distribution", "breakdown", "split"]):
        pos, neg, neu = stats.get("positive", 0), stats.get("negative", 0), stats.get("neutral", 0)
        return (f"**Sentiment Distribution** ({total} feedbacks):\n\n"
                f"- Positive: **{pos}** ({_pct(pos, total)})\n"
                f"- Negative: **{neg}** ({_pct(neg, total)})\n"
                f"- Neutral: **{neu}** ({_pct(neu, total)})\n\n"
                f"Net Sentiment Score: **{stats.get('net', 0)}**")

    # ── Issue categories (only if explicitly asked) ──
    if any(w in q for w in ["categor", "all issue", "issue type", "issue breakdown"]) and "issues" in stats:
        issues = stats["issues"]
        lines = [f"**Issue Categories** ({total} feedbacks):\n"]
        for cat, count in sorted(issues.items(), key=lambda x: -x[1]):
            lines.append(f"- {cat}: **{count}** ({_pct(count, total)})")
        lines.append(f"\n**Critical (Technical + Pricing): {stats.get('critical', 0)}**")
        return "\n".join(lines)

    # ── Summary / Overview ──
    if any(w in q for w in ["summary", "summarize", "overview", "report", "overall", "everything", "full report",
                             "dashboard", "stats", "statistic"]):
        pos, neg, neu = stats.get("positive", 0), stats.get("negative", 0), stats.get("neutral", 0)
        lines = [f"**Feedback Summary** ({total} entries):\n"]
        lines.append(f"**Sentiment:** {pos} positive, {neg} negative, {neu} neutral")
        lines.append(f"**Net Score:** {stats.get('net', 0)}")
        if "issues" in stats:
            lines.append(f"\n**Issues:**")
            for cat, count in sorted(stats["issues"].items(), key=lambda x: -x[1]):
                lines.append(f"- {cat}: {count}")
            lines.append(f"\n**Critical: {stats.get('critical', 0)}**")
        return "\n".join(lines)

    # ── Keywords ──
    if any(w in q for w in ["keyword", "trending", "common word", "frequent", "top word"]):
        stop = {'this', 'that', 'with', 'from', 'have', 'been', 'very', 'what',
                'when', 'they', 'your', 'will', 'more', 'about', 'than', 'them',
                'the', 'and', 'for', 'are', 'not', 'but', 'was', 'its', 'all'}
        all_words = " ".join(df['text']).lower().split()
        filtered = [w for w in all_words if len(w) > 3 and w not in stop]
        common = Counter(filtered).most_common(10)
        lines = ["**Top Keywords:**\n"]
        for word, count in common:
            lines.append(f"- **{word}**: {count} mentions")
        return "\n".join(lines)

    # ── Model / Confidence ──
    if any(w in q for w in ["confidence", "accuracy", "model"]) and not any(w in q for w in ["feedback", "review", "customer"]):
        lines = ["**Model Performance:**\n"]
        if "avg_conf" in stats:
            lines.append(f"- Average Confidence: **{stats['avg_conf']}**")
        lines.append(f"- Total predictions: {total}")
        return "\n".join(lines)

    # ── Improve / Suggest ──
    if any(w in q for w in ["improve", "suggestion", "recommend", "what to do", "action", "strategy"]):
        neg_samples = df[df['predicted'] == 'negative']['text'].head(4).tolist() if 'predicted' in df.columns else []
        lines = ["**Improvement Recommendations:**\n"]
        if "issues" in stats:
            issues = stats["issues"]
            if issues.get('Technical', 0) > 0:
                lines.append("1. **Technical:** Fix bugs and crashes, implement crash analytics")
            if issues.get('Performance', 0) > 0:
                lines.append("2. **Performance:** Optimize load times and reduce lag")
            if issues.get('Pricing', 0) > 0:
                lines.append("3. **Pricing:** Review pricing tiers, increase transparency")
            if issues.get('UI/UX', 0) > 0:
                lines.append("4. **UI/UX:** Modernize design, conduct user testing")
        if neg_samples:
            lines.append(f"\n**Top complaints:**")
            for s in neg_samples:
                lines.append(f'- "{s}"')
        return "\n".join(lines)

    # ── Compare ──
    if "compare" in q or "vs" in q or "versus" in q:
        pos, neg = stats.get("positive", 0), stats.get("negative", 0)
        diff = pos - neg
        result = f"**Comparison:** {pos} positive vs {neg} negative.\n"
        if diff > 0:
            result += f"**{diff} more positive** than negative."
        elif diff < 0:
            result += f"**{abs(diff)} more negative** than positive."
        else:
            result += "Equally split."
        return result

    # ── Help ──
    if any(w in q for w in ["help", "what can you", "what do you", "capability"]):
        return ("I can answer questions about your feedback data:\n\n"
                "- **\"How many feedbacks?\"** — Total count\n"
                "- **\"How many negative/positive?\"** — Sentiment counts\n"
                "- **\"How many critical?\"** — Critical issues\n"
                "- **\"Show sentiment distribution\"** — Full breakdown\n"
                "- **\"Show categories\"** — Issue types\n"
                "- **\"Any feedback about battery drain?\"** — Topic search\n"
                "- **\"What do customers say about pricing?\"** — Topic deep-dive\n"
                "- **\"Show keywords\"** — Trending words\n"
                "- **\"What should we improve?\"** — Recommendations\n"
                "- **\"Give me a summary\"** — Full report\n"
                "- Or ask about ANY specific topic!")

    return None


# ─────────────────────────────────────────────
#  Topic-based search (the smart part)
# ─────────────────────────────────────────────

def _try_topic_search(query, df):
    """
    Search feedback data for specific topics mentioned in the query.
    This handles questions like:
      - "any feedback about battery drain?"
      - "what do customers say about pricing?"
      - "show me feedback about crashes"
    """
    topic_words = _extract_topic_words(query)
    if not topic_words:
        return None

    topic = " ".join(topic_words)

    # Step 1: Direct text search (exact keyword match in feedback text)
    matches = _text_search(df, topic_words)

    # Step 2: If few results, also try FAISS similarity search
    if len(matches) < 3:
        store = FeedbackVectorStore.get_instance(df)
        faiss_results = store.search(topic, top_k=8)
        # Add FAISS results that aren't already in text matches
        matched_texts = set(matches['text'].tolist()) if len(matches) > 0 else set()
        for r in faiss_results:
            if r['text'] not in matched_texts and r['distance'] < 1.5:
                new_row = df[df['text'] == r['text']]
                if len(new_row) > 0:
                    matches = __import__('pandas').concat([matches, new_row])
                    matched_texts.add(r['text'])

    # Remove duplicates
    if len(matches) > 0:
        matches = matches.drop_duplicates(subset=['text'])

    result = _format_results(matches, topic)
    if result:
        return result

    # If nothing found at all, say so honestly
    if len(topic_words) > 0:
        return (f"No feedbacks found matching **\"{topic}\"**.\n\n"
                f"Try rephrasing or ask about a different topic. "
                f"Type **help** to see what I can do.")

    return None


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def chatbot_response(query, df):
    """
    Smart chatbot pipeline:
    1. Greetings → instant reply
    2. Explicit stats questions → direct analytics answer
    3. Topic queries → text search + FAISS similarity
    4. Fallback → Ollama LLM or local best-effort
    """

    # 1. Greetings
    if _is_casual(query):
        return ("Hello! I'm Cognitive Core, your local AI feedback analyst. "
                "Ask me about any topic (e.g. 'battery drain', 'pricing'), "
                "or ask for stats, sentiment, or improvements. "
                "Type **help** to see everything I can do.")

    # 2. Stats / analytics questions
    stats_answer = _try_stats_answer(query, df)
    if stats_answer:
        return stats_answer

    # 3. Topic-based search (the smart part)
    topic_answer = _try_topic_search(query, df)
    if topic_answer:
        return topic_answer

    # 4. Fallback — try LLM, or give local best-effort
    return _llm_fallback(query, df)


def _llm_fallback(query, df):
    """Try Ollama, fall back to local FAISS search."""
    store = FeedbackVectorStore.get_instance(df)
    faiss_results = store.search(query, top_k=6)
    stats = _build_stats(df)

    context_lines = [f"[{r['predicted'].upper()}] {r['text']}" for r in faiss_results]
    context_str = "\n".join(f" - {line}" for line in context_lines)
    stats_str = (f"Total: {stats['total']}, Positive: {stats.get('positive', '?')}, "
                 f"Negative: {stats.get('negative', '?')}, Neutral: {stats.get('neutral', '?')}")

    prompt = f"""You are a concise data analyst. Rules:
1. Answer ONLY the user's question using the data below.
2. Keep your answer under 100 words.
3. Quote specific feedback when relevant.

Stats: {stats_str}
Related Feedback:
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
    except:
        pass

    # Local fallback if LLM unavailable
    if faiss_results:
        lines = [f"**Related feedbacks** ({stats['total']} total in dataset):\n"]
        for r in faiss_results[:5]:
            lines.append(f"- **[{r['predicted'].upper()}]** \"{r['text']}\"")
        return "\n".join(lines)

    return (f"I have {stats['total']} feedbacks loaded. "
            "Try asking about a specific topic, sentiment counts, or type **help**.")
