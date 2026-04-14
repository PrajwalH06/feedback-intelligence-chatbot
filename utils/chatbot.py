"""
Local AI Chatbot Engine.
Answers questions about feedback data using rule-based NLP + data queries.
No external API required — fully local intelligence.
"""

from collections import Counter
from utils.recommender import suggest, generate_summary_report


def chatbot_response(query, df):
    """
    Process a user query against the feedback DataFrame and return
    an intelligent, context-aware response.

    Supports queries about:
      - Issues / problems / complaints
      - Technical problems
      - Pricing concerns
      - Performance problems
      - UI/UX feedback
      - Improvements / suggestions
      - Sentiment overview
      - Summary / report
      - Common keywords / trends
      - Positive / negative feedback
    """
    q = query.lower().strip()

    # ---- Summary / Report ----
    if any(kw in q for kw in ["summary", "report", "overview", "overall", "dashboard"]):
        return generate_summary_report(df)

    # ---- Issue breakdown ----
    if any(kw in q for kw in ["issue", "problem", "complaint", "concern", "what are"]):
        issues_dict = df['issue'].value_counts().to_dict()
        total = len(df)
        lines = ["Here are the main issues reported by customers:\n"]
        for cat, count in issues_dict.items():
            pct = count / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"  **{cat}**: {count} ({pct:.1f}%)  `{bar}`")
        return "\n".join(lines)

    # ---- Technical ----
    if any(kw in q for kw in ["technical", "bug", "crash", "error", "broken"]):
        tech_df = df[df['issue'] == 'Technical']
        count = len(tech_df)
        if count == 0:
            return "No technical issues have been reported."
        samples = tech_df['text'].head(3).tolist()
        lines = [f"⚙️ **{count} technical issues** detected.\n"]
        lines.append("Recent examples:")
        for s in samples:
            lines.append(f'  • _{s}_')
        lines.append("\n💡 **Suggestion:** Prioritize bug fixes and improve test coverage.")
        return "\n".join(lines)

    # ---- Pricing ----
    if any(kw in q for kw in ["pricing", "price", "cost", "expensive", "billing", "money"]):
        price_df = df[df['issue'] == 'Pricing']
        count = len(price_df)
        if count == 0:
            return "No pricing complaints have been reported."
        samples = price_df['text'].head(3).tolist()
        lines = [f"💰 **{count} pricing complaints** detected.\n"]
        lines.append("What users are saying:")
        for s in samples:
            lines.append(f'  • _{s}_')
        lines.append("\n💡 **Suggestion:** Consider reviewing your pricing tiers and adding flexible options.")
        return "\n".join(lines)

    # ---- Performance ----
    if any(kw in q for kw in ["performance", "slow", "speed", "lag", "loading"]):
        perf_df = df[df['issue'] == 'Performance']
        count = len(perf_df)
        if count == 0:
            return "No performance issues found in the feedback."
        samples = perf_df['text'].head(3).tolist()
        lines = [f"🚀 **{count} performance issues** reported.\n"]
        lines.append("User reports:")
        for s in samples:
            lines.append(f'  • _{s}_')
        lines.append("\n💡 **Suggestion:** Optimize load times and profile critical paths.")
        return "\n".join(lines)

    # ---- UI/UX ----
    if any(kw in q for kw in ["ui", "ux", "design", "interface", "navigation", "layout"]):
        ux_df = df[df['issue'] == 'UI/UX']
        count = len(ux_df)
        if count == 0:
            return "No UI/UX issues have been reported."
        samples = ux_df['text'].head(3).tolist()
        lines = [f"🎨 **{count} UI/UX issues** detected.\n"]
        lines.append("User feedback:")
        for s in samples:
            lines.append(f'  • _{s}_')
        lines.append("\n💡 **Suggestion:** Conduct usability testing and modernize the interface.")
        return "\n".join(lines)

    # ---- Improvements / Suggestions ----
    if any(kw in q for kw in ["improve", "suggestion", "recommend", "fix", "better", "action"]):
        issues = df['issue'].unique().tolist()
        suggestions = suggest(issues)
        lines = ["Based on the current feedback, here are actionable improvements:\n"]
        for i, s in enumerate(suggestions, 1):
            lines.append(f"  {i}. {s}")
        return "\n".join(lines)

    # ---- Sentiment overview ----
    if any(kw in q for kw in ["sentiment", "mood", "feeling", "happy", "angry"]):
        sentiments = df['predicted'].value_counts().to_dict()
        total = len(df)
        lines = ["📊 **Sentiment Distribution:**\n"]
        emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        for label, count in sentiments.items():
            pct = count / total * 100
            lines.append(f"  {emoji_map.get(label, '•')} **{label.capitalize()}**: {count} ({pct:.1f}%)")

        # Net sentiment score
        pos = sentiments.get('positive', 0)
        neg = sentiments.get('negative', 0)
        net = ((pos - neg) / total) * 100 if total > 0 else 0
        lines.append(f"\n  Net Sentiment Score: **{net:+.1f}**")
        return "\n".join(lines)

    # ---- Positive feedback ----
    if any(kw in q for kw in ["positive", "good", "praise", "love", "best"]):
        pos_df = df[df['predicted'] == 'positive']
        if len(pos_df) == 0:
            return "No positive feedback found."
        samples = pos_df['text'].head(5).tolist()
        lines = [f"😊 **{len(pos_df)} positive reviews** found.\n"]
        lines.append("Highlights:")
        for s in samples:
            lines.append(f'  • _{s}_')
        return "\n".join(lines)

    # ---- Negative feedback ----
    if any(kw in q for kw in ["negative", "bad", "worst", "hate", "terrible"]):
        neg_df = df[df['predicted'] == 'negative']
        if len(neg_df) == 0:
            return "No negative feedback found."
        samples = neg_df['text'].head(5).tolist()
        lines = [f"😞 **{len(neg_df)} negative reviews** found.\n"]
        lines.append("Critical feedback:")
        for s in samples:
            lines.append(f'  • _{s}_')
        return "\n".join(lines)

    # ---- Keywords / trends ----
    if any(kw in q for kw in ["keyword", "trend", "common", "frequent", "word", "topic"]):
        all_words = " ".join(df['text']).lower().split()
        # Filter out very short / common words
        filtered = [w for w in all_words if len(w) > 3]
        common = Counter(filtered).most_common(10)
        lines = ["🔤 **Top Keywords in Feedback:**\n"]
        for word, count in common:
            lines.append(f"  • `{word}` — {count} mentions")
        return "\n".join(lines)

    # ---- Fallback ----
    return (
        "I can help you analyze your feedback data! Try asking:\n\n"
        "  • _What are the main issues?_\n"
        "  • _Show me pricing complaints_\n"
        "  • _What technical problems exist?_\n"
        "  • _How is the overall sentiment?_\n"
        "  • _What should we improve?_\n"
        "  • _Give me a summary report_\n"
        "  • _What are the common keywords?_\n"
        "  • _Show positive feedback_"
    )
