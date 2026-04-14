"""
Improvement suggestion engine.
Generates actionable suggestions based on detected issue categories.
"""

SUGGESTION_MAP = {
    "Technical": [
        "Prioritize bug-fixing sprints and improve automated testing coverage.",
        "Implement crash analytics (e.g. Sentry) to catch issues before users report them.",
        "Create a public status page so users know when issues are being addressed.",
    ],
    "Performance": [
        "Profile and optimize critical code paths for speed.",
        "Implement lazy loading and caching strategies to reduce load times.",
        "Test performance on lower-end devices to ensure broad compatibility.",
    ],
    "Pricing": [
        "Review pricing tiers — consider adding a mid-range plan.",
        "Introduce a pay-as-you-go option for occasional users.",
        "Be more transparent about what each tier includes.",
    ],
    "UI/UX": [
        "Conduct user testing sessions to identify navigation pain points.",
        "Modernize the visual design with consistent spacing, typography, and color.",
        "Improve mobile responsiveness and touch-friendly interactions.",
    ],
    "General": [
        "Conduct deeper user research to understand unspecified concerns.",
        "Introduce feedback loops so users see their input driving changes.",
    ],
}


def suggest(issues):
    """
    Generate improvement suggestions for a list of issue categories.
    Returns a list of suggestion strings.
    """
    suggestions = []
    seen = set()

    for issue in issues:
        for s in SUGGESTION_MAP.get(issue, SUGGESTION_MAP["General"]):
            if s not in seen:
                suggestions.append(s)
                seen.add(s)

    return suggestions


def generate_summary_report(df):
    """
    Generate a high-level summary report from an analyzed DataFrame.
    Expects columns: 'predicted' (sentiment) and 'issue' (category).
    """
    total = len(df)
    if total == 0:
        return "No feedback data available to analyze."

    sentiment_counts = df['predicted'].value_counts()
    issue_counts = df['issue'].value_counts()

    lines = [
        f"📊 **Feedback Summary Report** ({total} entries analyzed)",
        "",
        "**Sentiment Breakdown:**",
    ]
    for label, count in sentiment_counts.items():
        pct = count / total * 100
        lines.append(f"  • {label.capitalize()}: {count} ({pct:.1f}%)")

    lines.append("")
    lines.append("**Issue Categories:**")
    for cat, count in issue_counts.items():
        pct = count / total * 100
        lines.append(f"  • {cat}: {count} ({pct:.1f}%)")

    # Top concern
    top_issue = issue_counts.index[0] if len(issue_counts) > 0 else "N/A"
    lines.append("")
    lines.append(f"⚠️ **Top Concern:** {top_issue} ({issue_counts.iloc[0]} mentions)")

    # Suggestions for top issue
    top_suggestions = SUGGESTION_MAP.get(top_issue, SUGGESTION_MAP["General"])
    lines.append("")
    lines.append("💡 **Recommended Actions:**")
    for s in top_suggestions:
        lines.append(f"  → {s}")

    return "\n".join(lines)
