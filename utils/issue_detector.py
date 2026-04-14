"""
Rule-based issue category detector.
Categorizes feedback into: Technical, Performance, Pricing, UI/UX, General.
"""

# Issue keywords mapping — order matters (first match wins)
ISSUE_RULES = [
    (
        "Technical",
        ["bug", "crash", "error", "broken", "fix", "issue",
         "fail", "exception", "glitch", "corrupt", "auth",
         "login", "connect", "server", "integration", "update broke",
         "doesn't work", "not working", "stopped working", "data lost"],
    ),
    (
        "Performance",
        ["slow", "lag", "speed", "performance", "loading", "freeze",
         "freezing", "latency", "memory", "battery", "drain",
         "unresponsive", "timeout", "heavy", "optimization"],
    ),
    (
        "Pricing",
        ["price", "expensive", "cost", "billing", "subscription",
         "overpriced", "fee", "charge", "payment", "refund",
         "tier", "plan", "afford", "worth", "money", "free tier"],
    ),
    (
        "UI/UX",
        ["ui", "design", "ux", "navigation", "layout", "interface",
         "theme", "color", "font", "button", "menu", "visual",
         "dark mode", "responsive", "mobile", "clunky", "outdated",
         "confusing", "intuitive"],
    ),
]


def detect_issue(text):
    """Detect the primary issue category from feedback text."""
    text_lower = text.lower()

    for category, keywords in ISSUE_RULES:
        if any(kw in text_lower for kw in keywords):
            return category

    return "General"


def detect_all_issues(text):
    """Detect ALL matching issue categories (multi-label)."""
    text_lower = text.lower()
    found = []

    for category, keywords in ISSUE_RULES:
        if any(kw in text_lower for kw in keywords):
            found.append(category)

    return found if found else ["General"]
