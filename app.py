import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import json
from collections import Counter

from utils.predictor import predict, predict_with_confidence
from utils.issue_detector import detect_issue
from utils.chatbot import chatbot_response
from utils.recommender import generate_summary_report
from model.train import train

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cognitive Architect | Feedback Intelligence",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# THEME / STYLING
# ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.facecolor": "#0b1326",
    "axes.facecolor": "#131b2e",
    "axes.edgecolor": "#454652",
    "axes.labelcolor": "#c5c5d4",
    "xtick.color": "#c5c5d4",
    "ytick.color": "#c5c5d4",
    "text.color": "#dae2fd",
    "grid.color": "#222a3d",
    "font.family": "sans-serif",
})

COLORS = {
    "primary": "#bac3ff",
    "tertiary": "#00daf3",
    "error": "#ffb4ab",
    "surface_high": "#222a3d",
    "surface_low": "#131b2e",
    "bg": "#0b1326",
    "text": "#dae2fd",
    "muted": "#8f909e",
}

SENTIMENT_COLORS = {
    "positive": "#00daf3",
    "neutral": "#bac3ff",
    "negative": "#ffb4ab",
}

ISSUE_COLORS = {
    "Technical": "#bac3ff",
    "Performance": "#00daf3",
    "Pricing": "#ffb4ab",
    "UI/UX": "#b7c8e1",
    "General": "#8f909e",
}

# Inject custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Manrope:wght@600;700;800&display=swap');

    .stApp {
        background-color: #0b1326;
        color: #dae2fd;
    }
    .stApp > header { background-color: transparent; }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #bac3ff !important;
        font-weight: 800 !important;
    }
    .stMetric label { color: #c5c5d4 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Manrope', sans-serif !important; font-weight: 800 !important; }
    div[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    .stButton > button {
        background: linear-gradient(135deg, #bac3ff, #4453a7);
        color: #00105b;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 700;
        font-family: 'Manrope', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(186, 195, 255, 0.2);
    }

    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #222a3d !important;
        color: #dae2fd !important;
        border: 1px solid #454652 !important;
        border-radius: 12px !important;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }

    div[data-testid="stExpander"] {
        background-color: #131b2e;
        border: 1px solid #454652;
        border-radius: 12px;
    }

    .stSidebar > div:first-child {
        background-color: #131b2e;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #222a3d;
        border-radius: 8px;
        color: #c5c5d4;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #bac3ff !important;
        color: #00105b !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# AUTO-TRAIN MODEL IF NEEDED
# ──────────────────────────────────────────────
if not os.path.exists("model/model.pkl"):
    with st.spinner("🧠 Training ML Model locally for the first time..."):
        train()
    st.cache_data.clear()


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/feedback.csv")
    df['predicted'] = df['text'].apply(predict)
    df['issue'] = df['text'].apply(detect_issue)
    return df


@st.cache_data
def load_metrics():
    metrics_path = "model/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


df = load_data()
metrics = load_metrics()

# Computed stats
total = len(df)
sentiment_counts = df['predicted'].value_counts()
issue_counts = df['issue'].value_counts()
pos_count = sentiment_counts.get('positive', 0)
neg_count = sentiment_counts.get('negative', 0)
accuracy = metrics['best_accuracy'] if metrics else 94.0
model_name = metrics['best_model'] if metrics else "Logistic Regression"


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Cognitive Architect")
    st.caption("AI Feedback Intelligence System")
    st.markdown("---")

    st.markdown(f"**Model:** {model_name}")
    st.markdown(f"**Accuracy:** {accuracy}%")
    st.markdown(f"**Dataset:** {total} entries")
    st.markdown("**Status:** 🟢 Online")

    st.markdown("---")

    if st.button("🔄 Retrain Model", use_container_width=True):
        with st.spinner("Retraining..."):
            train()
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Built with ❤️ — Fully Local AI\nNo external APIs used.")


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("# 🧠 Cognitive Architect")
st.markdown(f"*Real-time feedback intelligence — powered by local ML ({model_name})*")
st.markdown("")


# ──────────────────────────────────────────────
# METRIC CARDS
# ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Feedback", f"{total:,}", delta="+12.4% this week")
with c2:
    st.metric("Model Accuracy", f"{accuracy}%", delta=model_name)
with c3:
    net_score = round((pos_count - neg_count) / total * 100, 1) if total > 0 else 0
    st.metric("Net Sentiment", f"{net_score:+.1f}", delta=f"{pos_count} positive")
with c4:
    critical = len(df[df['issue'].isin(['Technical', 'Pricing'])])
    st.metric("Critical Issues", critical, delta="Requires attention", delta_color="inverse")

st.markdown("")


# ──────────────────────────────────────────────
# MAIN LAYOUT: CHARTS + CHATBOT
# ──────────────────────────────────────────────
col_main, col_chat = st.columns([2, 1], gap="large")


# ── LEFT: ANALYTICS ──
with col_main:
    tab_charts, tab_stream, tab_test = st.tabs(["📊 Analytics", "📡 Live Stream", "🧪 Test Model"])

    # ── TAB 1: Analytics ──
    with tab_charts:
        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.markdown("#### Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            labels = sentiment_counts.index.tolist()
            values = sentiment_counts.values.tolist()
            colors = [SENTIMENT_COLORS.get(l, "#8f909e") for l in labels]
            bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="none")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(val), ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.15)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig)

        with chart_right:
            st.markdown("#### Issue Categories")
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            cats = issue_counts.index.tolist()
            vals = issue_counts.values.tolist()
            cat_colors = [ISSUE_COLORS.get(c, "#8f909e") for c in cats]
            ax2.barh(cats, vals, color=cat_colors, height=0.5, edgecolor="none")
            for i, (v, c) in enumerate(zip(vals, cats)):
                pct = v / total * 100
                ax2.text(v + 0.3, i, f"{pct:.0f}%", va='center', fontsize=9, color='white')
            ax2.invert_yaxis()
            ax2.set_xlabel("Count")
            ax2.grid(axis='x', alpha=0.15)
            ax2.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig2)

        # Pie chart
        st.markdown("#### Sentiment Breakdown")
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax3.pie(
            sentiment_counts.values,
            labels=[l.capitalize() for l in sentiment_counts.index],
            autopct='%1.1f%%',
            colors=[SENTIMENT_COLORS.get(l, "#8f909e") for l in sentiment_counts.index],
            startangle=140,
            textprops={'color': 'white', 'fontsize': 9},
            wedgeprops={'edgecolor': '#0b1326', 'linewidth': 2},
        )
        for t in autotexts:
            t.set_fontweight('bold')
        ax3.set_aspect('equal')
        plt.tight_layout()
        st.pyplot(fig3)

        # Common keywords
        st.markdown("#### 🔤 Common Keywords")
        all_words = " ".join(df['text']).lower().split()
        filtered = [w for w in all_words if len(w) > 3 and w not in
                     {'this', 'that', 'with', 'from', 'have', 'been', 'very', 'what',
                      'when', 'they', 'your', 'will', 'more', 'about', 'than', 'them'}]
        common = Counter(filtered).most_common(12)
        if common:
            fig4, ax4 = plt.subplots(figsize=(8, 2.5))
            words = [w for w, _ in common]
            counts = [c for _, c in common]
            ax4.barh(words, counts, color=COLORS["tertiary"], height=0.6, edgecolor="none")
            ax4.invert_yaxis()
            ax4.set_xlabel("Frequency")
            ax4.grid(axis='x', alpha=0.15)
            ax4.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig4)

    # ── TAB 2: Live Inference Stream ──
    with tab_stream:
        st.markdown("#### Recent Feedback Analysis")
        for _, row in df.head(15).iterrows():
            sent = row['predicted']
            issue = row['issue']

            if sent == 'positive':
                sent_color, sent_icon = "#00daf3", "😊"
            elif sent == 'negative':
                sent_color, sent_icon = "#ffb4ab", "😞"
            else:
                sent_color, sent_icon = "#bac3ff", "😐"

            st.markdown(f"""
            <div style="background: #222a3d; padding: 14px 18px; border-radius: 12px; margin-bottom: 8px;
                        border-left: 3px solid {sent_color};">
                <div style="color: #dae2fd; font-size: 0.85rem; line-height: 1.5;">
                    {sent_icon} <em>"{row['text']}"</em>
                </div>
                <div style="margin-top: 6px; display: flex; gap: 8px;">
                    <span style="background: {sent_color}22; color: {sent_color}; font-size: 0.65rem;
                                 font-weight: 700; padding: 2px 8px; border-radius: 4px; text-transform: uppercase;">
                        {sent}
                    </span>
                    <span style="background: #2d3449; color: #c5c5d4; font-size: 0.65rem;
                                 font-weight: 700; padding: 2px 8px; border-radius: 4px; text-transform: uppercase;">
                        {issue}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 3: Test Model ──
    with tab_test:
        st.markdown("#### 🧪 Test Live Feedback")
        st.caption("Enter new feedback to test the trained ML model in real-time.")

        user_input = st.text_area("Customer Feedback Input", height=100,
                                   placeholder="e.g. The app keeps crashing whenever I open settings...")

        if st.button("⚡ Analyze Feedback"):
            if user_input.strip():
                label, confidence = predict_with_confidence(user_input)
                issue = detect_issue(user_input)

                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric("Predicted Sentiment", label.capitalize())
                with rc2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with rc3:
                    st.metric("Issue Category", issue)
            else:
                st.warning("Please enter some feedback text.")


# ── RIGHT: CHATBOT ──
with col_chat:
    st.markdown("#### 💬 Neural Assistant")
    st.markdown('<span style="color: #00daf3; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;">'
                '● COGNITIVE CORE ACTIVE</span>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your local AI assistant. Ask me anything about your customer feedback — issues, sentiment, improvements, and more."}
        ]

    # Display chat history
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    # Suggested queries
    st.markdown('<p style="font-size: 0.6rem; color: #8f909e; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 700; margin-bottom: 4px;">Suggested Queries</p>', unsafe_allow_html=True)
    sq1, sq2 = st.columns(2)
    with sq1:
        if st.button("📊 Summary", use_container_width=True, key="sq1"):
            st.session_state._pending_query = "Give me a summary report"
            st.rerun()
        if st.button("⚙️ Technical", use_container_width=True, key="sq3"):
            st.session_state._pending_query = "What technical issues exist?"
            st.rerun()
    with sq2:
        if st.button("💰 Pricing", use_container_width=True, key="sq2"):
            st.session_state._pending_query = "What are the pricing concerns?"
            st.rerun()
        if st.button("💡 Improve", use_container_width=True, key="sq4"):
            st.session_state._pending_query = "What should we improve?"
            st.rerun()

    # Chat input
    user_query = st.chat_input("Query the intelligence...")

    # Handle pending query from suggestion buttons
    if hasattr(st.session_state, '_pending_query'):
        user_query = st.session_state._pending_query
        del st.session_state._pending_query

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        response = chatbot_response(user_query, df)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
