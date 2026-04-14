"""
Cognitive Architect — FastAPI Backend Server
Serves the Stitch-designed frontend pages and provides API endpoints
for the ML pipeline, chatbot, and feedback analysis.
"""

import os
import json
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.predictor import predict, predict_with_confidence
from utils.issue_detector import detect_issue
from utils.chatbot import chatbot_response
from utils.recommender import suggest, generate_summary_report
from model.train import train as train_model

# ── App Setup ──
app = FastAPI(title="Cognitive Architect — Feedback Intelligence")

# ── Data Loading ──
def load_data():
    df = pd.read_csv("data/feedback.csv")
    df['predicted'] = df['text'].apply(predict)
    results = df['text'].apply(lambda t: predict_with_confidence(t))
    df['confidence'] = results.apply(lambda r: r[1])
    df['issue'] = df['text'].apply(detect_issue)
    return df

def load_metrics():
    path = "model/metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# Auto-train if model does not exist
if not os.path.exists("model/model.pkl"):
    print("[STARTUP] Training model for the first time...")
    train_model()

# Load on startup
df = load_data()
metrics = load_metrics()

# ── Pydantic Models ──
class PredictRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    query: str

# ── Page Routes ──
@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    with open("frontend/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page():
    with open("frontend/dataset.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/training", response_class=HTMLResponse)
async def training_page():
    with open("frontend/training.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/reports", response_class=HTMLResponse)
async def reports_page():
    with open("frontend/reports.html", "r", encoding="utf-8") as f:
        return f.read()

# Serve shared JS
@app.get("/frontend/{filepath:path}")
async def serve_frontend(filepath: str):
    return FileResponse(f"frontend/{filepath}")

# ── API Routes ──
@app.get("/api/stats")
async def get_stats():
    global df, metrics
    total = len(df)
    sentiment_counts = df['predicted'].value_counts().to_dict()
    pos = sentiment_counts.get('positive', 0)
    neg = sentiment_counts.get('negative', 0)
    net_score = round((pos - neg) / total * 100, 1) if total > 0 else 0
    issue_counts = df['issue'].value_counts().to_dict()
    critical = issue_counts.get('Technical', 0) + issue_counts.get('Pricing', 0)

    return {
        "total_feedback": total,
        "model_accuracy": metrics['best_accuracy'] if metrics else 0,
        "model_name": metrics['best_model'] if metrics else "Unknown",
        "net_sentiment": net_score,
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": sentiment_counts.get('neutral', 0),
        "critical_issues": critical,
        "sentiment_counts": sentiment_counts,
        "issue_counts": issue_counts,
    }

@app.get("/api/feedback")
async def get_feedback():
    global df
    items = []
    for _, row in df.iterrows():
        items.append({
            "text": row['text'],
            "label": row.get('label', ''),
            "predicted": row['predicted'],
            "confidence": round(row['confidence'], 3),
            "issue": row['issue'],
        })
    return items

@app.get("/api/categories")
async def get_categories():
    global df
    total = len(df)
    issue_counts = df['issue'].value_counts().to_dict()
    result = {}
    for cat, count in issue_counts.items():
        result[cat] = {
            "count": int(count),
            "percentage": round(count / total * 100, 1),
        }
    return result

@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    label, confidence = predict_with_confidence(req.text)
    issue = detect_issue(req.text)
    return {
        "sentiment": label,
        "confidence": round(confidence, 3),
        "issue": issue,
    }

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    global df
    response = chatbot_response(req.query, df)
    return {"response": response}

@app.post("/api/train")
async def api_train():
    global df, metrics
    result = train_model()
    # Reload data
    df = load_data()
    metrics = load_metrics()
    return {
        "status": "success",
        "metrics": result,
    }

@app.get("/api/keywords")
async def get_keywords():
    global df
    from collections import Counter
    all_words = " ".join(df['text']).lower().split()
    stop = {'this', 'that', 'with', 'from', 'have', 'been', 'very', 'what',
            'when', 'they', 'your', 'will', 'more', 'about', 'than', 'them',
            'the', 'and', 'for', 'are', 'not', 'but', 'was', 'its', 'all'}
    filtered = [w for w in all_words if len(w) > 3 and w not in stop]
    common = Counter(filtered).most_common(12)
    return [{"word": w, "count": c} for w, c in common]

@app.get("/api/report")
async def get_report():
    global df
    report = generate_summary_report(df)
    # Get top issues with suggestions
    issues = df['issue'].unique().tolist()
    suggestions = suggest(issues)
    # Curated highlights
    pos_samples = df[df['predicted'] == 'positive']['text'].head(3).tolist()
    neg_samples = df[df['predicted'] == 'negative']['text'].head(3).tolist()
    return {
        "report_text": report,
        "suggestions": suggestions,
        "positive_highlights": pos_samples,
        "negative_highlights": neg_samples,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
