"""
Cognitive Architect — FastAPI Backend Server
Serves the Stitch-designed frontend pages and provides API endpoints
for the ML pipeline, chatbot, and feedback analysis.
"""

import os
import io
import json
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.predictor import predict, predict_with_confidence, reload_models
from utils.issue_detector import detect_issue
from utils.chatbot import chatbot_response, FeedbackVectorStore
from utils.recommender import suggest, generate_summary_report
from model.train import train as train_model

# ── App Setup ──
app = FastAPI(title="Cognitive Architect — Feedback Intelligence")

# ── Data Loading ──
TRAIN_DATA_PATH = "data/training_data.csv"
ANALYSIS_DATA_PATH = "data/analyzed.csv"

def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['text', 'label', 'predicted', 'confidence', 'issue'])
    df = pd.read_csv(path)
    if len(df) == 0:
        return df
    # Strip any duplicate header rows that may have leaked in from bad imports
    df = df[~((df['text'] == 'text') & (df['label'] == 'label'))]
    df = df.dropna(subset=['text'])
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
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


def _clean_and_save_csv(df, path):
    """Ensure the CSV on disk is clean (no dup headers, no dupes)."""
    if 'label' in df.columns:
        df = df[~((df['text'] == 'text') & (df['label'] == 'label'))]
    df = df.dropna(subset=['text'])
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    df.to_csv(path, index=False)
    return df

def _invalidate_caches():
    """Force predictor and chatbot to reload after a retrain."""
    reload_models()
    FeedbackVectorStore._instance = None


# Auto-train if model does not exist
if not os.path.exists("model/model.pkl"):
    print("[STARTUP] Training model for the first time...")
    train_model()

# Clean up any corrupted data from previous bad imports on startup
_startup_df = pd.read_csv(TRAIN_DATA_PATH)
_clean_and_save_csv(_startup_df, TRAIN_DATA_PATH)
del _startup_df

# Load on startup: strictly use the analysis dataset for the dashboard
if not os.path.exists(ANALYSIS_DATA_PATH):
    pd.DataFrame(columns=['text', 'label', 'predicted', 'confidence', 'issue']).to_csv(ANALYSIS_DATA_PATH, index=False)
df = load_data(ANALYSIS_DATA_PATH)
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
    _invalidate_caches()
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

@app.get("/api/export")
async def export_dataset():
    filepath = os.path.abspath(DATA_PATH)
    return FileResponse(
        path=filepath,
        media_type="text/csv",
        filename="feedback_export.csv",
        headers={"Content-Disposition": "attachment; filename=feedback_export.csv"}
    )

@app.post("/api/import")
async def import_dataset(file: UploadFile = File(...)):
    """
    Merge (append) new feedback into the analysis dataset.
    DOES NOT retrain the model or pollute the training data.
    """
    global df, metrics

    # ── 1. Read and validate the uploaded CSV ──
    try:
        contents = await file.read()
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    if 'text' not in new_df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must have a 'text' column. Found: {list(new_df.columns)}"
        )

    # ── 2. Clean the incoming data ──
    new_df = new_df.dropna(subset=['text'])
    new_df = new_df[new_df['text'].str.strip() != '']

    # Auto-predict labels for analysis
    new_df['label'] = new_df['text'].apply(predict)

    if len(new_df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no valid rows.")

    # ── 3. Merge with existing analysis data ──
    if os.path.exists(ANALYSIS_DATA_PATH):
        existing_df = pd.read_csv(ANALYSIS_DATA_PATH)
    else:
        existing_df = pd.DataFrame(columns=['text', 'label'])
        
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    merged_df = _clean_and_save_csv(merged_df, ANALYSIS_DATA_PATH)

    new_count = len(merged_df) - len(existing_df)

    # ── 4. Reload dashboard data (NO retraining) ──
    df = load_data(ANALYSIS_DATA_PATH)

    return {
        "status": "success",
        "message": f"Analyzed {new_count} new records. Total in dashboard: {len(df)}.",
        "total": len(df),
        "new_records": new_count,
    }


@app.post("/api/replace-dataset")
async def replace_dataset(file: UploadFile = File(...)):
    """
    Replace the dashboard analysis data with a new uploaded CSV.
    DOES NOT retrain the model or pollute the training data.
    """
    global df, metrics

    try:
        contents = await file.read()
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    if 'text' not in new_df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must have a 'text' column. Found: {list(new_df.columns)}"
        )

    new_df = new_df.dropna(subset=['text'])
    new_df = new_df[new_df['text'].str.strip() != '']
    new_df['label'] = new_df['text'].apply(predict)

    new_df = _clean_and_save_csv(new_df, ANALYSIS_DATA_PATH)

    if len(new_df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no valid rows.")

    # Reload dashboard data (NO retraining)
    df = load_data(ANALYSIS_DATA_PATH)

    return {
        "status": "success",
        "message": f"Analysis dashboard updated. Displaying {len(df)} records.",
        "total": len(df),
    }



@app.post("/api/train-import")
async def train_import_dataset(file: UploadFile = File(...)):
    """Merge (append) new feedback into the training dataset and RETRAIN."""
    global metrics
    try:
        contents = await file.read()
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    if 'text' not in new_df.columns or 'label' not in new_df.columns:
        raise HTTPException(status_code=400, detail="Training CSV must have 'text' and 'label' columns.")

    new_df = new_df.dropna(subset=['text', 'label'])
    
    if os.path.exists(TRAIN_DATA_PATH):
        existing_df = pd.read_csv(TRAIN_DATA_PATH)
    else:
        existing_df = pd.DataFrame(columns=['text', 'label'])
        
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    _clean_and_save_csv(merged_df, TRAIN_DATA_PATH)

    train_model()
    _invalidate_caches()
    metrics = load_metrics()

    return {"status": "success", "total": len(merged_df), "metrics": metrics}

@app.post("/api/train-replace-dataset")
async def train_replace_dataset(file: UploadFile = File(...)):
    """Replace the training dataset and RETRAIN."""
    global metrics
    try:
        contents = await file.read()
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    if 'text' not in new_df.columns or 'label' not in new_df.columns:
        raise HTTPException(status_code=400, detail="Training CSV must have 'text' and 'label' columns.")

    new_df = new_df.dropna(subset=['text', 'label'])
    _clean_and_save_csv(new_df, TRAIN_DATA_PATH)

    train_model()
    _invalidate_caches()
    metrics = load_metrics()

    return {"status": "success", "total": len(new_df), "metrics": metrics}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
