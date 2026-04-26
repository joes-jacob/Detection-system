
import io
import json
import uuid
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# --- FIREBASE IMPORTS ---
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os

# Get the port from the environment, default to 8000 if not found
port = int(os.environ.get("PORT", 8000))
# ==========================================
# 1. INITIALIZE FASTAPI
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. LOAD AI COMPONENTS
# ==========================================
print("🚀 Loading AI Components...")
try:
    lgbm_model = lgb.Booster(model_file="lgbm_ransom_detector.txt")
    lstm_model = tf.keras.models.load_model("final_lstm_model.keras")
    
    # Load the global scalers
    train_mean = np.load("train_mean.npy")
    train_std = np.load("train_std.npy")
    train_std[train_std == 0] = 1e-8
    
    print("✅ AI Models Ready.")
except Exception as e:
    print(f"❌ Initialization Failed: {e}")

# ==========================================
# 3. INITIALIZE FIREBASE (SECURELY)
# ==========================================
db = None
try:
    if not firebase_admin._apps:
        # Securely grab the JSON string from Railway's Environment Variables
        firebase_env = os.environ.get("FIREBASE_KEY")
        
        if firebase_env:
            print("☁️ Loading Firebase credentials from Environment Variable...")
            cred_dict = json.loads(firebase_env)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("✅ Firestore initialized successfully.")
        else:
            print("⚠️ FIREBASE_KEY variable not found. Database logging is disabled.")
            
except Exception as e:
    print(f"❌ Firestore Init Failed: {e}")

# ==========================================
# 4. CONFIGURATION
# ==========================================
FEATURE_COLS = [
    "mem_write_count", "mem_avg_entropy", "mem_std_entropy", 
    "mem_total_bytes", "mem_gpa_variance", "disk_write_count", 
    "disk_avg_entropy", "disk_total_bytes", "disk_lba_variance", "time_sec"
]

SEQ_LEN = 50
LGBM_LOW = 0.35
LGBM_HIGH = 0.90
LSTM_THRESH = 0.99

# ==========================================
# 5. THE ANALYTICS ENDPOINT
# ==========================================
@app.post("/upload-excel")
async def analyze_excel_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # -----------------------------------------
    # STAGE 1: LIGHTGBM (Window Analysis)
    # -----------------------------------------
    if "size_mean" in df.columns:
        trained_features = lgbm_model.feature_name()
        X_lgb = df.copy()
        for col in trained_features:
            if col not in X_lgb.columns:
                X_lgb[col] = 0
        X_lgb = X_lgb[trained_features]

        probs = lgbm_model.predict(X_lgb)
        window_results = []
        
        for i, p in enumerate(probs):
            if p < LGBM_LOW: label = "BENIGN"
            elif p > LGBM_HIGH: label = "RANSOMWARE"
            else: label = "UNCERTAIN"
            
            window_results.append({
                "window_index": i + 1,
                "probability": round(float(p), 4),
                "label": label
            })

        uncertain_count = sum(1 for w in window_results if w["label"] == "UNCERTAIN")
        
        # If UNCERTAIN, we stop here and tell the client to send LSTM data.
        # We DO NOT log to Firebase yet.
        if uncertain_count > 0:
            return {
                "model": "LightGBM",
                "overall_verdict": "UNCERTAIN",
                "next_step": "LSTM_REQUIRED",
                "detailed_windows": window_results
            }
            
        # If CONFIDENT, we finalize the verdict and log it to Firebase immediately.
        else:
            avg_p = float(np.mean(probs))
            verdict = "RANSOMWARE" if avg_p > LGBM_HIGH else "BENIGN"
            
            if db:
                action_value = "kill" if verdict == "RANSOMWARE" else "pass"
                doc_id = str(uuid.uuid4())
                db.collection("research").document(doc_id).set({
                    "lightgbm": verdict.lower(),
                    "lstm": "bypassed",
                    "action": action_value,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })

            return {
                "model": "LightGBM",
                "overall_verdict": verdict,
                "next_step": "COMPLETED",
                "detailed_windows": window_results
            }

    # -----------------------------------------
    # STAGE 2: LSTM (Deep Sequence Analysis)
    # -----------------------------------------
    elif "mem_write_count" in df.columns:
        X = df[FEATURE_COLS].values.astype(np.float32)

        # Use SAME normalization logic as Code 1
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_scaled = (X - mean) / std

        if len(X_scaled) < SEQ_LEN:
            return {"error": "File too short for LSTM analysis (min 50 rows)"}

        # Sequence building (same as Code 1)
        sequences = [X_scaled[i:i+SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN + 1)]
        X_seq = np.array(sequences, dtype=np.float32)

        # Prediction
        probs = lstm_model.predict(X_seq, verbose=0).flatten()

        # Sequence-level classification (same as Code 1)
        preds = (probs >= LSTM_THRESH).astype(int)

        total = len(preds)
        ransomware_count = int((preds == 1).sum())
        benign_count = int((preds == 0).sum())

        # ✅ CRITICAL: Use MEAN probability (NOT MAX)
        run_probability = float(probs.mean()) if len(probs) > 0 else 0.0

        final_verdict = "RANSOMWARE" if run_probability >= LSTM_THRESH else "BENIGN"

        # Log final LSTM fallback result to Firebase
        if db:
            action_value = "kill" if final_verdict == "RANSOMWARE" else "pass"
            doc_id = str(uuid.uuid4())
            db.collection("research").document(doc_id).set({
                "lightgbm": "uncertain",
                "lstm": final_verdict.lower(),
                "action": action_value,
                "timestamp": firestore.SERVER_TIMESTAMP
            })

        return {
            "model": "LSTM",
            "final_verdict": final_verdict,
            "average_probability": round(run_probability, 4),
            "total_sequences": total,
            "ransomware_sequences": ransomware_count,
            "benign_sequences": benign_count,
            "ransomware_percentage": round((ransomware_count / total) * 100, 2) if total > 0 else 0.0,
            "benign_percentage": round((benign_count / total) * 100, 2) if total > 0 else 0.0
        }

    # Fallback if no known columns are found
    return {"error": "Unrecognized CSV format"}

# ==========================================
# 6. HEALTH CHECKS
# ==========================================
@app.get("/")
async def health():
    return {"status": "online"}
