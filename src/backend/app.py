# # src/backend/app.py

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# import pandas as pd
# import joblib
# import logging
# import os
# import sys
# import traceback
# import re
# from typing import Union
# from difflib import get_close_matches
# from collections import deque
# from datetime import datetime

# # ---------------------------------------------------
# # Add project root to path so local modules resolve
# # ---------------------------------------------------
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # local helpers
# from preprocessing import clean_column_names, encode_plant_age
# from model_inference import predict
# from train import get_feature_importance
# from logging.handlers import RotatingFileHandler

# RECENT_PREDICTIONS = deque(maxlen=10)


# LOG_FILE = "Capex_Estimation_API.log"

# handler = RotatingFileHandler(
#     LOG_FILE,
#     maxBytes=5 * 1024 * 1024,  # 5 MB
#     backupCount=5              # keep last 5 files
# )

# formatter = logging.Formatter(
#     "%(asctime)s | %(levelname)s | %(message)s"
# )
# handler.setFormatter(formatter)

# logger = logging.getLogger("CAPEX_API")
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)
# logger.addHandler(logging.StreamHandler())

# logger.info("🚀 Starting CAPEX Estimation FastAPI Server…")

# # ---------------------------------------------------
# # Load Artifacts
# # ---------------------------------------------------
# try:
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     MODELS_DIR = os.path.join(BASE_DIR, "models")

#     model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
#     preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
#     feature_order = joblib.load(os.path.join(MODELS_DIR, "feature_order.pkl"))

#     # Extract numeric/categorical cols from fitted preprocessor if possible
#     try:
#         transformers = preprocessor.named_steps["preprocessor"].transformers_
#         numeric_cols_in_pipe = list(transformers[0][2])
#         categorical_cols_in_pipe = list(transformers[1][2])
#     except Exception:
#         numeric_cols_in_pipe = []
#         categorical_cols_in_pipe = []

#     logger.info("✔ Model, preprocessor, and feature order loaded successfully.")
# except Exception as e:
#     logger.error(f"❌ Failed loading artifacts: {str(e)}")
#     logger.error(traceback.format_exc())
#     raise RuntimeError("Model or preprocessing artifacts missing/corrupted.")


# # ---------------------------------------------------
# # FastAPI
# # ---------------------------------------------------
# app = FastAPI(
#     title="CAPEX Estimation API",
#     version="1.0",
#     description="Robust API for CAPEX prediction using ML models."
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------------------------------------------------
# # Input Schema
# # ---------------------------------------------------
# class InputData(BaseModel):
#     Vehicle_Type: Optional[str] = None
#     Material_Type: Optional[str] = None
#     Drivetrain: Optional[str] = None
#     Automation_Level: Optional[str] = None
#     # plant_age: Union[str, int, float]
#     plant_age : Optional[str] = None
#     Line_Reuse: Optional[str] = None
#     Lifetime_Volume: Optional[float] = None
#     Target_Annual_Volume: Optional[float] = None
#     Variants: Optional[int] = None
#     Number_of_Parts: Optional[int] = None
#     Avg_Part_Complexity: Optional[float] = None
#     BIW_Weight: Optional[float] = None
#     Stamping_Dies: Optional[float] = None
#     Injection_Molds: Optional[float] = None
#     Casting_Tools: Optional[float] = None
#     Jigs_and_Fixtures: Optional[float] = None
#     Assembly_Line_Equipment: Optional[float] = None
#     Robotics: Optional[float] = None
#     Paint_Shop_Mods: Optional[float] = None
#     # Estimated_CAPEX: Optional[float] = None

#     class Config:
#         extra = "allow"

# class BatchInputData(BaseModel):
#     data: List[InputData]

# # ---------------------------------------------------
# # Helpers — normalization & mapping
# # ---------------------------------------------------
# def _normalize_key(key: str) -> str:
#     """
#     Normalize any incoming key to a compact form for matching.
#     Removes non-alphanumerics and lowercases.
#     """
#     if key is None:
#         return ""
#     k = str(key).strip().lower()
#     k = re.sub(r"[^a-z0-9]", "", k)
#     return k

# # build mapping from normalized form -> canonical feature name (feature_order entries)
# _norm_to_feature = {re.sub(r"[^a-z0-9]", "", f.lower()): f for f in feature_order}


# def map_payload_keys(payload: dict) -> dict:
#     """
#     Map incoming payload keys (any case/format) to canonical feature names.
#     Uses exact normalized match or fuzzy match fallback.
#     """
#     mapped = {}
#     for k, v in payload.items():
#         kn = _normalize_key(k)
#         if kn in _norm_to_feature:
#             mapped_name = _norm_to_feature[kn]
#         else:
#             # fuzzy match best candidate among normalized feature keys
#             candidates = list(_norm_to_feature.keys())
#             match = get_close_matches(kn, candidates, n=1, cutoff=0.7)
#             if match:
#                 mapped_name = _norm_to_feature[match[0]]
#                 logger.debug(f"Fuzzy matched incoming key '{k}' -> '{mapped_name}'")
#             else:
#                 # unknown field: ignore it (or you may keep it under original name)
#                 logger.debug(f"Ignoring unknown incoming key: '{k}'")
#                 continue
#         mapped[mapped_name] = v
#     return mapped


# def normalize_values(df: pd.DataFrame, categorical_cols: list, numeric_cols: list) -> pd.DataFrame:
#     """
#     Normalize values:
#     - Strip and lowercase categorical strings
#     - Coerce numeric columns to numeric dtype
#     - For numeric NaNs, fill with median (from current batch) to avoid preprocessor errors
#     """
#     df = df.copy()

#     # categorical normalization
#     for c in categorical_cols:
#         if c in df.columns:
#             df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": None})

#     # numeric coercion
#     for c in numeric_cols:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")

#     # fill numeric NaNs with median of training preprocessor if available else batch median
#     for c in numeric_cols:
#         if c in df.columns:
#             if df[c].isna().any():
#                 # try to get training median from preprocessor if possible (not always available)
#                 try:
#                     # fallback: compute batch median
#                     med = df[c].median()
#                     df[c] = df[c].fillna(med)
#                 except Exception:
#                     df[c] = df[c].fillna(0.0)

#     return df


# # ---------------------------------------------------
# # Prepare input: full pipeline from raw payload -> model-ready df
# # ---------------------------------------------------
# def prepare_input_from_payload(payload: dict) -> pd.DataFrame:
#     """
#     1) Map keys to canonical feature names
#     2) Build a DataFrame with feature_order columns
#     3) encode plant_age
#     4) normalize values (categorical lowercasing, numeric coercion)
#     5) return dataframe ready for preprocessor.transform()
#     """
#     # Map keys to canonical feature names
#     mapped = map_payload_keys(payload)

#     # Create df and ensure all feature_order columns exist
#     df = pd.DataFrame([mapped])
#     for col in feature_order:
#         if col not in df.columns:
#             df[col] = None

#     # Reorder to canonical order
#     df = df[feature_order]

#     # Clean column names (defensive, though feature_order should already be canonical)
#     df = clean_column_names(df)

#     # Plant age encoding
#     df = encode_plant_age(df, column="plant_age")

#     # Normalize values
#     df = normalize_values(df, categorical_cols_in_pipe, numeric_cols_in_pipe)

#     return df

# # ---------------------------------------------------
# # Routes
# # ---------------------------------------------------
# @app.get("/")
# def root():
#     return {"message": "CAPEX Estimation API is alive. Visit /docs for usage."}


# @app.get("/health")
# def health():
#     return {"status": "OK"}


# @app.post("/predict")
# def predict_capex(data: InputData):
#     print("RAW INPUT FROM FRONTEND:", data.dict())
#     try:
#         payload = data.dict()
#         df = prepare_input_from_payload(payload)

#         # transform and predict
#         X = preprocessor.transform(df)
#         y_pred = float(model.predict(X)[0])

#         # ✅ NEW: store recent prediction
#         RECENT_PREDICTIONS.append({
#             "timestamp": datetime.now().isoformat(timespec="seconds"),
#             "predicted_capex": round(y_pred, 2)
#         })

#         logger.info(f"Prediction successful → {y_pred:.4f}")
#         return {"predicted_CAPEX": y_pred}

#     except Exception as e:
#         logger.error("❌ Prediction error: %s", str(e))
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# @app.post("/predict_batch")
# def predict_batch(batch: BatchInputData):
#     try:
#         rows = [item.dict() for item in batch.data]
#         processed_rows = []

#         for r in rows:
#             df_single = prepare_input_from_payload(r)
#             processed_rows.append(df_single)

#         df_all = pd.concat(processed_rows, ignore_index=True).reindex(columns=feature_order)
#         X = preprocessor.transform(df_all)
#         preds = model.predict(X)

#         # ✅ NEW: store each batch prediction
#         for p in preds:
#             RECENT_PREDICTIONS.append({
#                 "timestamp": datetime.now().isoformat(timespec="seconds"),
#                 "predicted_capex": round(float(p), 2)
#             })

#         return {"predictions": preds.tolist()}

#     except Exception as e:
#         logger.error("❌ Batch prediction error: %s", str(e))
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# @app.get("/recent_predictions")
# def recent_predictions():
#     return list(RECENT_PREDICTIONS)

# # @app.get("/metrics")
# # def get_model_metrics():
# #     return MODEL_METRICS

# @app.get("/feature_importance")
# def feature_importance(n: Optional[int] = 10):
#     try:
#         fi = get_feature_importance(model, feature_order, top_n=n)
#         return {"top_features": fi.to_dict(orient="records")}
#     except Exception as e:
#         logger.error("❌ Feature importance error: %s", str(e))
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Failed to compute feature importance: {str(e)}")
    

# @app.get("/categories")
# def get_categories():
#     """
#     Safely extract categorical options from the fitted preprocessor,
#     regardless of pipeline nesting.
#     """
#     try:
#         # Step 1 — Get the ColumnTransformer from pipeline
#         col_transformer = preprocessor.named_steps.get("preprocessor")

#         if col_transformer is None:
#             raise ValueError("ColumnTransformer not found in preprocessor pipeline")

#         # Step 2 — Find the categorical OneHotEncoder
#         ohe = None
#         categorical_cols = None

#         for name, transformer, columns in col_transformer.transformers_:
#             # Skip passthrough / drop
#             if transformer == "drop" or transformer == "passthrough":
#                 continue

#             # Case 1: transformer IS the OneHotEncoder
#             if hasattr(transformer, "categories_"):
#                 ohe = transformer
#                 categorical_cols = columns
#                 break

#             # Case 2: transformer is a Pipeline containing OneHotEncoder
#             if hasattr(transformer, "steps"):
#                 for step_name, step_obj in transformer.steps:
#                     if hasattr(step_obj, "categories_"):
#                         ohe = step_obj
#                         categorical_cols = columns
#                         break

#         if ohe is None:
#             raise ValueError("OneHotEncoder not found in preprocessor pipeline")

#         # Step 3 — Package the results in a clean JSON dict
#         categories_dict = {
#             col: list(values) 
#             for col, values in zip(categorical_cols, ohe.categories_)
#         }

#         return categories_dict

#     except Exception as e:
#         logger.error(f"❌ Failed to extract categories: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Failed to load categories: {str(e)}")



import os
import sys
import re
import logging
import traceback
import joblib
import pandas as pd

from typing import List, Optional
from collections import deque
from datetime import datetime
from difflib import get_close_matches

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------
# Add project root to path
# ---------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from src.model_training.preprocessing import clean_column_names, encode_plant_age
from train import get_feature_importance

# ---------------------------------------------------
# Globals
# ---------------------------------------------------
RECENT_PREDICTIONS = deque(maxlen=10)

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
LOG_FILE = "Capex_Estimation_API.log"

handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,
    backupCount=5
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)
handler.setFormatter(formatter)

logger = logging.getLogger("CAPEX_API")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())

logger.info("🚀 Starting CAPEX Estimation FastAPI Server…")

# ---------------------------------------------------
# Load Artifacts
# ---------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    feature_order = joblib.load(os.path.join(MODELS_DIR, "feature_order.pkl"))
    MODEL_METRICS = joblib.load(os.path.join(MODELS_DIR, "model_metrics.pkl"))

    # Try extracting column info from pipeline
    try:
        transformers = preprocessor.named_steps["preprocessor"].transformers_
        numeric_cols_in_pipe = list(transformers[0][2])
        categorical_cols_in_pipe = list(transformers[1][2])
    except Exception:
        numeric_cols_in_pipe = []
        categorical_cols_in_pipe = []

    logger.info("✔ All artifacts loaded successfully.")

except Exception as e:
    logger.error("❌ Artifact loading failed")
    logger.error(traceback.format_exc())
    raise RuntimeError("Model or metrics artifacts missing")

# ---------------------------------------------------
# FastAPI Init
# ---------------------------------------------------
app = FastAPI(
    title="CAPEX Estimation API",
    version="1.0",
    description="ML-backed CAPEX estimation service"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Input Schemas
# ---------------------------------------------------
class InputData(BaseModel):
    Vehicle_Type: Optional[str] = None
    Material_Type: Optional[str] = None
    Drivetrain: Optional[str] = None
    Automation_Level: Optional[str] = None
    plant_age: Optional[str] = None
    Line_Reuse: Optional[str] = None
    Lifetime_Volume: Optional[float] = None
    Target_Annual_Volume: Optional[float] = None
    Variants: Optional[int] = None
    Number_of_Parts: Optional[int] = None
    Avg_Part_Complexity: Optional[float] = None
    BIW_Weight: Optional[float] = None
    Stamping_Dies: Optional[float] = None
    Injection_Molds: Optional[float] = None
    Casting_Tools: Optional[float] = None
    Jigs_and_Fixtures: Optional[float] = None
    Assembly_Line_Equipment: Optional[float] = None
    Robotics: Optional[float] = None
    Paint_Shop_Mods: Optional[float] = None

    class Config:
        extra = "allow"


class BatchInputData(BaseModel):
    data: List[InputData]

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


_norm_to_feature = {
    _normalize_key(f): f for f in feature_order
}


def map_payload_keys(payload: dict) -> dict:
    mapped = {}
    for k, v in payload.items():
        kn = _normalize_key(k)
        if kn in _norm_to_feature:
            mapped[_norm_to_feature[kn]] = v
        else:
            match = get_close_matches(kn, _norm_to_feature.keys(), n=1, cutoff=0.7)
            if match:
                mapped[_norm_to_feature[match[0]]] = v
    return mapped


def normalize_values(df, categorical_cols, numeric_cols):
    df = df.copy()

    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip().replace("nan", None)

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    return df


def prepare_input_from_payload(payload: dict) -> pd.DataFrame:
    mapped = map_payload_keys(payload)
    df = pd.DataFrame([mapped])

    for col in feature_order:
        if col not in df.columns:
            df[col] = None

    df = df[feature_order]
    df = clean_column_names(df)
    df = encode_plant_age(df, column="plant_age")
    df = normalize_values(df, categorical_cols_in_pipe, numeric_cols_in_pipe)

    return df

# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.get("/")
def root():
    return {"message": "CAPEX Estimation API running"}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict")
def predict_capex(data: InputData):
    try:
        df = prepare_input_from_payload(data.dict())
        X = preprocessor.transform(df)
        y_pred = float(model.predict(X)[0])

        RECENT_PREDICTIONS.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "predicted_capex": round(y_pred, 2)
        })

        return {"predicted_CAPEX": round(y_pred, 2)}

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed")

# @app.post("/predict_batch")
# def predict_batch(batch: BatchInputData):
#     try:
#         dfs = [prepare_input_from_payload(d.dict()) for d in batch.data]
#         df_all = pd.concat(dfs, ignore_index=True)

#         X = preprocessor.transform(df_all)
#         preds = model.predict(X)

#         for p in preds:
#             RECENT_PREDICTIONS.append({
#                 "timestamp": datetime.now().isoformat(timespec="seconds"),
#                 "predicted_capex": round(float(p), 2)
#             })

#         return {"predictions": preds.tolist()}

#     except Exception:
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.post("/predict_batch")
def predict_batch(batch: BatchInputData):
    try:
        processed = []

        for item in batch.data:
            try:
                df = prepare_input_from_payload(item.dict())
                processed.append(df)
            except Exception as e:
                logger.warning(f"Skipping invalid row: {e}")

        if not processed:
            raise HTTPException(status_code=400, detail="No valid rows found")

        df_all = pd.concat(processed, ignore_index=True)
        X = preprocessor.transform(df_all)
        preds = model.predict(X)

        for p in preds:
            RECENT_PREDICTIONS.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "predicted_capex": round(float(p), 2)
            })

        return {"predictions": preds.tolist()}

    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Batch prediction failed")



@app.get("/recent_predictions")
def recent_predictions():
    return list(RECENT_PREDICTIONS)

@app.get("/metrics")
def get_model_metrics():
    return MODEL_METRICS

@app.get("/feature_importance")
def feature_importance(n: int = 10):
    fi = get_feature_importance(model, feature_order, top_n=n)
    return {"top_features": fi.to_dict(orient="records")}

@app.get("/categories")
def get_categories():
    """
    Safely extract categorical options from the fitted preprocessor,
    regardless of pipeline nesting.
    """
    try:
        # Step 1 — Get the ColumnTransformer from pipeline
        col_transformer = preprocessor.named_steps.get("preprocessor")

        if col_transformer is None:
            raise ValueError("ColumnTransformer not found in preprocessor pipeline")

        # Step 2 — Find the categorical OneHotEncoder
        ohe = None
        categorical_cols = None

        for name, transformer, columns in col_transformer.transformers_:
            # Skip passthrough / drop
            if transformer == "drop" or transformer == "passthrough":
                continue

            # Case 1: transformer IS the OneHotEncoder
            if hasattr(transformer, "categories_"):
                ohe = transformer
                categorical_cols = columns
                break

            # Case 2: transformer is a Pipeline containing OneHotEncoder
            if hasattr(transformer, "steps"):
                for step_name, step_obj in transformer.steps:
                    if hasattr(step_obj, "categories_"):
                        ohe = step_obj
                        categorical_cols = columns
                        break

        if ohe is None:
            raise ValueError("OneHotEncoder not found in preprocessor pipeline")

        # Step 3 — Package the results in a clean JSON dict
        categories_dict = {
            col: list(values) 
            for col, values in zip(categorical_cols, ohe.categories_)
        }

        return categories_dict

    except Exception as e:
        logger.error(f"❌ Failed to extract categories: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to load categories: {str(e)}")
