
# # main.py - Final Version
# import os
# import logging
# import joblib
# import pandas as pd
# from config import load_csv
# from data_validation import inspect_data, remove_duplicates
# from train_test_split import split_data
# from train import get_feature_importance, train_model
# from preprocessing import (
#     build_preprocessing_pipeline,
#     fit_preprocessor,
#     transform_preprocessor,
#     encode_plant_age,
#     clean_column_names
# )
# # --------------------------
# # Logging
# # --------------------------
# log_file = "Capex_Estimation_Pipeline.log"
# logging.basicConfig(
#     filename=log_file,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger("Capex_Estimation_Pipeline")
# logger.addHandler(logging.StreamHandler())
# logger.info("=== CAPEX ESTIMATION PIPELINE STARTED ===")

# # --------------------------
# # Directories
# # --------------------------
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# models_dir = os.path.join(project_root, "models")
# data_dir = os.path.join(project_root, "data")
# os.makedirs(models_dir, exist_ok=True)
# os.makedirs(data_dir, exist_ok=True)

# # --------------------------
# # Load & clean data
# # --------------------------
# data_path = os.path.join(data_dir, "vehicle_program_data.csv")
# logger.info(f"Loading data from: {data_path}")
# df = load_csv(data_path)

# inspect_data(df)
# df = remove_duplicates(df)

# # Clean column names and encode plant_age
# df = clean_column_names(df)
# df = encode_plant_age(df, column="plant_age")

# # --------------------------
# # Split data
# # --------------------------
# X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column="estimated_capex")

# # Identify categorical and numeric columns
# categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
# numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# # Define feature order
# feature_order = categorical_cols + numeric_cols

# # --------------------------
# # Build + fit preprocessing pipeline
# # --------------------------
# preprocessor = build_preprocessing_pipeline(categorical_cols, numeric_cols)
# preprocessor = fit_preprocessor(preprocessor, X_train, feature_order)

# # Transform data
# X_train_processed = transform_preprocessor(preprocessor, X_train, feature_order)
# X_val_processed = transform_preprocessor(preprocessor, X_val, feature_order)
# X_test_processed = transform_preprocessor(preprocessor, X_test, feature_order)

# # --------------------------
# # Train model
# # --------------------------
# best_rf, preprocessor, feature_order = train_model(
#     X_train, y_train,
#     categorical_cols=categorical_cols,
#     numeric_cols=numeric_cols,
#     random_search=True
# )

# # --------------------------
# # Save artifacts
# # --------------------------
# joblib.dump(best_rf, os.path.join(models_dir, "rf_model.pkl"))
# joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
# joblib.dump(feature_order, os.path.join(models_dir, "feature_order.pkl"))
# logger.info(f"Saved model, preprocessor, and feature_order to {models_dir}")

# # --------------------------
# # Evaluate
# # --------------------------
# pred_val = best_rf.predict(X_val_processed)
# pred_test = best_rf.predict(X_test_processed)

# logger.info("Validation R²: %.4f", best_rf.score(X_val_processed, y_val))
# logger.info("Test R²: %.4f", best_rf.score(X_test_processed, y_test))

# # Feature importance
# top_features = get_feature_importance(best_rf, feature_order, top_n=12)
# logger.info("Top 12 Features:\n%s", top_features)

# logger.info("=== CAPEX ESTIMATION PIPELINE COMPLETED SUCCESSFULLY ===")



# main.py - Final Production Version

import os
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error
)

from config import load_csv
from src.model_training.data_validation import inspect_data, remove_duplicates
from train_test_split import split_data
from train import get_feature_importance, train_model
from src.model_training.preprocessing import (
    build_preprocessing_pipeline,
    fit_preprocessor,
    transform_preprocessor,
    encode_plant_age,
    clean_column_names
)

# --------------------------
# Logging
# --------------------------
log_file = "Capex_Estimation_Pipeline.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Capex_Estimation_Pipeline")
logger.addHandler(logging.StreamHandler())
logger.info("=== CAPEX ESTIMATION PIPELINE STARTED ===")

# --------------------------
# Directories
# --------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, "models")
data_dir = os.path.join(project_root, "data")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# --------------------------
# Load & Clean Data
# --------------------------
data_path = os.path.join(data_dir, "vehicle_program_data.csv")
logger.info(f"Loading data from: {data_path}")

df = load_csv(data_path)

inspect_data(df)
df = remove_duplicates(df)

df = clean_column_names(df)
df = encode_plant_age(df, column="plant_age")

# --------------------------
# Train / Val / Test Split
# --------------------------
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    df,
    target_column="estimated_capex"
)

categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

feature_order = categorical_cols + numeric_cols

# --------------------------
# Preprocessing
# --------------------------
preprocessor = build_preprocessing_pipeline(
    categorical_cols,
    numeric_cols
)

preprocessor = fit_preprocessor(
    preprocessor,
    X_train,
    feature_order
)

X_train_p = transform_preprocessor(preprocessor, X_train, feature_order)
X_val_p = transform_preprocessor(preprocessor, X_val, feature_order)
X_test_p = transform_preprocessor(preprocessor, X_test, feature_order)

# --------------------------
# Train Model
# --------------------------
best_model, preprocessor, feature_order = train_model(
    X_train,
    y_train,
    categorical_cols=categorical_cols,
    numeric_cols=numeric_cols,
    random_search=True
)

# --------------------------
# Predictions
# --------------------------
pred_val = best_model.predict(X_val_p)
pred_test = best_model.predict(X_test_p)

# --------------------------
# Regression Metrics
# --------------------------
r2_val = r2_score(y_val, pred_val)
r2_test = r2_score(y_test, pred_test)

n_val = X_val_p.shape[0]
p = X_val_p.shape[1]

adj_r2_val = 1 - (1 - r2_val) * (n_val - 1) / (n_val - p - 1)

mae_val = mean_absolute_error(y_val, pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
mape_val = np.mean(np.abs((y_val - pred_val) / y_val)) * 100
median_ae_val = median_absolute_error(y_val, pred_val)

mae_test = mean_absolute_error(y_test, pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

MODEL_METRICS = {
    "model_name": type(best_model).__name__,
    "n_features": int(p),

    "samples": {
        "train": int(X_train_p.shape[0]),
        "validation": int(X_val_p.shape[0]),
        "test": int(X_test_p.shape[0]),
    },

    "validation": {
        "r2": round(r2_val, 4),
        "adjusted_r2": round(adj_r2_val, 4),
        "mae": round(mae_val, 2),
        "rmse": round(rmse_val, 2),
        "mape": round(mape_val, 2),
        "median_ae": round(median_ae_val, 2),
    },

    "test": {
        "r2": round(r2_test, 4),
        "mae": round(mae_test, 2),
        "rmse": round(rmse_test, 2),
    }
}

# --------------------------
# Feature Importance
# --------------------------
top_features = get_feature_importance(
    best_model,
    feature_order,
    top_n=12
)

# --------------------------
# Save Artifacts
# --------------------------
joblib.dump(best_model, os.path.join(models_dir, "rf_model.pkl"))
joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
joblib.dump(feature_order, os.path.join(models_dir, "feature_order.pkl"))
joblib.dump(MODEL_METRICS, os.path.join(models_dir, "model_metrics.pkl"))
joblib.dump(top_features, os.path.join(models_dir, "feature_importance.pkl"))

logger.info("Saved all artifacts to models/")
logger.info("Validation R²: %.4f", r2_val)
logger.info("Test R²: %.4f", r2_test)
logger.info("Top Features:\n%s", top_features)

logger.info("=== CAPEX ESTIMATION PIPELINE COMPLETED SUCCESSFULLY ===")
