# # main.py - REFACTORED VERSION WITH NEAT LOGGING
# import os
# import logging
# import joblib
# import pandas as pd
# from sklearn.calibration import LabelEncoder
# from config import load_csv
# from data_validation import inspect_data, remove_duplicates
# from train_test_split import split_data
# from train import get_feature_importance, train_model, save_model
# from model_inference import predict, evaluate_model
# from preprocessing import build_preprocessing_pipeline, fit_preprocessor, transform_preprocessor

# # --------------------------
# # Configure logging
# # --------------------------
# log_file = "Capex_Estimation_Pipeline.log"
# logging.basicConfig(
#     filename=log_file,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )

# logger = logging.getLogger("Capex_Estimation_Pipeline")

# # Add console output as well
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_formatter = logging.Formatter("%(levelname)s - %(message)s")
# console_handler.setFormatter(console_formatter)
# logger.addHandler(console_handler)

# logger.info("\n\n" + "="*60)
# logger.info("CAPEX ESTIMATION PIPELINE STARTED")
# logger.info("="*60 + "\n")

# # --------------------------
# # Setup directories
# # --------------------------
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# models_dir = os.path.join(project_root, "models")
# data_dir = os.path.join(project_root, "data")
# results_dir = os.path.join(project_root, "results")

# for directory in [models_dir, data_dir, results_dir]:
#     os.makedirs(directory, exist_ok=True)
#     logger.info(f"Ensuring directory exists: {directory}")

# logger.info("\n")  # extra space

# # --------------------------
# # Load data
# # --------------------------
# logger.info("STEP 1: Loading data")
# data_path = os.path.join(data_dir, "vehicle_program_data.csv")
# logger.info(f"Loading data from: {data_path}")
# df = load_csv(data_path)
# logger.info(f"Data loaded successfully. Shape: {df.shape}\n")

# logger.info("First 5 rows of dataset:\n%s", df.head().to_string(index=False))
# logger.info("\n")  # spacing

# # --------------------------
# # Inspect and clean data
# # --------------------------
# logger.info("STEP 2: Inspecting and cleaning data")
# inspect_data(df)
# df = remove_duplicates(df)
# logger.info(f"Data shape after removing duplicates: {df.shape}\n")

# # --------------------------
# # Split data
# # --------------------------
# logger.info("STEP 3: Splitting data into train, validation, and test sets")
# X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column="Estimated_CAPEX")
# logger.info(f"Train shape: {X_train.shape}")
# logger.info(f"Validation shape: {X_val.shape}")
# logger.info(f"Test shape: {X_test.shape}\n")

# # --------------------------
# # Identify categorical and numeric columns
# # --------------------------
# logger.info("STEP 4: Identifying column types")
# categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
# numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
# logger.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
# logger.info(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}\n")

# # --------------------------
# # Preprocessing
# # --------------------------
# logger.info("STEP 5: Building and applying preprocessing pipeline")
# preprocessor = build_preprocessing_pipeline(categorical_cols, numeric_cols)
# logger.info("Preprocessing pipeline built\n")

# logger.info("Fitting full preprocessing pipeline on training data")
# preprocessor.fit(X_train)

# logger.info("Transforming training data")
# X_train_processed = transform_preprocessor(preprocessor, X_train)
# logger.info(f"Training data processed shape: {X_train_processed.shape}")

# logger.info("Transforming validation data")
# X_val_processed = transform_preprocessor(preprocessor, X_val)
# logger.info(f"Validation data processed shape: {X_val_processed.shape}")

# logger.info("Transforming test data")
# X_test_processed = transform_preprocessor(preprocessor, X_test)
# logger.info(f"Test data processed shape: {X_test_processed.shape}\n")


# # --------------------------
# # Save preprocessing artifacts
# # --------------------------
# logger.info("STEP 6: Saving preprocessing pipeline")
# try:
#     joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
#     logger.info("✓ Preprocessor saved successfully\n")

#     # Save processed datasets
#     X_train_processed.to_csv(os.path.join(data_dir, "X_train_processed.csv"), index=False)
#     X_val_processed.to_csv(os.path.join(data_dir, "X_val_processed.csv"), index=False)
#     X_test_processed.to_csv(os.path.join(data_dir, "X_test_processed.csv"), index=False)

#     y_train.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
#     y_val.to_csv(os.path.join(data_dir, "y_val.csv"), index=False)
#     y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)

#     logger.info("✓ Processed data saved successfully\n")

# except Exception as e:
#     logger.error(f"Error saving files: {e}\n")
#     logger.error("Check write permissions in the models/data directories\n")
# #---------------------------
# # Train model
# # --------------------------
# logger.info("STEP 7: Training model")
# best_rf = train_model(X_train_processed, y_train, random_search=True)
# save_model(best_rf, os.path.join(models_dir, "rf_model.pkl"))
# logger.info("✓ Model trained and saved successfully\n")

# # --------------------------
# # Evaluate on validation set
# # --------------------------
# logger.info("STEP 8: Evaluation on Validation Set")
# predictions = predict(best_rf, X_val_processed)
# metrics = evaluate_model(y_val, predictions)

# logger.info("Evaluation Metrics (Validation Set):")
# logger.info(f"MAE:  {metrics['MAE']:.4f}")
# logger.info(f"MSE:  {metrics['MSE']:.4f}")
# logger.info(f"RMSE: {metrics['RMSE']:.4f}")
# logger.info(f"R-sqaured:   {metrics['R2']:.4f}\n")

# # --------------------------
# # Feature importance
# # --------------------------
# logger.info("STEP 9: Calculating feature importance")
# feature_names = X_train_processed.columns.tolist()
# top_features = get_feature_importance(best_rf, feature_names, top_n=12)

# logger.info("Top 12 Features:\n%s", top_features.to_string(index=False))
# print("\n=== Top 12 Important Features ===")
# for i, row in enumerate(top_features.itertuples(index=False), 1):
#     print(f"{i:2}. {row.feature:30} : {row.importance:.4f}")
# logger.info("\n")

# # --------------------------
# # Evaluate on test set
# # --------------------------
# logger.info("STEP 10: Evaluation on Test Set")
# test_predictions = predict(best_rf, X_test_processed)
# test_metrics = evaluate_model(y_test, test_predictions)
# logger.info("Evaluation Metrics (Test Set):")
# logger.info(f"MAE:  {test_metrics['MAE']:.4f}")
# logger.info(f"RMSE: {test_metrics['RMSE']:.4f}")
# logger.info(f"R-squared:   {test_metrics['R2']:.4f}\n")

# print("\n=== Evaluation Metrics on Test Set ===")
# print(f"MAE:  {test_metrics['MAE']:.4f}")
# print(f"RMSE: {test_metrics['RMSE']:.4f}")
# print(f"R²:   {test_metrics['R2']:.4f}\n")

# # --------------------------
# # Pipeline completed
# # --------------------------
# logger.info("="*60)
# logger.info("CAPEX ESTIMATION PIPELINE COMPLETED SUCCESSFULLY")
# logger.info("="*60 + "\n")
# logger.info(f"All models and preprocessors saved in: {models_dir}")
# logger.info(f"Processed data saved in: {data_dir}\n")

# main.py - Final Version
import os
import logging
import joblib
import pandas as pd
from config import load_csv
from data_validation import inspect_data, remove_duplicates
from train_test_split import split_data
from train import get_feature_importance, train_model
from preprocessing import (
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
# Load & clean data
# --------------------------
data_path = os.path.join(data_dir, "vehicle_program_data.csv")
logger.info(f"Loading data from: {data_path}")
df = load_csv(data_path)

inspect_data(df)
df = remove_duplicates(df)

# Clean column names and encode plant_age
df = clean_column_names(df)
df = encode_plant_age(df, column="plant_age")

# --------------------------
# Split data
# --------------------------
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column="estimated_capex")

# Identify categorical and numeric columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Define feature order
feature_order = categorical_cols + numeric_cols

# --------------------------
# Build + fit preprocessing pipeline
# --------------------------
preprocessor = build_preprocessing_pipeline(categorical_cols, numeric_cols)
preprocessor = fit_preprocessor(preprocessor, X_train, feature_order)

# Transform data
X_train_processed = transform_preprocessor(preprocessor, X_train, feature_order)
X_val_processed = transform_preprocessor(preprocessor, X_val, feature_order)
X_test_processed = transform_preprocessor(preprocessor, X_test, feature_order)

# --------------------------
# Train model
# --------------------------
best_rf, preprocessor, feature_order = train_model(
    X_train, y_train,
    categorical_cols=categorical_cols,
    numeric_cols=numeric_cols,
    random_search=True
)

# --------------------------
# Save artifacts
# --------------------------
joblib.dump(best_rf, os.path.join(models_dir, "rf_model.pkl"))
joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
joblib.dump(feature_order, os.path.join(models_dir, "feature_order.pkl"))
logger.info(f"Saved model, preprocessor, and feature_order to {models_dir}")

# --------------------------
# Evaluate
# --------------------------
pred_val = best_rf.predict(X_val_processed)
pred_test = best_rf.predict(X_test_processed)

logger.info("Validation R²: %.4f", best_rf.score(X_val_processed, y_val))
logger.info("Test R²: %.4f", best_rf.score(X_test_processed, y_test))

# Feature importance
top_features = get_feature_importance(best_rf, feature_order, top_n=12)
logger.info("Top 12 Features:\n%s", top_features)

logger.info("=== CAPEX ESTIMATION PIPELINE COMPLETED SUCCESSFULLY ===")
