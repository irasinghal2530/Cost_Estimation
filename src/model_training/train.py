# # # # train_model.py
# # # from sklearn.ensemble import RandomForestRegressor
# # # from sklearn.model_selection import RandomizedSearchCV
# # # from sklearn.base import BaseEstimator
# # # import joblib
# # # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # # import numpy as np

# # # def train_model(X_train, y_train, X_test=None, y_test=None, random_search=True):
# # #     """
# # #     Train a RandomForestRegressor with optional RandomizedSearchCV hyperparameter tuning.
# # #     """
# # #     if random_search:
# # #         param_dist = {
# # #             'n_estimators': [100, 200, 300, 500, 800],
# # #             'max_depth': [5, 10, 20, 30, None],
# # #             'min_samples_split': [2, 5, 10, 15],
# # #             'min_samples_leaf': [1, 2, 5, 10],
# # #             'max_features': ['sqrt', 'log2', 0.5, 0.7],  # removed 'auto'
# # #             'bootstrap': [True, False]
# # #         }

# # #         rf = RandomForestRegressor(random_state=42)

# # #         rf_random = RandomizedSearchCV(
# # #             estimator=rf,
# # #             param_distributions=param_dist,
# # #             n_iter=50,
# # #             scoring='r2',
# # #             cv=5,
# # #             verbose=2,
# # #             random_state=42,
# # #             n_jobs=-1
# # #         )

# # #         rf_random.fit(X_train, y_train)
# # #         best_model = rf_random.best_estimator_

# # #         if X_test is not None and y_test is not None:
# # #             y_pred = best_model.predict(X_test)
# # #             mae = mean_absolute_error(y_test, y_pred)
# # #             mse = mean_squared_error(y_test, y_pred)
# # #             rmse = np.sqrt(mse)
# # #             r2 = r2_score(y_test, y_pred)
# # #             print("=== Test Metrics ===")
# # #             print(f"MAE: {mae:.4f}")
# # #             print(f"MSE: {mse:.4f}")
# # #             print(f"RMSE: {rmse:.4f}")
# # #             print(f"R2: {r2:.4f}")

# # #         print("Best Hyperparameters:", rf_random.best_params_)
# # #         print("Best CV R2:", rf_random.best_score_)

# # #         return best_model

# # #     else:
# # #         model = RandomForestRegressor(n_estimators=100, random_state=42)
# # #         model.fit(X_train, y_train)
# # #         return model

# # # def save_model(model: BaseEstimator, filepath: str):
# # #     joblib.dump(model, filepath)

# # # def load_model(filepath: str):
# # #     return joblib.load(filepath)


# # # train_model.py
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.model_selection import RandomizedSearchCV
# # from sklearn.base import BaseEstimator
# # import joblib

# # def train_model(X_train, y_train, random_search=True):
# #     """
# #     Train a RandomForestRegressor with optional RandomizedSearchCV hyperparameter tuning.
# #     Returns the trained model.
# #     """
# #     if random_search:
# #         param_dist = {
# #             'n_estimators': [100, 200, 300, 500, 800],
# #             'max_depth': [5, 10, 20, 30, None],
# #             'min_samples_split': [2, 5, 10, 15],
# #             'min_samples_leaf': [1, 2, 5, 10],
# #             'max_features': ['sqrt', 'log2', 0.5, 0.7],
# #             'bootstrap': [True, False]
# #         }

# #         rf = RandomForestRegressor(random_state=42)
# #         rf_random = RandomizedSearchCV(
# #             estimator=rf,
# #             param_distributions=param_dist,
# #             n_iter=50,
# #             scoring='r2',
# #             cv=5,
# #             verbose=2,
# #             random_state=42,
# #             n_jobs=-1
# #         )
# #         rf_random.fit(X_train, y_train)
# #         best_model = rf_random.best_estimator_
# #         print("Best Hyperparameters:", rf_random.best_params_)
# #         print("Best CV R2:", rf_random.best_score_)

# #     else:
# #         best_model = RandomForestRegressor(n_estimators=100, random_state=42)
# #         best_model.fit(X_train, y_train)

# #     return best_model

# # ## Also find the feature importance function here


# # def save_model(model: BaseEstimator, filepath: str):
# #     joblib.dump(model, filepath)

# # def load_model(filepath: str):
# #     return joblib.load(filepath)


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.base import BaseEstimator
# import joblib
# import pandas as pd

# def train_model(X_train, y_train, random_search=True):
#     """
#     Train a RandomForestRegressor with optional RandomizedSearchCV hyperparameter tuning.
#     Returns the trained model.
#     """
#     if random_search:
#         param_dist = {
#             'n_estimators': [100, 200, 300, 500, 800],
#             'max_depth': [5, 10, 20, 30, None],
#             'min_samples_split': [2, 5, 10, 15],
#             'min_samples_leaf': [1, 2, 5, 10],
#             'max_features': ['sqrt', 'log2', 0.5, 0.7],
#             'bootstrap': [True, False]
#         }

#         rf = RandomForestRegressor(random_state=42)
#         rf_random = RandomizedSearchCV(
#             estimator=rf,
#             param_distributions=param_dist,
#             n_iter=50,
#             scoring='r2',
#             cv=5,
#             verbose=2,
#             random_state=42,
#             n_jobs=-1
#         )
#         rf_random.fit(X_train, y_train)
#         best_model = rf_random.best_estimator_
#         print("Best Hyperparameters:", rf_random.best_params_)
#         print("Best CV R2:", rf_random.best_score_)

#     else:
#         best_model = RandomForestRegressor(n_estimators=100, random_state=42)
#         best_model.fit(X_train, y_train)

#     return best_model


# def get_feature_importance(model: RandomForestRegressor, feature_names: list, top_n: int = 12):
#     """
#     Returns a DataFrame with the top N features and their importance.
#     """
#     fi = pd.DataFrame({
#         "feature": feature_names,
#         "importance": model.feature_importances_
#     }).sort_values(by="importance", ascending=False).head(top_n)
#     return fi


# def save_model(model: BaseEstimator, filepath: str):
#     joblib.dump(model, filepath)


# def load_model(filepath: str):
#     return joblib.load(filepath)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
import joblib
import pandas as pd
from src.model_training.preprocessing import build_preprocessing_pipeline, fit_preprocessor

def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                categorical_cols: list, numeric_cols: list, 
                random_search=True):
    """
    Train RandomForestRegressor with optional hyperparameter tuning.
    Returns fitted model and preprocessing pipeline.
    """
    feature_order = categorical_cols + numeric_cols

    # ----------------------------
    # 1. Build + fit preprocessing
    # ----------------------------
    preprocessor = build_preprocessing_pipeline(categorical_cols, numeric_cols)
    preprocessor = fit_preprocessor(preprocessor, X_train, feature_order)

    # Transform training data
    X_train_processed = preprocessor.transform(X_train)

    # ----------------------------
    # 2. Train model
    # ----------------------------
    if random_search:
        param_dist = {
            'n_estimators': [100, 200, 300, 500, 800],
            'max_depth': [5, 10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5, 0.7],
            'bootstrap': [True, False]
        }

        rf = RandomForestRegressor(random_state=42)
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=50,
            scoring='r2',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        rf_random.fit(X_train_processed, y_train)
        best_model = rf_random.best_estimator_
        print("Best Hyperparameters:", rf_random.best_params_)
        print("Best CV R2:", rf_random.best_score_)
    else:
        best_model = RandomForestRegressor(n_estimators=100, random_state=42)
        best_model.fit(X_train_processed, y_train)

    return best_model, preprocessor, feature_order


def get_feature_importance(model: RandomForestRegressor, feature_names: list, top_n: int = 12):
    """
    Returns top N features as DataFrame.
    """
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False).head(top_n)
    return fi


def save_model(model: BaseEstimator, preprocessor, feature_order, folder: str):
    """
    Save model, preprocessor, and feature_order metadata.
    """
    joblib.dump(model, f"{folder}/rf_model.pkl")
    joblib.dump(preprocessor, f"{folder}/preprocessor.pkl")
    joblib.dump(feature_order, f"{folder}/feature_order.pkl")


def load_model(folder: str):
    model = joblib.load(f"{folder}/rf_model.pkl")
    preprocessor = joblib.load(f"{folder}/preprocessor.pkl")
    feature_order = joblib.load(f"{folder}/feature_order.pkl")
    return model, preprocessor, feature_order
