
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OrdinalEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import re

# # -------------------------
# # 1. Clean column names
# # -------------------------
# def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df.columns = (
#         df.columns.str.strip()
#                   .str.lower()
#                   .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
#                   .str.replace(r"_+", "_", regex=True)
#     )
#     return df

# # -------------------------
# # 2. Encode plant_age safely
# # -------------------------
# def encode_plant_age(df: pd.DataFrame, column="plant_age") -> pd.DataFrame:
#     df = df.copy()

#     def age_to_numeric(val):
#         if pd.isna(val) or val == "":
#             return -1
#         val = str(val).lower().replace("yrs", "").strip()
#         if val == "":
#             return -1
#         # Handle <X years
#         if "<" in val:
#             try:
#                 return float(re.findall(r"\d+", val)[0]) / 2
#             except: return -1
#         # Handle >X years
#         if ">" in val:
#             try:
#                 return float(re.findall(r"\d+", val)[0]) + 5
#             except: return -1
#         # Handle range X-Y
#         if "-" in val:
#             try:
#                 nums = [float(n) for n in val.split("-") if n.strip() != ""]
#                 if len(nums) == 0:
#                     return -1
#                 return sum(nums) / len(nums)
#             except:
#                 return -1
#         # Single numeric value
#         try:
#             return float(val)
#         except:
#             return -1

#     if column in df.columns:
#         df[column] = df[column].apply(age_to_numeric)
#     return df

# # -------------------------
# # 3. Validate & reorder columns
# # -------------------------
# def validate_columns(df: pd.DataFrame, feature_order: list):
#     df = clean_column_names(df)
#     df = encode_plant_age(df)
#     missing = set(feature_order) - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing columns: {missing}")
#     df = df[feature_order]
#     return df

# # -------------------------
# # 4. Build preprocessing pipeline
# # -------------------------
# def build_preprocessing_pipeline(categorical_cols, numeric_cols):
#     numeric_transformer = StandardScaler()
#     categorical_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, numeric_cols),
#             ("cat", categorical_transformer, categorical_cols)
#         ]
#     )
#     return Pipeline([("preprocessor", preprocessor)])

# # -------------------------
# # 5. Fit pipeline
# # -------------------------
# def fit_preprocessor(pipe, X_train: pd.DataFrame, feature_order: list):
#     X_train = validate_columns(X_train, feature_order)
#     pipe.fit(X_train)
#     return pipe

# # -------------------------
# # 6. Transform
# # -------------------------
# def transform_preprocessor(pipe, X: pd.DataFrame, feature_order: list):
#     X = validate_columns(X, feature_order)
#     X_arr = pipe.transform(X)
#     numeric_cols = pipe.named_steps["preprocessor"].transformers_[0][2]
#     categorical_cols = pipe.named_steps["preprocessor"].transformers_[1][2]
#     feature_names = list(numeric_cols) + list(categorical_cols)
#     return pd.DataFrame(X_arr, columns=feature_names)



import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
from difflib import get_close_matches


# ====================================================
# 0. Master Controlled Vocabulary (Your True Column Names)
# ====================================================
EXPECTED_COLS = [
    "vehicle_type",
    "material_type",
    "drivetrain",
    "automation_level",
    "plant_age",
    "line_reuse",
    "lifetime_volume",
    "target_annual_volume",
    "variants",
    "number_of_parts",
    "avg_part_complexity",
    "biw_weight",
    "stamping_dies",
    "injection_molds",
    "casting_tools",
    "jigs_and_fixtures",
    "assembly_line_equipment",
    "robotics",
    "paint_shop_mods",
]


# ====================================================
# 1. Canonical column name normalizer
# ====================================================
def normalize_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


# ====================================================
# 2. Full Dataframe Column Normalizer
# ====================================================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]
    return df


# ====================================================
# 3. Fuzzy Matching + Mapping to Expected Schema
# ====================================================
def fuzzy_map_to_expected(col: str) -> str:
    col_norm = normalize_name(col)
    match = get_close_matches(col_norm, EXPECTED_COLS, n=1, cutoff=0.7)
    return match[0] if match else col_norm  # return best guess or original


def map_columns_to_expected(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapped = {}
    for c in df.columns:
        mapped[c] = fuzzy_map_to_expected(c)

    df.columns = list(mapped.values())
    return df


# ====================================================
# 4. Robust Plant Age Handling
# ====================================================
def encode_plant_age(df: pd.DataFrame, column="plant_age") -> pd.DataFrame:
    df = df.copy()

    def age_to_numeric(val):
        if pd.isna(val) or str(val).strip() == "":
            return -1

        val = str(val).lower().replace("yrs", "").replace("years", "").strip()

        # <X
        if val.startswith("<"):
            nums = re.findall(r"\d+", val)
            return float(nums[0]) / 2 if nums else -1

        # >X
        if val.startswith(">"):
            nums = re.findall(r"\d+", val)
            return float(nums[0]) + 5 if nums else -1

        # X-Y
        if "-" in val:
            nums = [n for n in re.findall(r"\d+", val)]
            nums = [float(x) for x in nums]
            return sum(nums) / len(nums) if nums else -1

        # Single value
        try:
            return float(val)
        except:
            return -1

    if column in df.columns:
        df[column] = df[column].apply(age_to_numeric)

    return df


# ====================================================
# 5. Validate, Map, Normalize, Reorder
# ====================================================
def validate_and_prepare(df: pd.DataFrame, feature_order: list):
    df = clean_column_names(df)
    df = map_columns_to_expected(df)
    df = encode_plant_age(df)

    missing = set(feature_order) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    df = df[feature_order]
    return df


# ====================================================
# 6. Build preprocessing pipeline
# ====================================================
def build_preprocessing_pipeline(categorical_cols, numeric_cols):
    numeric_transformer = StandardScaler()
    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return Pipeline([("preprocessor", preprocessor)])


# ====================================================
# 7. Fit
# ====================================================
def fit_preprocessor(pipe, X_train: pd.DataFrame, feature_order: list):
    X_train = validate_and_prepare(X_train, feature_order)
    pipe.fit(X_train)
    return pipe


# ====================================================
# 8. Transform
# ====================================================
def transform_preprocessor(pipe, X: pd.DataFrame, feature_order: list):
    X = validate_and_prepare(X, feature_order)
    X_arr = pipe.transform(X)

    numeric_cols = pipe.named_steps["preprocessor"].transformers_[0][2]
    categorical_cols = pipe.named_steps["preprocessor"].transformers_[1][2]

    feature_names = list(numeric_cols) + list(categorical_cols)
    return pd.DataFrame(X_arr, columns=feature_names)
