from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path="config/config.yaml"):
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(config):
    return pd.read_csv(project_path(config["data"]["raw_path"]))


def clean_categorical(df):
    df = df.copy()

    if "parental_education" in df.columns:
        parental_education_map = {
            "bachelor": "Bachelor",
            "high school": "High School",
            "master": "Master",
            "none": np.nan,
        }
        df["parental_education"] = (
            df["parental_education"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(parental_education_map)
        )

    if "extracurricular" in df.columns:
        extracurricular_map = {
            "yes": "Yes",
            "no": "No",
            "none": np.nan,
        }
        df["extracurricular"] = (
            df["extracurricular"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(extracurricular_map)
        )

    if "gender" in df.columns:
        gender_map = {
            "m": "Male",
            "male": "Male",
            "f": "Female",
            "female": "Female",
            "none": np.nan,
        }
        df["gender"] = (
            df["gender"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(gender_map)
        )

    return df


def clean_numeric(df, config):
    df = df.copy()
    c = config["cleaning"]

    # WAJIB: hapus "%" dulu, baru convert ke numeric
    df["attendance_rate"] = df["attendance_rate"].astype(str).str.replace("%", "", regex=False)
    df["attendance_rate"] = pd.to_numeric(df["attendance_rate"], errors="coerce")

    df.loc[df["attendance_rate"] > c["attendance_rate_max"], "attendance_rate"] = np.nan
    df.loc[df["attendance_rate"] < c["attendance_rate_min"], "attendance_rate"] = np.nan
    df.loc[df["hours_studied"] < c["hours_studied_min"], "hours_studied"] = np.nan
    df.loc[df["previous_score"] > c["previous_score_max"], "previous_score"] = np.nan
    return df


def handle_missing(df):
    df = df.copy()
    numeric_col = ["hours_studied", "attendance_rate", "sleep_hours", "previous_score"]
    for col in numeric_col:
        df[col] = df[col].fillna(df[col].median())
    df["parental_education"] = df["parental_education"].fillna(df["parental_education"].mode()[0])
    return df

def remove_duplicates(df):
    df = df.copy()
    subset_col = [c for c in df.columns if c != "student_id"]
    df = df.drop_duplicates(subset=subset_col, keep="first")
    return df

def run_cleaning(config):
    df = load_raw(config)
    df = clean_categorical(df)
    df = clean_numeric(df, config)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df.to_csv(project_path(config["data"]["interim_path"]), index=False)
    return df

if __name__ == "__main__":
    config = load_config()
    df_clean = run_cleaning(config)
    print(f"Cleaned shape: {df_clean.shape}") 
    print(f"Missing values:\n{df_clean.isnull().sum()}")
