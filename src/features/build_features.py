import pandas as pd

from src.data.data_cleaning import load_config, project_path

def load_interim(config):
    return pd.read_csv(project_path(config["data"]["interim_path"]))

def encode_categorical(df):
    df = df.copy()

    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['extracurricular'] = df['extracurricular'].map({'No': 0, 'Yes': 1})

    # one-hot untuk parental_education(>2 categories)

    df = pd.get_dummies(df, columns=['parental_education'], prefix='edu', dtype=int)

    return df

def add_derived_features(df):
    df = df.copy()
    # fitur turunan sederhana: efisiensi belajar (skor sebelumnya per jam belajar)
    # tambahkan epsilon kecil untuk hindari pembagian dengan nol
    df["study_efficiency"] = df["previous_score"] / (df["hours_studied"] + 0.1)
    return df

def build_features(config):
    df = load_interim(config)
    df = encode_categorical(df)
    df = add_derived_features(df)
    df.to_csv(project_path(config["data"]["processed_path"]), index=False)
    return df


if __name__ == "__main__":
    config = load_config()
    df_processed = build_features(config)
    print(f"Processed shape: {df_processed.shape}")
    print(df_processed.columns.tolist())
    print(df_processed.head())