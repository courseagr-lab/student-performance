import pandas as pd

from src.data.data_cleaning import clean_data, load_config


def test_load_config_reads_default_yaml():
    config = load_config()

    assert config["data"]["raw_path"] == "data/raw/student_performance_raw.csv"
    assert config["data"]["target"] == "passed"


def test_clean_data_filters_invalid_rows_and_duplicates():
    config = load_config()
    df = pd.DataFrame(
        {
            "student_id": [1, 1, 2, 3, 4],
            "hours_studied": [2, 2, -1, 4, 5],
            "attendance_rate": [80, 80, 90, 101, 95],
            "previous_score": [70, 70, 80, 90, 120],
            "passed": [1, 1, 0, 1, 0],
        }
    )

    cleaned = clean_data(df, config)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "student_id"] == 1
