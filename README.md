student-performance/
├── data/
│   ├── raw/
│   │   └── student_performance_raw.csv    # jangan diubah
│   ├── interim/                            # output cleaning
│   ├── processed/                          # output feature engineering, siap modeling
│   └── external/                           # tidak dipakai project ini, disediakan untuk konsistensi
├── notebooks/
│   ├── 01-data-cleaning.ipynb              # ← BARU
│   ├── 02-eda.ipynb
│   ├── 03-feature-engineering.ipynb
│   └── 04-modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py                 # ← BARU: fungsi cleaning reusable
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py
├── models/
├── reports/
│   └── figures/
├── tests/
│   └── test_make_dataset.py
├── config/
│   └── config.yaml
├── requirements.txt
├── .env
├── .gitignore
└── README.md
