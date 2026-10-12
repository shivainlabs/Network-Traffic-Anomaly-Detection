- Phase 1 — Data understanding and EDA
- Phase 2 — Statistical analysis
- Phase 3 — Feature engineering and preprocessing
- Phase 4 — Model development
- Phase 5 — Model evaluation and interpretation
- Phase 6 — Streamlit deployment


```text
network-traffic-anomaly-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── best_model.joblib
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```