# Network Traffic Anomaly Detection

A hands-on **machine learning learning project**: can a model tell malicious network traffic from normal traffic on a small, imbalanced dataset? The project walks through the full workflow (EDA, feature engineering, model training, honest evaluation) and presents the results in an interactive Streamlit app.

> **This is an educational project, not a security tool.** The models perform close to random on this dataset, and the project is built around understanding *why*, not hiding it.

**Live demo:** https://network-traffic-anomaly.streamlit.app *(update this link to match your deployed app URL)*

---

## Key findings

| Question | What the experiments showed |
|---|---|
| Is accuracy a good metric here? | No. 90% of records are Normal, so predicting "Normal" for everything already scores ~90% while catching 0 attacks. |
| Do the models beat a do-nothing baseline? | Not in cross-validation: ROC-AUC 0.45-0.49 (random = 0.5), Average Precision ~0.12-0.13 (random = 0.10). |
| Is that distinguishable from pure luck? | No. Real scores fall inside the distribution produced by shuffled labels. |
| Does any single feature carry signal? | None clearly passes the "best of 14 features by luck" threshold. |
| Is the pipeline broken? | No. A positive control (fake signal injected into a feature) is detected, and AUC rises as the signal grows. |
| What about the test set? | Test ROC-AUC of 0.62-0.67 looks slightly better, but with only 20 malicious test records the confidence intervals are wide, and it conflicts with cross-validation. Treated as unresolved, not as a success. |

**Main lesson:** a model cannot find signal that isn't in the data. Trying more models would mostly add more chances of a lucky-looking number.

---

## Project structure

```text
Network-Traffic-Anomaly-Detection/
│
├── app/
│   └── streamlit_app.py            # Streamlit demo (reads saved results only)
│
├── artifacts/                      # Results saved by notebook 04, read by the app
│   ├── cv_model_comparison.csv
│   ├── feature_auc.csv
│   └── oof_probabilities.csv
│
├── data/
│   ├── raw/
│   │   └── network_traffic_dataset.csv
│   └── processed/
│       └── network_traffic_cleaned.csv
│
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_experiments.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_understanding_eda` | Explore the dataset, class imbalance, distributions, and feature-label relationships. |
| `02_feature_engineering` | Transformations, correlations, categorical encoding, and the cleaned dataset. |
| `03_model_training` | First modelling pass: Logistic Regression, Random Forest, XGBoost; cross-validation, probabilities, thresholds. Kept as the original record of the experiments. |
| `04_model_experiments` | Diagnostic experiments added afterwards: fair model comparison, shuffled-label test, per-feature signal, positive control, and test-set uncertainty. Also saves the files in `artifacts/`. |

## Dataset

The [Network Traffic Anomaly Detection Dataset](https://www.kaggle.com/datasets/ziya07/network-traffic-anomaly-detection-dataset) by [ziya07](https://www.kaggle.com/ziya07) on Kaggle: a generated (simulated) network-traffic dataset for anomaly detection in embedded and IoT-style environments. After encoding it has **1,000 records and 16 features**, labelled `0` (normal) or `1` (malicious). Features include packet size, inter-arrival time, protocol type, source/destination IP, destination port, TCP flags, packet count per 5-second window, and wavelet-based spectral entropy and frequency-band energy.

- Train: 800 records (720 Normal, 80 Malicious)
- Test: 200 records (180 Normal, 20 Malicious)
- Split: stratified, `random_state=42`

## Streamlit app

Four tabs, all read-only:

1. **Dataset:** class balance and per-feature distributions by class.
2. **Model comparison:** cross-validated ROC-AUC and Average Precision with random baselines.
3. **Threshold explorer:** move the decision threshold and watch the confusion matrix, recall, precision, and ROC curve change.
4. **Feature signal:** per-feature AUC against the "luck" threshold.

The app never trains a model; it only displays results saved by notebook 04.

## Run locally

```bash
git clone https://github.com/shivainlabs/Network-Traffic-Anomaly-Detection.git
cd Network-Traffic-Anomaly-Detection
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Run the command from the repository root. To regenerate the files in `artifacts/`, run `notebooks/04_model_experiments.ipynb` (Colab or local) and copy the `artifacts/` folder into the repo.

## Limitations

- Only 1,000 records and 100 malicious examples; test-set scores are very noisy (20 malicious records).
- The features show little to no measurable relationship with the label, so model quality is limited by the data, not the algorithm.
- The dataset is simulated and does not represent real network attacks.
- Not suitable for real security use.

## Roadmap

- [x] Phase 1 - Data understanding and EDA
- [x] Phase 2 - Feature engineering and preprocessing
- [x] Phase 3 - Model training
- [x] Phase 4 - Diagnostic experiments and honest evaluation
- [x] Phase 5 - Streamlit demo
- [ ] Repeated random train/test splits to resolve the test-vs-cross-validation gap
- [ ] "Demo prediction" tab using a saved model
- [ ] Small fixes in notebook 03 (local data path, `scale_pos_weight` from training labels only, minority-class metrics)

## Acknowledgements

- **Dataset:** [Network Traffic Anomaly Detection Dataset](https://www.kaggle.com/datasets/ziya07/network-traffic-anomaly-detection-dataset), created and shared on Kaggle by [ziya07](https://www.kaggle.com/ziya07). Thank you for making it available. The dataset belongs to its author; please check its Kaggle page for the license and terms of use before reusing it.
- This repository contains my own analysis, notebooks, and demo app, and is not affiliated with or endorsed by the dataset author.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Streamlit.