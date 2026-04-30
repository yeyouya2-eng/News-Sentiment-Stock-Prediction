# 📰 News Sentiment and Directional Price Forecasting
### A Comparative Study of Technology (XLK) vs Financial (XLF) Sectors

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20LSTM-orange)
![NLP](https://img.shields.io/badge/NLP-Custom%20Classifier-green)
![Course](https://img.shields.io/badge/HKBU-ECON7055-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Author:** YE Yongyi (25402137) | **Course:** ECON 7055, HKBU | **Year:** 2026

---

## 🔍 Research Questions

> Do technology and financial sector ETFs respond differently to news sentiment in terms of stock price predictability?

| Hypothesis | Statement | Result |
|---|---|---|
| **H1** (Sector Divergence) | XLK exhibits higher sentiment-driven predictability than XLF | ✅ **VALIDATED** (t=2.747, p=0.003, n=1,010) |
| **H2** (Sentiment Value) | Sentiment features significantly improve prediction accuracy | ❌ **REJECTED** (McNemar p>0.05 across all models) |

---

## 📊 Key Results

| Model | XLK Acc. | XLF Acc. | XLK Sharpe | XLF Sharpe |
|-------|----------|----------|------------|------------|
| **XGBoost ★** | **54.65%** | 48.71% | **0.47** | -0.29 |
| LSTM | 53.50% | 50.00% | 0.52 | 0.29 |
| Logistic Reg. | 50.99% | 50.59% | -1.10 | -0.53 |
| Random Forest | 51.39% | 46.73% | -0.98 | -0.72 |
| ARIMA(5,1,0) | 48.00% | 56.00% | N/A | N/A |

> All models evaluated under rolling walk-forward backtest with **0.1% transaction costs**.

### 🔑 Core Finding: The Feature Importance–Predictability Paradox

XLF assigns **higher** sentiment importance (41.7% vs XLK's 38.8%) yet achieves **lower** accuracy. Granger causality resolves this:

- **XLK:** Sentiment Granger-causes returns at lags 1–4 days (6 features, p<0.05) → *causal transmission*
- **XLF:** Zero significant Granger relationships → *contemporaneous co-movement only*

---

## 🏗️ Pipeline

```
Step 1   │ Data Collection — NYT Archive API (holdings-based keyword strategy)
Step 2a  │ Sentiment Labelling — Loughran-McDonald Dictionary
Step 2b  │ Train 4 Custom Classifiers: NB / SVM★ / LDA-XGB / Word2Vec-XGB
Step 2c  │ Feature Engineering — 30 features (technical + sentiment + macro)
Step 2d  │ Granger Causality Analysis
Step 3   │ Hyperparameter Tuning — XGBoost + LSTM
Step 4   │ Rolling Walk-Forward Backtest — 5 models
Step 4d  │ Feature Importance Extraction
Step 4e  │ Correlation Heatmap (30 features)
Step 5.1 │ H2 Validation — McNemar's Test (XGBoost + LSTM)
Step 5.3 │ H1 Validation — Daily Paired t-test (n=1,010)
Step 6   │ Real-Time Inference Pipeline
```

---

## 🧠 Methodology Highlights

- **No pre-trained models.** All sentiment classifiers trained from scratch on LM-labelled data.
- **Best classifier:** Dense SVM (TF-IDF → SVD) — 88.85% XLK / 82.47% XLF
- **30 engineered features:** 9 technical + 11 sentiment derivatives + 7 macro + 3 auxiliary
- **Rigorous backtest:** 252-day rolling window, 0.1% transaction cost, zero look-ahead bias
- **Dual statistical tests:** Paired t-test (H1) + McNemar's test (H2)

---

## 📁 Repository Structure

```
├── step1_holdings_based_2021_present.py
├── step2a_create_sentiment_labels.py
├── step2b_train_sentiment_classifier.py
├── step2c_generate_sentiment_features_WITH_MACRO.py
├── step2d_granger_causality_SENTIMENT.py
├── step3_tune_xgboost_FORCE_SAVE.py
├── step3_tune_lstm.py
├── step4_run_rolling_model_FIXED.py
├── step4_run_lstm.py
├── step4b_run_baselines_ROLLING_FIXED.py
├── step4c_run_arima.py
├── step4d_extract_feature_importance.py
├── step4e_heatmap.py
├── step5_1_h2_validation_XGBoost_FIXED.py
├── step5_1b_h2_validation_LSTM.py
├── step5_3_h1_daily_validation.py
├── step6_realtime_inference_IMPROVED.py
└── requirements.txt
```

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
```

Create `config.py`:
```python
NYT_API_KEY = "your_key_here"
# PROXY_URL = "http://127.0.0.1:7890"  # optional, for mainland China
```

Run scripts in order from Step 1 → Step 6. Apply for NYT API key at https://developer.nytimes.com/

---

## 📦 Data Sources

| Source | Content | Period |
|--------|---------|--------|
| NYT Archive API | News headlines (XLK/XLF holdings-based) | Jan 2021 – Feb 2026 |
| Yahoo Finance | XLK, XLF daily OHLCV | Jan 2021 – Feb 2026 |
| Yahoo Finance | VIX, 10Y Treasury, Dollar Index | Jan 2021 – Feb 2026 |

Raw news data not included. Re-run Step 1 to regenerate.

---

## 📚 References

Loughran & McDonald (2011) · Tetlock (2007) · Fishbein & Ajzen (1975) · Granger (1969) · McNemar (1947)

---
*ECON 7055 Final Project — HKBU — 2026*
