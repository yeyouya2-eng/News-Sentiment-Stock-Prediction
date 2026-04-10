# News Sentiment and Directional Price Forecasting
### A Comparative Study of Technology (XLK) vs Financial (XLF) Sectors
**ECON 7055 | HKBU | YE Yongyi (25402137)**

## Project Overview
This project investigates whether NYT news sentiment improves directional stock price prediction, and whether the effect differs between technology and financial sector ETFs.

## Key Results
| Model | XLK Acc. | XLF Acc. | XLK Sharpe |
|-------|----------|----------|------------|
| XGBoost ★ | 54.65% | 48.71% | 0.47 |
| LSTM | 53.50% | 50.00% | 0.52 |
| Logistic Reg. | 50.99% | 50.59% | -1.10 |
| Random Forest | 51.39% | 46.73% | -0.98 |
| ARIMA(5,1,0) | 48.00% | 56.00% | N/A |

**H1 VALIDATED:** XLK shows significantly higher sentiment-driven predictability (t=2.747, p=0.003, n=1,010)  
**H2 REJECTED:** Sentiment features do not significantly improve accuracy (McNemar p>0.05 across all models)

## Pipeline
```
Step 1: Data Collection (NYT Archive API, holdings-based)
Step 2a: Sentiment Labelling (Loughran-McDonald Dictionary)
Step 2b: Train Custom Classifiers (4 pipelines: NB/SVM/LDA-XGB/W2V-XGB)
Step 2c: Feature Engineering (30 features: technical + sentiment + macro)
Step 2d: Granger Causality Analysis
Step 3: Hyperparameter Tuning (XGBoost + LSTM)
Step 4: Rolling Walk-Forward Backtest
Step 5: Hypothesis Validation (H1 paired t-test, H2 McNemar)
```

## How to Run
```bash
pip install -r requirements.txt
# Add your NYT API key to config.py
python step1_holdings_based_2021_present.py
python step2a_create_sentiment_labels.py
# ... follow step order
```

## Data
News data collected from NYT Archive API (2021–2026). Price data from Yahoo Finance.  
Raw data not included due to size. Apply for NYT API key at: https://developer.nytimes.com/
