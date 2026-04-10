"""
STEP 5.3: H1 Validation - Daily Prediction Level Analysis (FIXED)
=================================================================
修复：sign test 的模型准确率改为从 CSV 动态读取，而非硬编码。

读取顺序：
  1. rolling_predictions_XLK/XLF.csv          (step4 XGBoost)
  2. rolling_predictions_Logistic_XLK/XLF.csv  (step4b Logistic)
  3. rolling_predictions_RandomForest_XLK/XLF.csv (step4b RF)
  4. results/rolling_backtest_lstm_XLK/XLF.csv 或 h2_validation_summary_LSTM.csv (step4/5.1b LSTM)

Requires:
  - results/rolling_predictions_XLK.csv  (from step4_run_rolling_model_FIXED.py)
  - results/rolling_predictions_XLF.csv  (from step4_run_rolling_model_FIXED.py)
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
try:
    from scipy.stats import binomtest as _binomtest
    def binom_test(k, n, p, alternative='greater'):
        return _binomtest(k, n=n, p=p, alternative=alternative).pvalue
except ImportError:
    from scipy.stats import binom_test
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔬 H1 VALIDATION: Daily Prediction Level (n≈996)")
print("=" * 70)
print()

# ─── 1. Load rolling predictions (XGBoost, from step4) ────────────────────────

xlk_path = "results/rolling_predictions_XLK.csv"
xlf_path = "results/rolling_predictions_XLF.csv"

if not os.path.exists(xlk_path) or not os.path.exists(xlf_path):
    print("❌ Rolling prediction files not found.")
    print("   Please run step4_run_rolling_model_FIXED.py first.")
    exit(1)

xlk_df = pd.read_csv(xlk_path, parse_dates=['Date'])
xlf_df = pd.read_csv(xlf_path, parse_dates=['Date'])

print(f"Loaded XLK predictions: {len(xlk_df)} days")
print(f"Loaded XLF predictions: {len(xlf_df)} days")

# ─── 2. Compute daily correctness ─────────────────────────────────────────────

xlk_df['correct'] = (xlk_df['Pred'] == xlk_df['Target']).astype(int)
xlf_df['correct'] = (xlf_df['Pred'] == xlf_df['Target']).astype(int)

merged = xlk_df[['Date', 'correct']].merge(
    xlf_df[['Date', 'correct']], on='Date', suffixes=('_xlk', '_xlf')
)

print(f"Overlapping trading days: {len(merged)}")
print()

# ─── 3. Descriptive stats ─────────────────────────────────────────────────────

xlk_acc = merged['correct_xlk'].mean()
xlf_acc = merged['correct_xlf'].mean()
diff = xlk_acc - xlf_acc

xlk_better_days = (merged['correct_xlk'] > merged['correct_xlf']).sum()
xlf_better_days = (merged['correct_xlk'] < merged['correct_xlf']).sum()
tied_days       = (merged['correct_xlk'] == merged['correct_xlf']).sum()

print("=" * 70)
print("📊 DESCRIPTIVE STATISTICS")
print("=" * 70)
print(f"  XLK daily accuracy:  {xlk_acc:.2%}")
print(f"  XLF daily accuracy:  {xlf_acc:.2%}")
print(f"  Mean difference:     {diff:+.2%} (XLK - XLF)")
print()
print(f"  Days XLK correct & XLF wrong: {xlk_better_days}")
print(f"  Days XLF correct & XLK wrong: {xlf_better_days}")
print(f"  Days both same result:         {tied_days}")
print()

# ─── 4. Paired t-test ──────────────────────────────────────────────────────────

print("=" * 70)
print(f"🔬 STATISTICAL TEST 1: Paired t-test (daily level, n={len(merged)})")
print("=" * 70)

t_stat, p_two = ttest_rel(merged['correct_xlk'], merged['correct_xlf'])
p_one = p_two / 2

print(f"  t-statistic:         {t_stat:.4f}")
print(f"  P-value (two-tail):  {p_two:.4f}")
print(f"  P-value (one-tail):  {p_one:.4f}")
print()

if p_one < 0.01:
    sig_label = "✅ HIGHLY SIGNIFICANT (p < 0.01)"
elif p_one < 0.05:
    sig_label = "✅ SIGNIFICANT (p < 0.05)"
elif p_one < 0.10:
    sig_label = "⚠️  MARGINALLY SIGNIFICANT (p < 0.10)"
else:
    sig_label = "❌ NOT SIGNIFICANT (p >= 0.10)"

print(f"  Result: {sig_label}")
print()

# ─── 5. Wilcoxon signed-rank test ─────────────────────────────────────────────

print("=" * 70)
print("🔬 STATISTICAL TEST 2: Wilcoxon Signed-Rank Test")
print("=" * 70)

diffs = merged['correct_xlk'] - merged['correct_xlf']
non_tied = diffs[diffs != 0]

if len(non_tied) > 0:
    w_stat, w_p_two = wilcoxon(non_tied, alternative='two-sided')
    w_p_one = w_p_two / 2

    print(f"  Non-tied pairs:      {len(non_tied)}")
    print(f"  W-statistic:         {w_stat:.1f}")
    print(f"  P-value (one-tail):  {w_p_one:.4f}")
    print()

    if w_p_one < 0.05:
        print("  ✅ SIGNIFICANT (p < 0.05)")
    elif w_p_one < 0.10:
        print("  ⚠️  MARGINALLY SIGNIFICANT (p < 0.10)")
    else:
        print("  ❌ NOT SIGNIFICANT")
else:
    print("  ⚠️  All differences tied — cannot compute Wilcoxon test")
    w_p_one = 1.0
print()

# ─── 6. Sign Test — DYNAMIC: load model accuracies from CSVs ──────────────────

print("=" * 70)
print("🔬 STATISTICAL TEST 3: Sign Test (Sentiment Models Only, n=4)")
print("=" * 70)
print()
print("  Loading model accuracies dynamically from results CSVs...")
print()

def get_model_acc(pred_csv_xlk, pred_csv_xlf):
    """Compute accuracy from rolling prediction CSVs."""
    try:
        df_xlk = pd.read_csv(pred_csv_xlk)
        df_xlf = pd.read_csv(pred_csv_xlf)
        acc_xlk = (df_xlk['Actual'] == df_xlk['Prediction']).mean() if 'Actual' in df_xlk.columns else (df_xlk['Target'] == df_xlk['Pred']).mean()
        acc_xlf = (df_xlf['Actual'] == df_xlf['Prediction']).mean() if 'Actual' in df_xlf.columns else (df_xlf['Target'] == df_xlf['Pred']).mean()
        return acc_xlk, acc_xlf
    except Exception as e:
        return None, None

def get_acc_from_xgb_rolling():
    """XGBoost: from step4 rolling predictions."""
    xlk = (xlk_df['Target'] == xlk_df['Pred']).mean()
    xlf = (xlf_df['Target'] == xlf_df['Pred']).mean()
    return xlk, xlf

def get_acc_from_lstm_summary():
    """LSTM: try h2_validation_summary_LSTM.csv first, then rolling CSV."""
    # Try h2 summary (has rolling accuracy stored under different name)
    lstm_summary = "results/h2_validation_summary_LSTM.csv"
    lstm_rolling_xlk = "results/rolling_predictions_LSTM_XLK.csv"
    lstm_rolling_xlf = "results/rolling_predictions_LSTM_XLF.csv"

    # Prefer rolling predictions if they exist
    if os.path.exists(lstm_rolling_xlk) and os.path.exists(lstm_rolling_xlf):
        return get_model_acc(lstm_rolling_xlk, lstm_rolling_xlf)

    # Fallback: use Full_Accuracy from LSTM h2 summary (80/20 split, not rolling)
    if os.path.exists(lstm_summary):
        df = pd.read_csv(lstm_summary)
        xlk_row = df[df['Ticker'] == 'XLK']
        xlf_row = df[df['Ticker'] == 'XLF']
        if len(xlk_row) > 0 and len(xlf_row) > 0:
            return xlk_row['Full_Accuracy'].values[0], xlf_row['Full_Accuracy'].values[0]

    return None, None

# Collect model accuracies
sentiment_models = {}

# XGBoost (always available since step4 CSVs are loaded above)
xgb_xlk, xgb_xlf = get_acc_from_xgb_rolling()
sentiment_models['XGBoost'] = (xgb_xlk, xgb_xlf)

# Logistic Regression
lr_xlk, lr_xlf = get_model_acc(
    "results/rolling_predictions_Logistic_XLK.csv",
    "results/rolling_predictions_Logistic_XLF.csv"
)
if lr_xlk is not None:
    sentiment_models['Logistic Reg'] = (lr_xlk, lr_xlf)
else:
    print("  ⚠️  Logistic results not found — run step4b first.")

# Random Forest
rf_xlk, rf_xlf = get_model_acc(
    "results/rolling_predictions_RandomForest_XLK.csv",
    "results/rolling_predictions_RandomForest_XLF.csv"
)
if rf_xlk is not None:
    sentiment_models['Random Forest'] = (rf_xlk, rf_xlf)
else:
    print("  ⚠️  RandomForest results not found — run step4b first.")

# LSTM
lstm_xlk, lstm_xlf = get_acc_from_lstm_summary()
if lstm_xlk is not None:
    sentiment_models['LSTM'] = (lstm_xlk, lstm_xlf)
else:
    print("  ⚠️  LSTM results not found — run step4_run_lstm.py or step5_1b first.")

print()
print(f"  {'Model':<18} {'XLK':>8} {'XLF':>8} {'XLK>XLF':>10}")
print(f"  {'-'*50}")
xlk_wins = 0
for model, (xlk_a, xlf_a) in sentiment_models.items():
    win = "✅ YES" if xlk_a > xlf_a else "❌ NO"
    if xlk_a > xlf_a:
        xlk_wins += 1
    print(f"  {model:<18} {xlk_a:>7.2%} {xlf_a:>8.2%} {win:>10}")

n_models = len(sentiment_models)
print()
print(f"  XLK outperforms XLF: {xlk_wins}/{n_models} sentiment models")
print()

if n_models > 0:
    sign_p = binom_test(xlk_wins, n=n_models, p=0.5, alternative='greater')
    print(f"  Binomial sign test p-value (one-tail): {sign_p:.4f}")
    if sign_p < 0.05:
        print("  ✅ SIGNIFICANT")
    elif sign_p <= 0.0625:
        print("  ⚠️  MARGINALLY SIGNIFICANT (p=0.0625 is minimum achievable for n=4)")
    else:
        print("  ❌ NOT SIGNIFICANT")
    print()
    print(f"  Note: With n={n_models}, minimum achievable p-value is {1/2**n_models:.4f}.")
    print("  Statistical significance at α=0.05 is mathematically impossible with")
    print("  only 4 data points. This is a limitation of model-level comparison.")
else:
    sign_p = 1.0
    print("  ⚠️  No model results found for sign test.")
print()

# ─── 7. Summary ────────────────────────────────────────────────────────────────

print("=" * 70)
print("🎯 H1 VALIDATION SUMMARY")
print("=" * 70)
print()

summary_rows = [
    ["Daily paired t-test", f"n={len(merged)}", f"p={p_one:.4f}",
     "✅ SIG" if p_one < 0.05 else ("⚠️ MARGINAL" if p_one < 0.10 else "❌ n.s.")],
    ["Wilcoxon signed-rank", f"n={len(non_tied) if len(non_tied)>0 else 'N/A'}",
     f"p={w_p_one:.4f}" if len(non_tied) > 0 else "N/A",
     "✅ SIG" if (len(non_tied)>0 and w_p_one<0.05) else "❌ n.s."],
    [f"Sign test ({n_models} sent. models)", f"n={n_models}", f"p={sign_p:.4f}",
     "⚠️ MARGINAL" if sign_p <= 0.0625 else "❌ n.s."],
]

print(f"  {'Test':<32} {'Sample':<10} {'P-value':<12} {'Result'}")
print(f"  {'-'*68}")
for row in summary_rows:
    print(f"  {row[0]:<32} {row[1]:<10} {row[2]:<12} {row[3]}")
print()

if p_one < 0.05:
    overall = "✅ H1 VALIDATED (statistically significant at α=0.05)"
    interp = (f"The daily prediction accuracy of XLK is significantly higher than XLF "
              f"(Δ={diff:+.2%}, p={p_one:.4f}, n={len(merged)}). This confirms that technology stocks "
              "exhibit higher sentiment-driven predictability than financial stocks.")
elif p_one < 0.10:
    overall = "⚠️  H1 PARTIALLY SUPPORTED (marginal significance, α=0.10)"
    interp = (f"XLK shows higher prediction accuracy on the majority of trading days "
              f"(Δ={diff:+.2%}, p={p_one:.4f}). The effect is consistent but does not "
              "reach the conventional 5% significance threshold.")
else:
    overall = "❌ H1 NOT VALIDATED at conventional significance levels"
    interp = (f"Despite a consistent directional trend (Δ={diff:+.2%}, "
              f"{xlk_wins}/{n_models} sentiment models favour XLK), the difference does not "
              "reach statistical significance.")

print(f"  Overall: {overall}")
print()
print("  Interpretation:")
print(f"  {interp}")
print()

# ─── 8. Save ──────────────────────────────────────────────────────────────────

if not os.path.exists('results'):
    os.makedirs('results')

result_dict = {
    'n_days': len(merged),
    'xlk_daily_accuracy': xlk_acc,
    'xlf_daily_accuracy': xlf_acc,
    'mean_difference_pp': diff * 100,
    'paired_ttest_statistic': t_stat,
    'paired_ttest_p_two_tail': p_two,
    'paired_ttest_p_one_tail': p_one,
    'wilcoxon_p_one_tail': w_p_one if len(non_tied) > 0 else None,
    'sign_test_p': sign_p,
    'n_models_sign_test': n_models,
    'xlk_wins_sentiment_models': f"{xlk_wins}/{n_models}",
    'overall_status': overall
}

# Also save model-level accuracies for report table
model_rows = [{'Model': m, 'XLK_Acc': v[0], 'XLF_Acc': v[1], 'XLK_wins': v[0] > v[1]}
              for m, v in sentiment_models.items()]

pd.DataFrame([result_dict]).to_csv('results/h1_daily_validation_summary.csv', index=False)
pd.DataFrame(model_rows).to_csv('results/h1_model_accuracy_comparison.csv', index=False)

print(f"  ✅ Saved: results/h1_daily_validation_summary.csv")
print(f"  ✅ Saved: results/h1_model_accuracy_comparison.csv")
print()
print("=" * 70)
print("✅ H1 DAILY VALIDATION COMPLETE")
print("=" * 70)