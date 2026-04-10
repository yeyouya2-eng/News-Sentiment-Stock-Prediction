"""
STEP 5.1: H2 Hypothesis Validation (XGBoost)
=============================================
验证Sentiment Features的增量价值

H2: Sentiment features significantly improve prediction accuracy

方法：
• Baseline Model: Technical indicators only (NO sentiment)
• Full Model: Technical + Sentiment features
• Statistical Test: McNemar's test (p < 0.05)
• Evaluation: Rolling window (252-day training)

数据：features_sentiment_focused_XX.csv (23 features)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

print("="*70)
print("🔬 H2 VALIDATION: XGBoost (Sentiment Value)")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']
TRAIN_WINDOW = 252
REFIT_FREQ = 20

def load_data(ticker):
    """加载当前数据结构"""
    data_path = f"data/features_sentiment_macro_{ticker}.csv"
    if not os.path.exists(data_path):
        data_path = f"data/features_sentiment_focused_{ticker}.csv"
    if not os.path.exists(data_path):
        print(f"   ❌ Data not found. Tried:")
        print(f"      data/features_sentiment_macro_{ticker}.csv")
        print(f"      data/features_sentiment_focused_{ticker}.csv")
        print(f"   💡 Make sure you have run step2c to generate features")
        return None
    print(f"   📂 Loading: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    print(f"   ✅ Loaded {len(df)} days of data")
    return df

def load_best_params(ticker):
    """加载已调优的XGBoost参数"""
    param_file = f"results/best_params_xgb_{ticker}.json"
    
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            params = json.load(f)
        print(f"   ✅ Loaded tuned parameters")
        return params
    
    # 默认参数
    print(f"   ⚠️  Using default parameters (no tuned params found)")
    return {
        'max_depth': 5,
        'learning_rate': 0.03,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }

def define_feature_sets(df):
    """
    定义Baseline和Full特征集
    
    当前数据结构（23 features）：
    - Technical (9): MA5, MA10, MA20, MA50, RSI, MACD, BB_Upper, BB_Lower, Log_Return
    - Sentiment (11): Sentiment_FinBERT, Sentiment_EMA, Sentiment_MA3, etc.
    - Other (3): News_Count, Volume_Ratio
    """
    
    # 排除target和price相关列
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
               'Target', 'Next_Open', 'Next_Close', 'Tradable_Return', 'Log_Return']
    
    all_features = [c for c in df.columns if c not in exclude]
    
    # Baseline: 只用technical和other，NO sentiment
    baseline_features = [f for f in all_features if 'Sentiment' not in f and 'sentiment' not in f]
    
    # Full: 所有features（包括sentiment）
    full_features = all_features
    
    # Sentiment only
    sentiment_features = [f for f in all_features if 'Sentiment' in f or 'sentiment' in f]
    
    return baseline_features, full_features, sentiment_features

def run_h2_validation(ticker):
    """H2验证主函数"""
    
    print(f"\n{'='*70}")
    print(f"🎯 H2 VALIDATION: {ticker}")
    print(f"{'='*70}")
    
    # 1. 加载数据
    df = load_data(ticker)
    if df is None:
        return None
    
    # 2. 定义特征集
    baseline_features, full_features, sentiment_features = define_feature_sets(df)
    
    print(f"\n   📊 Feature Sets:")
    print(f"      • Baseline (No Sentiment): {len(baseline_features)} features")
    if len(baseline_features) > 0:
        print(f"        Examples: {baseline_features[:3]}")
    
    print(f"      • Full (With Sentiment): {len(full_features)} features")
    print(f"      • Sentiment Features: {len(sentiment_features)} features")
    if len(sentiment_features) > 0:
        print(f"        Examples: {sentiment_features[:3]}")
    
    # 验证特征集合理性
    if len(sentiment_features) == 0:
        print(f"\n   ❌ ERROR: No sentiment features found!")
        print(f"      Available columns: {df.columns.tolist()}")
        return None
    
    if len(baseline_features) == 0:
        print(f"\n   ❌ ERROR: No baseline features found!")
        return None
    
    # 3. 加载参数
    params = load_best_params(ticker)
    
    # 4. Rolling Window Comparison
    print(f"\n   🏃 Running Rolling Window Comparison...")
    print(f"      Training Window: {TRAIN_WINDOW} days")
    print(f"      Refit Frequency: {REFIT_FREQ} days")
    
    baseline_preds = []
    full_preds = []
    actuals = []
    dates = []
    
    last_refit_baseline = 0
    last_refit_full = 0
    model_baseline = None
    model_full = None
    
    for i in tqdm(range(TRAIN_WINDOW, len(df)), 
                  desc=f"   Testing {ticker}",
                  unit="day"):
        
        # 决定是否需要refit
        need_refit = (i - last_refit_baseline >= REFIT_FREQ) or (i == TRAIN_WINDOW)
        
        if need_refit:
            # Prepare training data
            train_data = df.iloc[i-TRAIN_WINDOW:i]
            y_train = train_data['Target']
            
            # Train baseline model
            X_train_baseline = train_data[baseline_features]
            model_baseline = xgb.XGBClassifier(**params)
            model_baseline.fit(X_train_baseline, y_train)
            last_refit_baseline = i
            
            # Train full model
            X_train_full = train_data[full_features]
            model_full = xgb.XGBClassifier(**params)
            model_full.fit(X_train_full, y_train)
            last_refit_full = i
        
        # Predict
        test_row = df.iloc[i:i+1]
        y_test = test_row['Target'].values[0]
        
        X_test_baseline = test_row[baseline_features]
        X_test_full = test_row[full_features]
        
        pred_baseline = model_baseline.predict(X_test_baseline)[0]
        pred_full = model_full.predict(X_test_full)[0]
        
        baseline_preds.append(pred_baseline)
        full_preds.append(pred_full)
        actuals.append(y_test)
        dates.append(test_row.index[0])
    
    print(f"\n   ✅ Predictions Complete! ({len(actuals)} days)")
    
    # 5. 计算性能指标
    baseline_acc = accuracy_score(actuals, baseline_preds)
    full_acc = accuracy_score(actuals, full_preds)
    
    baseline_prec = precision_score(actuals, baseline_preds, zero_division=0)
    full_prec = precision_score(actuals, full_preds, zero_division=0)
    
    improvement = (full_acc - baseline_acc) * 100
    
    print(f"\n   📊 PERFORMANCE COMPARISON:")
    print(f"   {'='*60}")
    print(f"\n   Baseline Model (No Sentiment):")
    print(f"      • Accuracy:  {baseline_acc:.2%}")
    print(f"      • Precision: {baseline_prec:.2%}")
    
    print(f"\n   Full Model (With Sentiment):")
    print(f"      • Accuracy:  {full_acc:.2%}")
    print(f"      • Precision: {full_prec:.2%}")
    
    print(f"\n   📈 Improvement:")
    print(f"      • Accuracy Gain: {improvement:+.2f} percentage points")
    
    if improvement > 0:
        print(f"      ✅ Sentiment features IMPROVE performance!")
    elif improvement < 0:
        print(f"      ⚠️  Sentiment features HURT performance")
    else:
        print(f"      ➖ No difference")
    
    # 6. 统计显著性检验（McNemar's Test）
    print(f"\n   🔬 Statistical Significance Test (McNemar):")
    
    # Contingency table
    both_correct = sum((b == a) and (f == a) for b, f, a in zip(baseline_preds, full_preds, actuals))
    baseline_only = sum((b == a) and (f != a) for b, f, a in zip(baseline_preds, full_preds, actuals))
    full_only = sum((b != a) and (f == a) for b, f, a in zip(baseline_preds, full_preds, actuals))
    both_wrong = sum((b != a) and (f != a) for b, f, a in zip(baseline_preds, full_preds, actuals))
    
    contingency_table = np.array([
        [both_correct, baseline_only],
        [full_only, both_wrong]
    ])
    
    print(f"\n      Contingency Table:")
    print(f"                          Full Correct   Full Wrong")
    print(f"      Baseline Correct    {both_correct:>12}  {baseline_only:>11}")
    print(f"      Baseline Wrong      {full_only:>12}  {both_wrong:>11}")
    
    # McNemar's test
    if baseline_only + full_only > 0:
        result = mcnemar(contingency_table, exact=False, correction=True)
        p_value = result.pvalue
        
        print(f"\n      McNemar's Test:")
        print(f"      • Chi-square: {result.statistic:.4f}")
        print(f"      • P-value: {p_value:.4f}")
        
        # 判断H2状态
        if p_value < 0.05:
            if improvement > 0:
                print(f"      ✅ SIGNIFICANT (p < 0.05)")
                print(f"      → Sentiment SIGNIFICANTLY IMPROVES performance")
                h2_status = "VALIDATED"
            else:
                print(f"      ⚠️  SIGNIFICANT (p < 0.05)")
                print(f"      → Sentiment SIGNIFICANTLY HURTS performance")
                h2_status = "VALIDATED (Negative)"
        elif p_value < 0.10:
            print(f"      ⚠️  MARGINALLY SIGNIFICANT (0.05 < p < 0.10)")
            h2_status = "PARTIAL"
        else:
            print(f"      ❌ NOT SIGNIFICANT (p >= 0.10)")
            print(f"      → Improvement is NOT statistically significant")
            h2_status = "REJECTED"
    else:
        print(f"      ⚠️  Cannot perform McNemar's test (no disagreements)")
        p_value = 1.0
        h2_status = "INCONCLUSIVE"
    
    # 7. 保存结果
    print(f"\n   💾 Saving Results...")
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 保存predictions
    comparison_df = pd.DataFrame({
        'Date': dates,
        'Actual': actuals,
        'Baseline_Pred': baseline_preds,
        'Full_Pred': full_preds
    })
    comparison_df.to_csv(f'results/h2_baseline_comparison_{ticker}.csv', index=False)
    print(f"      ✅ Saved: results/h2_baseline_comparison_{ticker}.csv")
    
    # Summary
    summary = {
        'Ticker': ticker,
        'Model': 'XGBoost',
        'Baseline_Features': len(baseline_features),
        'Full_Features': len(full_features),
        'Sentiment_Features': len(sentiment_features),
        'Baseline_Accuracy': baseline_acc,
        'Full_Accuracy': full_acc,
        'Accuracy_Improvement': improvement,
        'McNemar_P_Value': p_value,
        'H2_Status': h2_status
    }
    
    # 可视化
    create_comparison_plot(ticker, baseline_acc, full_acc, improvement, p_value, h2_status)
    
    print(f"\n{'='*70}\n")
    
    return summary

def create_comparison_plot(ticker, baseline_acc, full_acc, improvement, p_value, h2_status):
    """创建对比可视化"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1: 准确率对比
    models = ['Baseline\n(No Sentiment)', 'Full Model\n(With Sentiment)']
    accuracies = [baseline_acc * 100, full_acc * 100]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = axes[0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].axhline(50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{ticker} - XGBoost Model Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([45, max(accuracies) * 1.1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{acc:.2f}%',
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 图2: 统计显著性
    axes[1].text(0.5, 0.7, f'Accuracy Improvement:', 
                ha='center', fontsize=14, fontweight='bold', transform=axes[1].transAxes)
    axes[1].text(0.5, 0.55, f'{improvement:+.2f} percentage points', 
                ha='center', fontsize=16, color='green' if improvement > 0 else 'red',
                fontweight='bold', transform=axes[1].transAxes)
    
    axes[1].text(0.5, 0.35, f'McNemar Test:', 
                ha='center', fontsize=12, fontweight='bold', transform=axes[1].transAxes)
    axes[1].text(0.5, 0.25, f'P-value: {p_value:.4f}', 
                ha='center', fontsize=12, transform=axes[1].transAxes)
    
    # H2状态
    if h2_status == "VALIDATED":
        status_text = "✅ H2 VALIDATED"
        status_color = 'green'
    elif h2_status == "PARTIAL":
        status_text = "⚠️ H2 PARTIALLY SUPPORTED"
        status_color = 'orange'
    else:
        status_text = "❌ H2 NOT VALIDATED"
        status_color = 'red'
    
    axes[1].text(0.5, 0.1, status_text, 
                ha='center', fontsize=14, fontweight='bold', color=status_color,
                transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=status_color, linewidth=2))
    
    axes[1].axis('off')
    
    plt.tight_layout()
    out_path = f'results/h2_comparison_{ticker}.png'
    import os as _os
    if _os.path.exists(out_path):
        _os.remove(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Saved: results/h2_comparison_{ticker}.png")

def main():
    """主函数"""
    
    print("📌 H2 HYPOTHESIS VALIDATION (XGBoost)\n")
    print("H2: Sentiment features significantly improve prediction accuracy\n")
    print("Method: Baseline (No Sentiment) vs Full (With Sentiment)")
    print("Statistical Test: McNemar's test (p < 0.05)\n")
    
    all_summaries = []
    
    for ticker in TICKERS:
        try:
            summary = run_h2_validation(ticker)
            if summary:
                all_summaries.append(summary)
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    if all_summaries:
        print("="*70)
        print("📊 FINAL H2 VALIDATION SUMMARY")
        print("="*70 + "\n")
        
        summary_df = pd.DataFrame(all_summaries)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Ticker']}:")
            print(f"   • Baseline Acc: {row['Baseline_Accuracy']:.2%}")
            print(f"   • Full Acc: {row['Full_Accuracy']:.2%}")
            print(f"   • Improvement: {row['Accuracy_Improvement']:+.2f} pp")
            print(f"   • P-value: {row['McNemar_P_Value']:.4f}")
            print(f"   • H2 Status: {row['H2_Status']}")
            print()
        
        # 保存汇总
        summary_df.to_csv('results/h2_validation_summary_XGBoost.csv', index=False)
        print(f"✅ Saved: results/h2_validation_summary_XGBoost.csv")
        
        # 判断H2整体状态
        validated_positive = (summary_df['H2_Status'] == 'VALIDATED').sum()
        
        print(f"\n🎯 H2 HYPOTHESIS OVERALL STATUS:")
        if validated_positive == len(TICKERS):
            print(f"   ✅ FULLY VALIDATED across all sectors")
        elif validated_positive > 0:
            print(f"   ⚠️  PARTIALLY VALIDATED ({validated_positive}/{len(TICKERS)} sectors)")
        else:
            print(f"   ❌ NOT VALIDATED")
    
    print("\n" + "="*70)
    print("✅ H2 VALIDATION COMPLETE!")
    print("="*70)
    print("\n📌 Next Steps:")
    print("   1. Check results/h2_validation_summary_XGBoost.csv")
    print("   2. Add results to Progress Report Section 3.4")
    print("   3. Use h2_comparison_XX.png in presentation")

if __name__ == "__main__":
    main()