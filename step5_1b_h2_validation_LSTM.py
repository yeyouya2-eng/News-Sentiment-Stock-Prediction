"""
STEP 5.1b: H2 Validation using LSTM
====================================
用LSTM重做H2验证，对比Baseline vs Full模型

目的：
• 验证H2在best-performing model上是否成立
• 与XGBoost结果对比，确保robustness
• 提供更strong的evidence

方法：
• Baseline LSTM: Technical + Macro only (no sentiment)
• Full LSTM: Technical + Macro + Sentiment
• 统计测试: McNemar's test
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

print("="*70)
print("🧠 H2 VALIDATION: LSTM VERSION")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']

def load_data(ticker):
    """加载数据"""
    data_path = f"data/features_sentiment_macro_{ticker}.csv"
    if not os.path.exists(data_path):
        data_path = f"data/features_sentiment_focused_{ticker}.csv"
    if not os.path.exists(data_path):
        return None
    print(f"   📂 Loading: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    return df

def define_feature_sets(df):
    """定义baseline和full特征集"""
    all_features = [c for c in df.columns if c not in [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
        'Target', 'Next_Open', 'Next_Close', 'Tradable_Return', 'Log_Return'
    ]]
    
    # Baseline: 不用sentiment
    baseline_features = [f for f in all_features if 'Sentiment' not in f]
    
    # Full: 所有特征
    full_features = all_features
    
    # Sentiment-only
    sentiment_features = [f for f in all_features if 'Sentiment' in f]
    
    return baseline_features, full_features, sentiment_features

def create_sequences(data, target, lookback):
    """创建LSTM序列"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def train_and_predict_lstm(X_train, y_train, X_test, units=64, dropout=0.2, lookback=60):
    """训练LSTM并预测"""
    
    # 构建模型
    model = Sequential([
        Input(shape=(lookback, X_train.shape[2])),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=0
    )
    
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    
    # 预测
    predictions = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    
    # 清理
    del model
    tf.keras.backend.clear_session()
    
    return predictions

def run_h2_validation_lstm(ticker):
    """LSTM版本的H2验证"""
    
    print(f"\n{'='*70}")
    print(f"🧠 LSTM H2 Validation: {ticker}")
    print(f"{'='*70}")
    
    # 1. 加载数据
    df = load_data(ticker)
    if df is None:
        print(f"   ❌ Data not found for {ticker}")
        return None
    
    # 2. 定义特征集
    baseline_features, full_features, sentiment_features = define_feature_sets(df)
    
    print(f"\n   📊 Features:")
    print(f"      • Baseline (no sentiment): {len(baseline_features)}")
    print(f"      • Full (with sentiment):   {len(full_features)}")
    print(f"      • Sentiment features:      {len(sentiment_features)}")
    
    # 3. 划分训练/测试集（80/20）
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\n   📊 Data Split:")
    print(f"      • Train: {len(train_df)} days")
    print(f"      • Test:  {len(test_df)} days")
    
    # 4. LSTM参数（根据ticker调整）
    # lookback统一改为20，原60步×29特征=1740维输入对249天测试集过拟合
    if ticker == 'XLF':
        units, dropout, lookback = 32, 0.4, 20
    else:
        units, dropout, lookback = 64, 0.2, 20
    
    print(f"\n   🤖 LSTM Config:")
    print(f"      • Units: {units}")
    print(f"      • Dropout: {dropout}")
    print(f"      • Lookback: {lookback}")
    
    # 5. Baseline Model (No Sentiment)
    print(f"\n   🔄 Training Baseline LSTM (no sentiment)...")
    
    # 标准化
    scaler_baseline = StandardScaler()
    train_scaled_baseline = scaler_baseline.fit_transform(train_df[baseline_features])
    test_scaled_baseline = scaler_baseline.transform(test_df[baseline_features])
    
    # 创建序列
    X_train_baseline, y_train_baseline = create_sequences(
        train_scaled_baseline, 
        train_df['Target'].values, 
        lookback
    )
    X_test_baseline, y_test_baseline = create_sequences(
        test_scaled_baseline, 
        test_df['Target'].values, 
        lookback
    )
    
    # 训练和预测
    baseline_preds = train_and_predict_lstm(
        X_train_baseline, y_train_baseline, 
        X_test_baseline, 
        units=units, dropout=dropout, lookback=lookback
    )
    
    baseline_acc = accuracy_score(y_test_baseline, baseline_preds)
    print(f"      ✅ Baseline Accuracy: {baseline_acc:.2%}")
    
    # 6. Full Model (With Sentiment)
    print(f"\n   🔄 Training Full LSTM (with sentiment)...")
    
    # 标准化
    scaler_full = StandardScaler()
    train_scaled_full = scaler_full.fit_transform(train_df[full_features])
    test_scaled_full = scaler_full.transform(test_df[full_features])
    
    # 创建序列
    X_train_full, y_train_full = create_sequences(
        train_scaled_full, 
        train_df['Target'].values, 
        lookback
    )
    X_test_full, y_test_full = create_sequences(
        test_scaled_full, 
        test_df['Target'].values, 
        lookback
    )
    
    # 训练和预测
    full_preds = train_and_predict_lstm(
        X_train_full, y_train_full, 
        X_test_full, 
        units=units, dropout=dropout, lookback=lookback
    )
    
    full_acc = accuracy_score(y_test_full, full_preds)
    print(f"      ✅ Full Accuracy: {full_acc:.2%}")
    
    # 7. 比较
    print(f"\n   📊 Comparison:")
    improvement = (full_acc - baseline_acc) * 100
    print(f"      • Baseline (No Sentiment): {baseline_acc:.2%}")
    print(f"      • Full (With Sentiment):   {full_acc:.2%}")
    print(f"      • Improvement:             {improvement:+.2f} percentage points")
    
    if improvement > 0:
        print(f"      ✅ Sentiment features improve performance!")
    else:
        print(f"      ⚠️  Sentiment features hurt performance")
    
    # 8. McNemar's Statistical Test
    print(f"\n   🔬 Statistical Significance Test (McNemar):")
    
    # 创建contingency table
    both_correct = ((baseline_preds == y_test_baseline) & (full_preds == y_test_full)).sum()
    baseline_only = ((baseline_preds == y_test_baseline) & (full_preds != y_test_full)).sum()
    full_only = ((baseline_preds != y_test_baseline) & (full_preds == y_test_full)).sum()
    both_wrong = ((baseline_preds != y_test_baseline) & (full_preds != y_test_full)).sum()
    
    contingency_table = np.array([[both_correct, baseline_only],
                                   [full_only, both_wrong]])
    
    print(f"\n      Contingency Table:")
    print(f"                        Full Correct  Full Wrong")
    print(f"      Baseline Correct    {both_correct:>12}  {baseline_only:>11}")
    print(f"      Baseline Wrong      {full_only:>12}  {both_wrong:>11}")
    
    # McNemar's test
    if baseline_only + full_only > 0:
        result = mcnemar(contingency_table, exact=False, correction=True)
        p_value = result.pvalue
        
        print(f"\n      McNemar's Test:")
        print(f"      • Chi-square statistic: {result.statistic:.4f}")
        print(f"      • P-value: {p_value:.4f}")
        
        # 判断H2状态
        if p_value < 0.05:
            if improvement > 0:
                print(f"      ✅ Significant (p < 0.05)")
                print(f"      → Sentiment IMPROVES performance significantly")
                h2_status = "VALIDATED"
            else:
                print(f"      ⚠️  Significant (p < 0.05)")
                print(f"      → Sentiment HURTS performance significantly")
                h2_status = "VALIDATED (Negative)"
        elif p_value < 0.10:
            if improvement > 0:
                print(f"      ⚠️  Marginally Significant (p < 0.10)")
                print(f"      → Sentiment shows marginal improvement")
                h2_status = "PARTIAL"
            else:
                print(f"      ⚠️  Marginally Significant (p < 0.10)")
                print(f"      → Sentiment shows marginal negative effect")
                h2_status = "PARTIAL (Negative)"
        else:
            print(f"      ❌ Not Significant (p >= 0.10)")
            print(f"      → The improvement is NOT statistically significant")
            h2_status = "REJECTED"
    else:
        print(f"      ⚠️  Cannot perform McNemar's test (no disagreements)")
        p_value = 1.0
        h2_status = "INCONCLUSIVE"
    
    # 9. 保存结果
    summary = {
        'Ticker': ticker,
        'Model': 'LSTM',
        'Baseline_Features': len(baseline_features),
        'Full_Features': len(full_features),
        'Sentiment_Features': len(sentiment_features),
        'Baseline_Accuracy': baseline_acc,
        'Full_Accuracy': full_acc,
        'Accuracy_Improvement': improvement,
        'McNemar_P_Value': p_value,
        'H2_Status': h2_status,
        'LSTM_Units': units,
        'LSTM_Dropout': dropout,
        'Lookback': lookback
    }
    
    print(f"\n{'='*70}\n")
    
    return summary

def main():
    """主函数"""
    print("📌 STEP 5.1b: H2 Validation using LSTM\n")
    print("H2: Sentiment features improve prediction accuracy\n")
    print("Method: Baseline (no sentiment) vs Full (with sentiment)\n")
    print("⏰ Expected time: 10-15 minutes per ticker\n")
    
    all_summaries = []
    
    for ticker in TICKERS:
        try:
            summary = run_h2_validation_lstm(ticker)
            if summary:
                all_summaries.append(summary)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存综合结果
    if all_summaries:
        print("="*70)
        print("📊 FINAL H2 VALIDATION SUMMARY (LSTM)")
        print("="*70 + "\n")
        
        comparison_df = pd.DataFrame(all_summaries)
        
        for _, row in comparison_df.iterrows():
            print(f"{row['Ticker']}:")
            print(f"   • Improvement: {row['Accuracy_Improvement']:+.2f} pp")
            print(f"   • P-value: {row['McNemar_P_Value']:.4f}")
            print(f"   • H2 Status: {row['H2_Status']}")
            print()
        
        # 保存
        if not os.path.exists('results'):
            os.makedirs('results')
        
        comparison_df.to_csv('results/h2_validation_summary_LSTM.csv', index=False)
        print(f"✅ Saved: results/h2_validation_summary_LSTM.csv")
        
        # 判断H2整体状态
        validated_positive = (comparison_df['H2_Status'] == 'VALIDATED').sum()
        validated_negative = (comparison_df['H2_Status'] == 'VALIDATED (Negative)').sum()
        
        print(f"\n🎯 H2 Hypothesis Overall Status (LSTM):")
        if validated_positive == len(TICKERS):
            print(f"   ✅ FULLY VALIDATED (positive impact)")
        elif validated_positive > 0:
            print(f"   ⚠️  PARTIALLY VALIDATED ({validated_positive}/{len(TICKERS)} positive)")
            if validated_negative > 0:
                print(f"   ⚠️  {validated_negative} sector(s) show NEGATIVE impact")
        elif validated_negative > 0:
            print(f"   ⚠️  VALIDATED but NEGATIVE in {validated_negative} sector(s)")
        else:
            print(f"   ❌ NOT VALIDATED")
    
    print("\n" + "="*70)
    print("✅ LSTM H2 VALIDATION COMPLETE!")
    print("="*70)
    print("\n📌 Compare with XGBoost results:")
    print("   • XGBoost: results/h2_validation_summary.csv")
    print("   • LSTM: results/h2_validation_summary_LSTM.csv")

if __name__ == "__main__":
    main()