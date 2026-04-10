"""
STEP 3: XGBoost Tuning - FORCE SAVE FIX
========================================
修复Windows下JSON文件不更新的问题
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import json
import time
from datetime import datetime

warnings.filterwarnings('ignore')

print("="*70)
print("🚀 STEP 3: XGBOOST TUNING (FORCE SAVE FIX)")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']

def force_save_json(filepath, data):
    """
    强制保存JSON文件并验证
    解决Windows文件缓存问题
    """
    # 1. 如果文件存在，先删除（确保重新创建）
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"      🗑️  Deleted old file: {filepath}")
            time.sleep(0.1)  # 短暂等待确保文件系统同步
        except Exception as e:
            print(f"      ⚠️  Could not delete old file: {e}")
    
    # 2. 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 3. 写入文件（带flush）
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()  # 强制刷新到磁盘
            os.fsync(f.fileno())  # 确保写入物理磁盘
        
        print(f"      💾 Written to: {filepath}")
        
        # 4. 立即验证文件存在且可读
        time.sleep(0.1)  # 短暂等待文件系统
        
        if os.path.exists(filepath):
            # 读回验证
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            if loaded == data:
                # 获取文件信息
                stat = os.stat(filepath)
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"      ✅ VERIFIED! File saved successfully")
                print(f"      📄 Size: {stat.st_size} bytes")
                print(f"      🕒 Modified: {mod_time}")
                return True
            else:
                print(f"      ❌ Content mismatch after save!")
                return False
        else:
            print(f"      ❌ File does not exist after save!")
            return False
            
    except Exception as e:
        print(f"      ❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_data(ticker):
    """Load the sentiment-focused dataset"""
    data_path = f"data/features_sentiment_macro_{ticker}.csv"
    if not os.path.exists(data_path):
        print(f"   ❌ Data not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)
    return df

def tune_xgboost(ticker):
    print(f"\n{'='*60}")
    print(f"🎯 TUNING: {ticker}")
    print(f"{'='*60}")
    
    df = load_data(ticker)
    if df is None: return None, 0

    # 1. Prepare Data
    exclude = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target', 'Tradable_Return']
    features = [c for c in df.columns if c not in exclude]
    
    X = df[features]
    y = df['Target']
    
    # Split Train/Test (Last 20% as Test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"   📋 Features: {len(features)}")
    
    # Show sentiment features
    sentiment_feats = [f for f in features if 'Sentiment' in f or 'sentiment' in f]
    print(f"   💭 Sentiment Features: {len(sentiment_feats)}")
    if sentiment_feats:
        print(f"      {', '.join(sentiment_feats[:5])}...")

    # 2. Define Grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'gamma': [0, 0.1]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    
    print("   ⏳ Running GridSearchCV...")
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    val_acc = grid.best_score_
    
    # 4. Final Test
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n   ✅ Best Params: {best_params}")
    print(f"   🏆 CV Accuracy: {val_acc:.2%}")
    print(f"   🧪 Test Accuracy: {test_acc:.2%}")

    # 5. Feature Importance
    importance = best_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(15)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title(f'Top 15 Features - {ticker} (XGBoost with Sentiment)')
    plt.tight_layout()
    
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig(f'results/feature_importance_xgb_{ticker}.png')
    plt.close()
    print(f"   🖼️  Saved plot: results/feature_importance_xgb_{ticker}.png")
    
    # Check sentiment features importance
    top_5 = feat_imp['Feature'].head(5).tolist()
    sentiment_in_top5 = [f for f in top_5 if 'Sentiment' in f or 'sentiment' in f]
    
    print(f"   🔍 Top 5 Features: {top_5}")
    if sentiment_in_top5:
        print(f"      ✅ Sentiment features in Top 5: {sentiment_in_top5}")
        print(f"      👉 H2 SUPPORTED: Sentiment adds predictive power!")
    else:
        print(f"      ⚠️  No sentiment in Top 5 (Technical indicators dominate)")
    
    # ============ 使用强制保存函数 ============
    print(f"\n   💾 Saving parameters...")
    
    json_file = f'results/best_params_xgb_{ticker}.json'
    success = force_save_json(json_file, best_params)
    
    if not success:
        print(f"\n   ⚠️⚠️⚠️ WARNING: Save may have failed! ⚠️⚠️⚠️")
        print(f"   Please manually check if file exists and is updated.")

    return best_params, test_acc

def main():
    results = []
    for ticker in TICKERS:
        try:
            params, acc = tune_xgboost(ticker)
            if params:
                results.append({'Ticker': ticker, 'Test_Accuracy': acc})
        except Exception as e:
            print(f"Error {ticker}: {e}")
            import traceback
            traceback.print_exc()
            
    if results:
        print("\n" + "="*60)
        print("📊 TUNING SUMMARY")
        print("="*60)
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))
        
        print("\n💡 Next Steps:")
        print("   1. Run step4_run_rolling_model_FIXED.py (XGBoost backtest)")
        print("   2. Run step3_tune_lstm.py (LSTM tuning)")
        print("   3. Run step4b_run_baselines.py (Baseline comparison)")
        print("   4. Run step4c_run_arima.py (ARIMA comparison)")

if __name__ == "__main__":
    main()
