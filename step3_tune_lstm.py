"""
STEP 3: LSTM Tuning - FORCE SAVE FIX
=====================================
修复Windows下JSON文件不更新的问题
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import warnings
import json
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

print("="*70)
print("🧠 STEP 3: LSTM TUNING (FORCE SAVE FIX)")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']
LOOKBACK = 10

def load_data(ticker):
    path = f"data/features_sentiment_macro_{ticker}.csv"
    if not os.path.exists(path):
        print(f"❌ Data not found: {path}")
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)

def force_save_json(filepath, data):
    """
    强制保存JSON文件并验证
    解决Windows文件缓存问题
    """
    import os
    import json
    import time
    from datetime import datetime
    
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

def tune_lstm(ticker):
    print(f"\n{'='*60}")
    print(f"🧬 TUNING LSTM: {ticker}")
    print(f"{'='*60}")
    
    df = load_data(ticker)
    if df is None: return None, 0

    exclude = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target', 'Tradable_Return']
    features = [c for c in df.columns if c not in exclude]
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(train_df[features])
    X_test_raw = scaler.transform(test_df[features])
    
    y_train_raw = train_df['Target'].values
    y_test_raw = test_df['Target'].values
    
    X_train, y_train = create_sequences(X_train_raw, y_train_raw, LOOKBACK)
    X_test, y_test = create_sequences(X_test_raw, y_test_raw, LOOKBACK)
    
    print(f"   📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    param_grid = [
        {'units': 32, 'dropout': 0.2, 'batch': 32, 'lr': 0.001},
        {'units': 64, 'dropout': 0.3, 'batch': 64, 'lr': 0.001},
    ]
    
    best_acc = 0
    best_params = {}
    best_model = None
    
    for params in param_grid:
        print(f"   🧪 Testing: {params} ...", end="")
        start_time = time.time()
        
        model = Sequential()
        model.add(LSTM(params['units'], input_shape=(LOOKBACK, len(features)), return_sequences=False))
        model.add(Dropout(params['dropout']))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=params['batch'],
            validation_split=0.2,
            callbacks=[es],
            verbose=0
        )
        
        pred_prob = model.predict(X_test, verbose=0)
        pred_class = (pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_test, pred_class)
        
        duration = time.time() - start_time
        print(f" Acc: {acc:.2%} ({duration:.1f}s)")
        
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model

    print(f"\n   🏆 Best LSTM Accuracy: {best_acc:.2%}")
    print(f"   ⚙️ Best Params: {best_params}")
    
    # ============ 使用强制保存函数 ============
    print(f"\n   💾 Saving parameters...")
    
    json_file = f'results/best_params_lstm_{ticker}.json'
    success = force_save_json(json_file, best_params)
    
    if not success:
        print(f"\n   ⚠️⚠️⚠️ WARNING: Save may have failed! ⚠️⚠️⚠️")
        print(f"   Please manually check if file exists and is updated.")
    
    return best_params, best_acc

def main():
    results = []
    for ticker in TICKERS:
        try:
            params, acc = tune_lstm(ticker)
            if params:
                results.append({'Ticker': ticker, 'LSTM_Accuracy': acc})
        except Exception as e:
            print(f"❌ Error {ticker}: {e}")
            import traceback
            traceback.print_exc()
            
    if results:
        print("\n" + "="*60)
        print("📊 FINAL LSTM TUNING SUMMARY")
        print("="*60)
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
