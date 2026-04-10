"""
STEP 4: LSTM Rolling Window Backtesting (Topic Focused) - FINAL FIXED
======================================================================
修复了 IndexError: index out of bounds 问题。
原因：create_sequences 内部会自动处理 lookback 偏移，调用时无需手动切片 y。
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm
import json

# 屏蔽 TensorFlow 的调试信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

print("="*70)
print("🧠 STEP 4: LSTM ROLLING BACKTEST (THE FULL RUN - FIXED)")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']
TRAIN_WINDOW = 252      # 初始训练窗口 (1年)
REFIT_STEP = 126        # 每半年重练一次
LOOKBACK = 10           # 时间序列长度

def load_data(ticker):
    path = f"data/features_sentiment_macro_{ticker}.csv"
    if not os.path.exists(path): 
        print(f"❌ Data not found: {path}")
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)

def load_best_params(ticker):
    """自动读取刚才调参生成的最佳参数"""
    path = f"results/best_params_lstm_{ticker}.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            params = json.load(f)
        print(f"   ⚙️  Loaded Best Params: {params}")
        return params
    else:
        print(f"   ⚠️  Params not found, using defaults.")
        return {'units': 50, 'dropout': 0.2, 'batch': 32, 'lr': 0.001}

def create_sequences(X, y, lookback):
    """
    将 2D 数据转换为 LSTM 需要的 3D 序列。
    注意：X 和 y 的长度必须一致 (N)。
    返回的序列数将是 N - lookback。
    """
    Xs, ys = [], []
    # 循环范围：从 0 到 N - lookback
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        # 这里的 y[i + lookback] 是自动取序列末尾对应的那个 Target
        ys.append(y[i + lookback]) 
    return np.array(Xs), np.array(ys)

def run_rolling_lstm(ticker):
    print(f"\n{'='*60}")
    print(f"🔄 ROLLING LSTM: {ticker}")
    print(f"{'='*60}")
    
    df = load_data(ticker)
    params = load_best_params(ticker) 
    
    if df is None: return None

    # 1. 准备特征
    exclude = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target', 'Tradable_Return']
    features = [c for c in df.columns if c not in exclude]
    
    print(f"   📋 Features: {len(features)}")
    
    # 转换为 Numpy 数组
    data_values = df[features].values
    target_values = df['Target'].values
    dates = df.index
    
    predictions = []
    probabilities = []
    test_dates = []
    
    total_len = len(df)
    start_index = TRAIN_WINDOW + LOOKBACK
    
    print(f"   ⏳ Backtesting {total_len - start_index} days...")
    print(f"   🔄 Refitting model every {REFIT_STEP} days...")
    
    scaler = StandardScaler()
    
    # --- 滚动循环 ---
    for t in tqdm(range(start_index, total_len, REFIT_STEP)):
        
        # A. 定义训练窗口
        train_end = t
        train_start = t - TRAIN_WINDOW
        
        # B. 准备训练数据
        X_train_raw = data_values[train_start:train_end]
        y_train_raw = target_values[train_start:train_end]
        
        # 实时标准化
        scaler.fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw)
        
        # C. 创建 3D 序列 (FIXED: 传入完整的 y_train_raw)
        # 错误写法: y_train_raw[LOOKBACK:] -> 导致 IndexError
        # 正确写法: y_train_raw          ->create_sequences 内部会处理偏移
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, LOOKBACK)
        
        # D. 训练模型
        tf.keras.backend.clear_session()
        
        model = Sequential()
        model.add(LSTM(int(params['units']), input_shape=(LOOKBACK, len(features)), return_sequences=False))
        model.add(Dropout(params['dropout']))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        model.fit(X_train_seq, y_train_seq, epochs=15, batch_size=int(params['batch']), 
                  verbose=0, callbacks=[es])
        
        # E. 预测未来一段
        predict_end = min(t + REFIT_STEP, total_len)
        
        # 提取测试数据 (需要包含 Lookback)
        X_test_raw_chunk = data_values[t-LOOKBACK : predict_end]
        X_test_scaled_chunk = scaler.transform(X_test_raw_chunk)
        
        # 创建测试序列 (y用全0占位即可，因为是预测)
        X_test_seq, _ = create_sequences(X_test_scaled_chunk, np.zeros(len(X_test_scaled_chunk)), LOOKBACK)
        
        if len(X_test_seq) > 0:
            probs = model.predict(X_test_seq, verbose=0).flatten()
            preds = (probs > 0.5).astype(int)
            
            predictions.extend(preds)
            probabilities.extend(probs)
            
            # 记录日期
            # create_sequences 会吃掉前 LOOKBACK 个数据
            # 所以如果 X_test_raw_chunk 长度是 N，生成的序列只有 N-LOOKBACK 个
            # 这正好对应从 t 开始到 predict_end 的日期
            chunk_dates = dates[t : t+len(preds)]
            test_dates.extend(chunk_dates)

    # --- 结果汇总 ---
    # 确保长度一致
    min_len = min(len(test_dates), len(predictions))
    result_df = pd.DataFrame({
        'Date': test_dates[:min_len],
        'Pred': predictions[:min_len],
        'Prob': probabilities[:min_len]
    })
    result_df.set_index('Date', inplace=True)
    
    # 合并真实收益
    analysis_df = result_df.join(df[['Target', 'Tradable_Return']], how='inner')
    
    # 计算策略收益
    analysis_df['Strategy_Return'] = analysis_df['Pred'] * analysis_df['Tradable_Return']
    trades = analysis_df['Pred'].diff().abs().fillna(1)
    analysis_df['Strategy_Return_Net'] = analysis_df['Strategy_Return'] - (trades * 0.001)
    
    analysis_df['Cum_Strategy'] = (1 + analysis_df['Strategy_Return_Net']).cumprod()
    analysis_df['Cum_Market'] = (1 + analysis_df['Tradable_Return']).cumprod()
    
    acc = accuracy_score(analysis_df['Target'], analysis_df['Pred'])
    precision = precision_score(analysis_df['Target'], analysis_df['Pred'])
    
    daily_mean = analysis_df['Strategy_Return_Net'].mean()
    daily_std = analysis_df['Strategy_Return_Net'].std()
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else 0
    
    total_ret = (analysis_df['Cum_Strategy'].iloc[-1] - 1) * 100
    market_ret = (analysis_df['Cum_Market'].iloc[-1] - 1) * 100
    
    print(f"\n   📊 LSTM FINAL RESULTS: {ticker}")
    print(f"      Accuracy: {acc:.2%}")
    print(f"      Precision: {precision:.2%}")
    print(f"      Sharpe Ratio: {sharpe:.2f}")
    print(f"      Total Return: {total_ret:.2f}% (vs Market: {market_ret:.2f}%)")
    
    plt.figure(figsize=(12, 6))
    plt.plot(analysis_df.index, analysis_df['Cum_Market'], label='Market', alpha=0.5, linestyle='--')
    plt.plot(analysis_df.index, analysis_df['Cum_Strategy'], label='LSTM Strategy', linewidth=2, color='purple')
    plt.title(f'LSTM Cumulative Return ({ticker})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists('results'): os.makedirs('results')
    out_path = f'results/rolling_backtest_lstm_{ticker}.png'
    if os.path.exists(out_path):
        os.remove(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'Ticker': ticker, 'LSTM_Acc': acc, 'LSTM_Sharpe': sharpe}

def main():
    summary = []
    for ticker in TICKERS:
        try:
            res = run_rolling_lstm(ticker)
            if res: summary.append(res)
        except Exception as e:
            print(f"Error {ticker}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n📊 FINAL LSTM ROLLING SUMMARY")
    print(pd.DataFrame(summary))

if __name__ == "__main__":
    main()