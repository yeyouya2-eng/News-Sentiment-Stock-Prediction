"""
STEP 4: XGBoost Rolling Window Backtesting (Topic Focused)
===========================================================
真正的回测：模拟真实交易环境，杜绝未来函数。

Process:
1. Load Topic Data & Best Params (from Step 3).
2. Rolling Window Loop:
   - Train on [t-window : t]
   - Predict [t+1]
   - Move window forward
3. Calculate Financial Metrics (Sharpe, Cumulative Return).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import warnings
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

print("="*70)
print("📈 STEP 4: XGBOOST ROLLING BACKTEST (WALK-FORWARD)")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']
TRAIN_WINDOW = 252  # 使用过去 1 年 (252交易日) 的数据来训练
TRANSACTION_COST = 0.001 # 交易成本 0.1%

def load_data(ticker):
    """加载 Topic 数据"""
    path = f"data/features_sentiment_macro_{ticker}.csv"
    if not os.path.exists(path):
        print(f"❌ Data not found: {path}")
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)

def load_best_params(ticker):
    """加载 Step 3 算出的最佳参数"""
    path = f"results/best_params_xgb_{ticker}.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    print(f"⚠️  Params not found for {ticker}, using defaults.")
    return {}

def run_rolling_backtest(ticker):
    print(f"\n{'='*60}")
    print(f"🔄 ROLLING: {ticker}")
    print(f"{'='*60}")
    
    df = load_data(ticker)
    params = load_best_params(ticker)
    
    if df is None: return None

    # 准备特征
    exclude = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target', 'Tradable_Return']
    features = [c for c in df.columns if c not in exclude]
    
    print(f"   📋 Features: {len(features)}")
    print(f"   ⚙️  Params: {params}")

    # 存储预测结果
    predictions = []
    probabilities = []
    dates = []
    
    # 开始滚动 (Walk-Forward)
    # 从第 TRAIN_WINDOW 天开始，每一天都做一次预测
    # 为了速度，我们可以每隔 20 天 Retrain 一次模型 (Refit)，但每天都 Predict
    REFIT_STEP = 20 
    
    total_days = len(df)
    start_index = TRAIN_WINDOW
    
    print(f"   ⏳ Backtesting {total_days - start_index} days...")
    
    model = None
    
    # 进度条
    for t in tqdm(range(start_index, total_days)):
        # 当前日期
        current_date = df.index[t]
        
        # 1. 只有在特定间隔才重新训练 (模拟定期更新模型)
        if (t - start_index) % REFIT_STEP == 0 or model is None:
            # 训练集窗口：滚动窗口 [t-window : t]
            train_start = t - TRAIN_WINDOW
            train_end = t
            
            X_train = df[features].iloc[train_start:train_end]
            y_train = df['Target'].iloc[train_start:train_end]
            
            # 训练模型
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1,
                random_state=42,
                **params # 使用最佳参数
            )
            model.fit(X_train, y_train)
        
        # 2. 预测当天 (Out-of-sample)
        # 获取当天特征 (reshape为2D)
        X_current = df[features].iloc[t:t+1]
        
        # 预测概率和类别
        pred_prob = model.predict_proba(X_current)[0][1]
        pred_class = int(pred_prob > 0.5)
        
        predictions.append(pred_class)
        probabilities.append(pred_prob)
        dates.append(current_date)

    # --- 结果分析 ---
    # 对齐数据
    result_df = pd.DataFrame({
        'Date': dates,
        'Pred': predictions,
        'Prob': probabilities
    })
    result_df.set_index('Date', inplace=True)
    
    # 合并真实 Target 和 Return
    # 注意：预测的是 t 的 Target，对应的是 t+1 的 Return
    # 原数据中，第 t 行的 Tradable_Return 就是 t 到 t+1 的收益
    analysis_df = result_df.join(df[['Target', 'Tradable_Return']], how='inner')
    
    # 计算指标
    acc = accuracy_score(analysis_df['Target'], analysis_df['Pred'])
    precision = precision_score(analysis_df['Target'], analysis_df['Pred'])
    
    # --- 策略收益计算 ---
    # 如果预测 1 (涨)，就买入；预测 0 (跌)，就空仓 (或者做空，这里假设只做多)
    # 策略收益 = 预测方向 * 实际收益 - 交易成本
    # 简单策略：Pred=1 买入，Pred=0 空仓 (Cash)
    analysis_df['Strategy_Return'] = analysis_df['Pred'] * analysis_df['Tradable_Return']
    
    # 扣除交易成本 (如果昨天持仓!=今天持仓，说明发生了交易)
    analysis_df['Position'] = analysis_df['Pred']
    analysis_df['Trade_Occurred'] = analysis_df['Position'].diff().abs().fillna(1) # 第一天默认买入
    analysis_df['Strategy_Return_Net'] = analysis_df['Strategy_Return'] - (analysis_df['Trade_Occurred'] * TRANSACTION_COST)
    
    # 累计收益
    analysis_df['Cum_Strategy'] = (1 + analysis_df['Strategy_Return_Net']).cumprod()
    analysis_df['Cum_Market'] = (1 + analysis_df['Tradable_Return']).cumprod()
    
    # Sharpe Ratio (年化)
    # 假设无风险利率为 0 (简化)
    daily_mean = analysis_df['Strategy_Return_Net'].mean()
    daily_std = analysis_df['Strategy_Return_Net'].std()
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else 0
    
    print(f"\n   📊 RESULTS: {ticker}")
    print(f"      Accuracy: {acc:.2%}")
    print(f"      Precision: {precision:.2%}")
    print(f"      Sharpe Ratio: {sharpe:.2f}")
    print(f"      Total Return: {(analysis_df['Cum_Strategy'].iloc[-1]-1)*100:.2f}% (vs Market: {(analysis_df['Cum_Market'].iloc[-1]-1)*100:.2f}%)")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(analysis_df.index, analysis_df['Cum_Market'], label='Buy & Hold (Market)', alpha=0.6)
    plt.plot(analysis_df.index, analysis_df['Cum_Strategy'], label='XGBoost Strategy (Topic)', linewidth=2)
    plt.title(f'Cumulative Return: XGBoost Rolling Strategy ({ticker})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists('results'): os.makedirs('results')
    out_path = f'results/rolling_backtest_{ticker}.png'
    if os.path.exists(out_path):
        os.remove(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   🖼️  Saved plot: results/rolling_backtest_{ticker}.png")
    
    # 保存结果 CSV
    analysis_df.to_csv(f'results/rolling_predictions_{ticker}.csv')
    
    return {
        'Ticker': ticker,
        'Accuracy': acc,
        'Precision': precision,
        'Sharpe': sharpe,
        'Total_Return': (analysis_df['Cum_Strategy'].iloc[-1]-1),
        'Market_Return': (analysis_df['Cum_Market'].iloc[-1]-1)
    }

def main():
    summary = []
    for ticker in TICKERS:
        try:
            res = run_rolling_backtest(ticker)
            if res: summary.append(res)
        except Exception as e:
            print(f"Error {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    if summary:
        print("\n" + "="*60)
        print("📊 FINAL ROLLING BACKTEST SUMMARY")
        print("="*60)
        df_sum = pd.DataFrame(summary)
        print(df_sum)
        df_sum.to_csv('results/final_rolling_summary.csv', index=False)

if __name__ == "__main__":
    main()