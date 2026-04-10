"""
STEP 4b Extended: Baseline Models Rolling Backtest
===================================================
给Logistic Regression和Random Forest补充完整的rolling评估
包括：Sharpe Ratio, Return, Precision等指标

目的：填补Excel表中的N/A
"""

import matplotlib
matplotlib.use('Agg')  # Fix: must be set BEFORE importing pyplot to prevent tkinter threading crash on Windows

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

print("="*70)
print("📈 BASELINE MODELS - ROLLING BACKTEST")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']
TRAIN_WINDOW = 252  # 1 year training window
REFIT_FREQ = 20     # Refit every 20 days

def load_data(ticker):
    """加载数据"""
    data_path = f"data/features_sentiment_macro_{ticker}.csv"
    if not os.path.exists(data_path):
        print(f"   ❌ Data not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)
    return df

def calculate_sharpe_ratio(returns):
    """计算Sharpe Ratio"""
    if len(returns) == 0:
        return 0
    
    # 假设无风险利率=0（简化）
    # Sharpe = mean(returns) / std(returns) * sqrt(252)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0
    
    sharpe = (mean_return / std_return) * np.sqrt(252)
    return sharpe

def run_rolling_backtest(ticker, model_name='Logistic'):
    """运行rolling window backtest"""
    
    print(f"\n{'='*60}")
    print(f"🔄 ROLLING BACKTEST: {ticker} - {model_name}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    df = load_data(ticker)
    if df is None:
        return None
    
    # 2. 定义features
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
               'Target', 'Next_Open', 'Next_Close', 'Tradable_Return', 'Log_Return']
    features = [c for c in df.columns if c not in exclude]
    
    print(f"   📋 Features: {len(features)}")
    
    # 3. 初始化模型
    if model_name == 'Logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:  # Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 4. Rolling Window Prediction
    predictions = []
    actuals = []
    returns = []
    dates = []
    
    last_refit = 0
    
    print(f"   ⏳ Backtesting {len(df) - TRAIN_WINDOW} days...")
    
    for i in tqdm(range(TRAIN_WINDOW, len(df)), desc=f"   {ticker}"):
        
        # 决定是否需要refit
        if i - last_refit >= REFIT_FREQ or i == TRAIN_WINDOW:
            # Refit model
            train_data = df.iloc[i-TRAIN_WINDOW:i]
            X_train = train_data[features]
            y_train = train_data['Target']
            
            model.fit(X_train, y_train)
            last_refit = i
        
        # Predict
        test_row = df.iloc[i:i+1]
        X_test = test_row[features]
        y_test = test_row['Target'].values[0]
        
        pred = model.predict(X_test)[0]
        
        # Calculate return
        # 如果预测up(1)，买入；如果预测down(0)，不买（现金）
        actual_return = test_row['Log_Return'].values[0]
        
        if pred == 1:  # 预测上涨，买入
            strategy_return = actual_return - 0.001  # 0.1% transaction cost
        else:  # 预测下跌，持有现金
            strategy_return = 0  # 现金无收益
        
        predictions.append(pred)
        actuals.append(y_test)
        returns.append(strategy_return)
        dates.append(test_row.index[0])
    
    # 5. 计算指标
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, zero_division=0)
    
    # Cumulative return
    cumulative_returns = np.exp(np.cumsum(returns)) - 1  # Log returns to simple returns
    total_return = cumulative_returns[-1]
    
    # Market return (buy & hold)
    market_returns = df.iloc[TRAIN_WINDOW:]['Log_Return'].values
    market_cumulative = np.exp(np.cumsum(market_returns)) - 1
    market_return = market_cumulative[-1]
    
    # Sharpe Ratio
    sharpe = calculate_sharpe_ratio(returns)
    
    # 6. 输出结果
    print(f"\n   📊 RESULTS: {ticker} - {model_name}")
    print(f"      Accuracy: {accuracy:.2%}")
    print(f"      Precision: {precision:.2%}")
    print(f"      Sharpe Ratio: {sharpe:.2f}")
    print(f"      Total Return: {total_return:.2%}")
    print(f"      Market Return: {market_return:.2%}")
    print(f"      vs Market: {(total_return - market_return):.2%}")
    
    # 7. 保存结果
    result = {
        'Model': model_name,
        'Ticker': ticker,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sharpe': sharpe,
        'Total_Return': total_return,
        'Market_Return': market_return,
        'vs_Market': total_return - market_return,
        'Test_Days': len(predictions)
    }
    
    # 保存predictions
    pred_df = pd.DataFrame({
        'Date': dates,
        'Actual': actuals,
        'Prediction': predictions,
        'Return': returns
    })
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    pred_df.to_csv(f'results/rolling_predictions_{model_name}_{ticker}.csv', index=False)
    print(f"   🖼️  Saved: results/rolling_predictions_{model_name}_{ticker}.csv")
    
    # 绘图
    create_backtest_plot(ticker, model_name, cumulative_returns, market_cumulative, dates)
    
    return result

def create_backtest_plot(ticker, model_name, strategy_returns, market_returns, dates):
    """创建回测可视化"""
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates, strategy_returns * 100, label=f'{model_name} Strategy', linewidth=2)
    plt.plot(dates, market_returns * 100, label='Buy & Hold', linewidth=2, linestyle='--')
    
    plt.title(f'{ticker} - {model_name} Rolling Backtest', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = f'results/rolling_backtest_{model_name}_{ticker}.png'
    if os.path.exists(out_path):
        os.remove(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   🖼️  Saved: results/rolling_backtest_{model_name}_{ticker}.png")

def main():
    """主函数"""
    
    all_results = []
    
    models = ['Logistic', 'RandomForest']
    
    for ticker in TICKERS:
        for model_name in models:
            try:
                result = run_rolling_backtest(ticker, model_name)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"\n❌ Error {ticker} {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # 汇总结果
    if all_results:
        print("\n" + "="*70)
        print("📊 BASELINE ROLLING BACKTEST SUMMARY")
        print("="*70)
        
        df_results = pd.DataFrame(all_results)
        print("\n" + df_results.to_string(index=False))
        
        # 保存
        df_results.to_csv('results/baseline_rolling_summary.csv', index=False)
        print(f"\n✅ Saved: results/baseline_rolling_summary.csv")
        
        print("\n💡 Now you can update the Excel table with these metrics!")
    
    print("\n" + "="*70)
    print("✅ BASELINE ROLLING BACKTEST COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()