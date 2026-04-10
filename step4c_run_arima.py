"""
STEP 4c: ARIMA Benchmark (Pure Time-Series) - FIXED
====================================================
修正了 'Close' 列缺失的问题。
如果 CSV 里没有股价，会自动从 Yahoo Finance 下载。
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score
import os
import warnings
from tqdm import tqdm
import yfinance as yf # 引入 yfinance

warnings.filterwarnings('ignore')

print("="*70)
print("⏳ STEP 4c: RUNNING ARIMA (The Econometric Benchmark)")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']

# 配置代理 (防止内地无法下载)
try:
    import config
    PROXY_URL = getattr(config, 'PROXY_URL', 'http://127.0.0.1:7890')
    os.environ['HTTP_PROXY'] = PROXY_URL
    os.environ['HTTPS_PROXY'] = PROXY_URL
except ImportError:
    pass

def get_price_data(ticker):
    """
    尝试从 CSV 读取，如果失败则下载
    """
    # 1. 尝试从本地 features 文件读取
    path = f"data/features_sentiment_macro_{ticker}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if 'Close' in df.columns:
            return df['Close']
    
    # 2. 如果本地没有 Close 列，则从 Yahoo 下载
    print(f"      ⚠️  'Close' price not found in CSV. Downloading from Yahoo...")
    try:
        # 使用不传 proxy 参数的方式 (依赖环境变量)
        df_yahoo = yf.download(ticker, start='2021-01-01', progress=False)
        if isinstance(df_yahoo.columns, pd.MultiIndex):
            df_yahoo.columns = df_yahoo.columns.get_level_values(0)
        return df_yahoo['Close']
    except Exception as e:
        print(f"      ❌ Download failed: {e}")
        return None

def run_arima(ticker):
    print(f"   📉 Modeling {ticker} with ARIMA...")
    
    prices = get_price_data(ticker)
    
    if prices is None or prices.empty:
        print(f"      ❌ No price data available for {ticker}.")
        return None
    
    # 确保是 Series 且没有 NaN
    prices = prices.dropna()
    
    # 滚动预测 (Rolling Forecast)
    # ARIMA 比较慢，我们只测最后 100 天
    test_size = 100
    if len(prices) < test_size + 50:
        print("      ❌ Not enough data.")
        return None

    train_data = prices.iloc[:-test_size].tolist()
    test_data = prices.iloc[-test_size:].tolist()
    
    predictions = []
    
    print(f"      Forecasting last {test_size} days...")
    
    # 使用 tqdm 显示进度
    for t in tqdm(range(len(test_data))):
        # ARIMA(5,1,0) 是个常用的参数组合
        try:
            model = ARIMA(train_data, order=(5,1,0)) 
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
        except:
            # 如果不收敛，用前一天的价格代替 (Naive Forecast)
            predictions.append(train_data[-1])
        
        # 把真实值加入训练集，供下一次预测
        obs = test_data[t]
        train_data.append(obs)
    
    # 计算方向准确率 (Directional Accuracy)
    real_direction = []
    pred_direction = []
    
    for i in range(len(predictions)):
        # 昨收价是 test_data[i-1] (如果是第一个，则是 train_data最后一个)
        prev_close = train_data[-(len(predictions)-i+1)] 
        
        # 涨跌方向：1=涨, 0=跌
        real_dir = 1 if test_data[i] > prev_close else 0
        pred_dir = 1 if predictions[i] > prev_close else 0
        
        real_direction.append(real_dir)
        pred_direction.append(pred_dir)
        
    acc = accuracy_score(real_direction, pred_direction)
    print(f"      ✅ ARIMA Accuracy: {acc:.2%}")
    
    return acc

def main():
    summary = []
    for ticker in TICKERS:
        try:
            acc = run_arima(ticker)
            if acc is not None:
                summary.append({'Ticker': ticker, 'ARIMA_Accuracy': acc})
        except Exception as e:
            print(f"Error {ticker}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n📊 ARIMA RESULTS:")
    if summary:
        df_res = pd.DataFrame(summary)
        print(df_res)
        df_res.to_csv('results/arima_benchmark.csv', index=False)
    else:
        print("No results.")

if __name__ == "__main__":
    main()