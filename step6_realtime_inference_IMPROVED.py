"""
STEP 6: Real-time Inference Pipeline (IMPROVED VERSION)
========================================================
改进点：
1. ✅ 修复pandas deprecated方法 (fillna(method='ffill'))
2. ✅ 增强特征一致性检查
3. ✅ 改进错误处理和日志输出
4. ✅ 添加特征验证步骤
5. ✅ 优化代理和重试逻辑
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import talib as ta
import datetime
import json
import warnings
import sys
import time

warnings.filterwarnings('ignore')

print("="*70)
print("🔮 REAL-TIME INFERENCE PIPELINE (IMPROVED)")
print("="*70 + "\n")

# ================= 1. 配置设置 =================
try:
    import config
    if hasattr(config, 'PROXY_URL') and config.PROXY_URL:
        os.environ['HTTP_PROXY'] = config.PROXY_URL
        os.environ['HTTPS_PROXY'] = config.PROXY_URL
        print(f"🌍 Proxy Activated: {config.PROXY_URL}")
    else:
        print("⚠️ No proxy configured")
except ImportError:
    print("⚠️ config.py not found. Using defaults.")

TICKERS = ['XLK', 'XLF']

def get_next_trading_day(latest_date):
    """计算下一个交易日（跳过周末）"""
    next_day = latest_date + datetime.timedelta(days=1)
    
    # 跳过周末
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += datetime.timedelta(days=1)
    
    return next_day

def load_best_params(ticker):
    """加载Step 3调优的最佳参数"""
    param_file = f"results/best_params_{ticker}.json"
    default_params = {
        'n_estimators': 100, 
        'max_depth': 4, 
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1
    }
    
    if os.path.exists(param_file):
        try:
            with open(param_file, 'r') as f:
                params = json.load(f)
            # 清理可能冲突的参数
            params.pop('eval_metric', None)
            params.pop('use_label_encoder', None)
            print(f"   ⚙️ Loaded tuned params for {ticker}")
            return params
        except Exception as e:
            print(f"   ⚠️ Error loading params: {e}")
            
    print(f"   ⚠️ Using default params for {ticker}")
    return default_params

def get_market_data_with_retry(ticker, max_retries=5):
    """带重试机制的市场数据下载"""
    # 下载足够长的数据以计算长周期指标
    start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    
    for i in range(max_retries):
        try:
            # 使用auto_adjust=True获取调整后价格
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            
            # 处理多层索引
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if not df.empty:
                print(f"   ✅ Downloaded {len(df)} days of data")
                return df
                
        except Exception as e:
            wait_time = 5 * (i + 1)
            print(f"   ⚠️ Retry {i+1}/{max_retries} failed: {e}")
            if i < max_retries - 1:
                print(f"   ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            
    print(f"   ❌ Failed to download {ticker} after {max_retries} attempts.")
    return None

def engineer_features_robust(df, sentiment_df=None):
    """
    完全复刻 Step 2.6 的特征工程逻辑
    确保 inference 时的特征与 training 时一致
    """
    print("   🔧 Engineering features...")
    
    # 1. 基础技术指标
    df['MA5'] = ta.SMA(df['Close'], timeperiod=5)
    df['MA10'] = ta.SMA(df['Close'], timeperiod=10)
    df['MA20'] = ta.SMA(df['Close'], timeperiod=20)
    df['MA50'] = ta.SMA(df['Close'], timeperiod=50)
    
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['BB_Upper'], _, df['BB_Lower'] = ta.BBANDS(df['Close'], timeperiod=20)
    
    # 2. 宏观数据 (尝试获取)
    try:
        print("   📊 Fetching macro indicators (VIX, TNX)...")
        vix = yf.download("^VIX", start=df.index[0], progress=False, auto_adjust=True)
        tnx = yf.download("^TNX", start=df.index[0], progress=False, auto_adjust=True)
        
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix['Close']
        elif 'Close' in vix.columns:
            vix = vix['Close']
            
        if isinstance(tnx.columns, pd.MultiIndex):
            tnx = tnx['Close']
        elif 'Close' in tnx.columns:
            tnx = tnx['Close']
        
        # 使用新的fillna语法
        df['VIX_Change'] = vix.pct_change().reindex(df.index).ffill().fillna(0)
        df['Yield_Change'] = tnx.pct_change().reindex(df.index).ffill().fillna(0)
        df['VIX_MA20'] = vix.rolling(20).mean().reindex(df.index).ffill().fillna(0)
        print("   ✅ Macro indicators loaded")
    except Exception as e:
        print(f"   ⚠️ Macro indicators unavailable: {e}")
        df['VIX_Change'] = 0
        df['Yield_Change'] = 0
        df['VIX_MA20'] = 0

    # 3. 情感特征集成
    if sentiment_df is not None:
        print("   📰 Integrating sentiment features...")
        sent_cols = [c for c in sentiment_df.columns if 'Sentiment' in c]
        if sent_cols:
            df = df.join(sentiment_df[sent_cols], how='left')
            print(f"   ✅ Added {len(sent_cols)} sentiment columns")
        else:
            df['Sentiment_FinBERT'] = 0
    else:
        df['Sentiment_FinBERT'] = 0
        
    # 4. 填充缺失的情感特征
    sent_cols = [c for c in df.columns if 'Sentiment' in c]
    if sent_cols:
        # 使用新的fillna语法
        df[sent_cols] = df[sent_cols].ffill().fillna(0)
    else:
        df['Sentiment_FinBERT'] = 0
        sent_cols = ['Sentiment_FinBERT']

    # 5. 衍生情感特征
    if 'Sentiment_FinBERT' in df.columns:
        base_sent = df['Sentiment_FinBERT']
    else:
        base_sent = df[sent_cols[0]] if sent_cols else pd.Series(0, index=df.index)

    df['Sentiment_Raw'] = base_sent
    df['Sentiment_EMA'] = base_sent.ewm(span=5, adjust=False).mean()
    df['Sentiment_MA3'] = base_sent.rolling(3).mean()
    df['Sentiment_MA10'] = base_sent.rolling(10).mean()
    
    df['Sentiment_Vol'] = base_sent.rolling(10).std()
    df['Sentiment_Std5'] = base_sent.rolling(5).std()
    df['Sentiment_Std10'] = base_sent.rolling(10).std()
    
    df['Sentiment_Momentum'] = df['Sentiment_EMA'] - df['Sentiment_MA10']
    df['Sentiment_Rate_of_Change'] = base_sent.diff(3)
    
    if 'RSI' in df.columns:
        df['Sentiment_Price_Divergence'] = df['Sentiment_EMA'] - (df['RSI']/100 - 0.5)
    else:
        df['Sentiment_Price_Divergence'] = 0
        
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # 6. 清洗
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"   ✅ Feature engineering complete ({len(df.columns)} features)")
    
    return df

def validate_features(X, expected_features):
    """验证特征完整性"""
    missing = set(expected_features) - set(X.columns)
    extra = set(X.columns) - set(expected_features)
    
    if missing:
        print(f"   ⚠️ WARNING: Missing {len(missing)} features: {list(missing)[:5]}...")
        for f in missing:
            X[f] = 0
            
    if extra:
        print(f"   ℹ️ Note: {len(extra)} extra features will be dropped")
        X = X[expected_features]
        
    return X

def predict_tomorrow(ticker):
    print(f"\n{'='*70}")
    print(f"🚀 Processing {ticker}")
    print(f"{'='*70}")
    
    # 1. 加载本地历史特征
    hist_path = f"data/features_sentiment_focused_{ticker}.csv"
    hist_df = None
    if os.path.exists(hist_path):
        try:
            hist_df = pd.read_csv(hist_path, index_col=0, parse_dates=True)
            print(f"   ✅ Loaded historical features ({len(hist_df)} days)")
        except Exception as e:
            print(f"   ⚠️ Could not load historical features: {e}")
    
    # 2. 获取最新股价
    market_df = get_market_data_with_retry(ticker)
    if market_df is None: 
        print(f"   ❌ Cannot proceed without market data")
        return None

    latest_date = market_df.index[-1].date()
    next_trading_day = get_next_trading_day(latest_date)
    
    print(f"   📅 Latest Market Close: {latest_date}")
    print(f"   🎯 Forecasting Target:  {next_trading_day}")

    # 3. 混合历史情感数据
    if hist_df is not None:
        sent_cols = [c for c in hist_df.columns if 'Sentiment_FinBERT' in c]
        if sent_cols:
            market_df = market_df.join(hist_df[sent_cols], how='left')
            print(f"   ✅ Merged sentiment data")
    
    # 4. 重新计算所有特征
    df_final = engineer_features_robust(market_df)
    
    # 5. 加载特征列表（训练时使用的特征）
    feat_file = f"data/sentiment_focused_features_{ticker}.txt"
    if os.path.exists(feat_file):
        with open(feat_file, 'r') as f:
            FEATURES = [line.strip() for line in f.readlines()]
        print(f"   ✅ Loaded feature list ({len(FEATURES)} features)")
    else:
        print("   ⚠️ Feature list missing. Using heuristic approach.")
        FEATURES = [c for c in df_final.columns 
                   if c not in ['Open','High','Low','Close','Volume','Adj Close', 'Target', 'Log_Return']]
        print(f"   ℹ️ Using {len(FEATURES)} features from dataframe")

    # 6. 验证特征完整性
    X = df_final[FEATURES] if all(f in df_final.columns for f in FEATURES) else df_final
    X = validate_features(X, FEATURES)
    
    # 7. 准备训练数据
    df_final['Log_Return'] = np.log(df_final['Close'] / df_final['Close'].shift(1))
    df_final['Target'] = (df_final['Log_Return'].shift(-1) > 0).astype(int)
    
    # 移除NaN（最后一行没有target）
    valid_idx = df_final['Target'].notna()
    X_train = X[valid_idx]
    y_train = df_final.loc[valid_idx, 'Target']
    X_latest = X.iloc[[-1]]  # 最新的一天用于预测
    
    print(f"   📚 Training on {len(X_train)} historical days...")
    
    # 8. 训练模型
    params = load_best_params(ticker)
    model = xgb.XGBClassifier(**params, eval_metric='logloss')
    
    try:
        model.fit(X_train, y_train)
        print(f"   ✅ Model trained successfully")
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return None
    
    # 9. 生成预测
    prob = model.predict_proba(X_latest)[0][1]
    signal = "BUY" if prob > 0.55 else ("SELL" if prob < 0.45 else "HOLD")
    
    # 10. 输出报告
    print(f"\n   {'='*60}")
    print(f"   📊 FORECAST REPORT: {ticker}")
    print(f"   {'='*60}")
    print(f"   Latest Market Close:  {latest_date}")
    print(f"   Forecasting Target:   {next_trading_day} (Next Trading Day)")
    print(f"   {'-'*60}")
    print(f"   SIGNAL:               {'🟢' if signal=='BUY' else '🔴' if signal=='SELL' else '🟡'} {signal}")
    print(f"   PROBABILITY (UP):     {prob:.2%}")
    print(f"   PROBABILITY (DOWN):   {1-prob:.2%}")
    print(f"   {'-'*60}")
    print(f"   RECOMMENDATION:       {'Strong' if abs(prob-0.5)>0.1 else 'Moderate'} {signal.lower()} signal")
    print(f"   {'='*60}")
    
    # 11. Feature Importance
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-5:][::-1]
    print("\n   🔍 KEY DRIVERS (Top 5 Features):")
    for idx in top_idx:
        feat_name = FEATURES[idx]
        feat_val = X_latest[feat_name].values[0]
        print(f"      • {feat_name:<30} {feat_val:>10.4f} (Imp: {importance[idx]:.3f})")
        
    print("\n   ⚠️  DISCLAIMER:")
    print("      • Research model, not financial advice")
    print("      • Past performance ≠ future results")
    print("      • Use proper risk management")
    print(f"   {'='*60}\n")

    return {
        'Ticker': ticker, 
        'Prob_Up': prob,
        'Signal': signal,
        'Latest_Date': latest_date,
        'Forecast_Date': next_trading_day
    }

def main():
    print("\n" + "="*70)
    print("🤖 NEAR REAL-TIME INFERENCE PIPELINE")
    print("="*70)
    print("Purpose:  Generate next-day directional forecast")
    print("Model:    XGBoost with sentiment-enhanced features")
    print("="*70 + "\n")
    
    results = []
    
    for ticker in TICKERS:
        try:
            result = predict_tomorrow(ticker)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error predicting {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # 综合市场观点
    if len(results) == 2:
        print("\n" + "="*70)
        print("📊 MARKET OVERVIEW")
        print("="*70)
        
        xlk_result = [r for r in results if r['Ticker'] == 'XLK'][0]
        xlf_result = [r for r in results if r['Ticker'] == 'XLF'][0]
        
        xlk_prob = xlk_result['Prob_Up']
        xlf_prob = xlf_result['Prob_Up']
        
        print(f"\nLatest Close: {xlk_result['Latest_Date']}")
        print(f"Forecasting:  {xlk_result['Forecast_Date']}\n")
        print(f"Technology (XLK): {xlk_prob*100:.1f}% bullish → {xlk_result['Signal']}")
        print(f"Financial (XLF):  {xlf_prob*100:.1f}% bullish → {xlf_result['Signal']}")
        
        # 市场情绪判断
        avg_prob = (xlk_prob + xlf_prob) / 2
        if avg_prob > 0.6:
            market_mood = "🟢 RISK-ON (Bullish sentiment across sectors)"
        elif avg_prob < 0.4:
            market_mood = "🔴 RISK-OFF (Bearish sentiment across sectors)"
        else:
            market_mood = "🟡 NEUTRAL (Mixed signals)"
            
        print(f"\nMarket Mood: {market_mood}")
        print("="*70)
        
    print("\n✅ Inference complete!")
    print("\n📌 Note: This system can be scheduled to run daily")
    print("   (e.g., via cron job at market close)\n")

if __name__ == "__main__":
    main()
