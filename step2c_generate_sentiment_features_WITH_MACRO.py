"""
STEP 2c: Generate Sentiment Features + Macro Indicators
========================================================
Enhancement: Added macro-economic indicators (VIX, TNX, DXY)
to improve predictive power beyond sentiment + technicals.

Macro indicators capture systematic risk and market regime,
which are crucial for financial assets.
"""

import pandas as pd
import numpy as np
import os
import re
import joblib
from tqdm import tqdm
import yfinance as yf
import talib as ta
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("📊 STEP 2c: GENERATING FEATURES (SENTIMENT + MACRO)")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']

def clean_text(text):
    """Clean text (must match Step 2b preprocessing!)"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def load_trained_model(ticker):
    """Load the trained sentiment classifier"""
    model_path = f"models/sentiment_classifier_{ticker}.pkl"
    
    if not os.path.exists(model_path):
        print(f"   ❌ Model not found: {model_path}")
        return None
    
    model = joblib.load(model_path)
    print(f"   ✅ Loaded model: {model_path}")
    return model

def predict_sentiment_scores(texts, model):
    """Predict sentiment for a list of texts"""
    texts_clean = [clean_text(t) for t in texts]
    probs = model.predict_proba(texts_clean)
    
    # Convert to sentiment score: -1 to +1
    sentiment_scores = []
    for prob in probs:
        score = (-1 * prob[0]) + (0 * prob[1]) + (1 * prob[2])
        sentiment_scores.append(score)
    
    return np.array(sentiment_scores)

def aggregate_daily_sentiment(df_news, model, text_col, date_col):
    """Aggregate news sentiment by day"""
    print("   🧠 Predicting sentiment for all articles...")
    
    sentiments = predict_sentiment_scores(df_news[text_col].tolist(), model)
    df_news['Sentiment_Score'] = sentiments
    
    df_news[date_col] = pd.to_datetime(df_news[date_col])
    
    daily_sentiment = df_news.groupby(date_col).agg({
        'Sentiment_Score': ['mean', 'std', 'count']
    }).reset_index()
    
    daily_sentiment.columns = ['Date', 'Sentiment_FinBERT', 'Sentiment_Std_Daily', 'News_Count']
    daily_sentiment.set_index('Date', inplace=True)
    
    return daily_sentiment

def get_stock_data(ticker):
    """Download stock price data with retry"""
    print(f"   📈 Downloading stock data for {ticker}...")
    
    # Try to use proxy if available
    try:
        import config
        if hasattr(config, 'PROXY_URL'):
            os.environ['HTTP_PROXY'] = config.PROXY_URL
            os.environ['HTTPS_PROXY'] = config.PROXY_URL
    except:
        pass
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start='2021-01-01', progress=False, auto_adjust=True)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if not df.empty:
                print(f"      ✅ Downloaded {len(df)} days of stock data")
                return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                print(f"      ⚠️ Attempt {attempt+1} failed, waiting {wait_time}s...")
                import time
                time.sleep(wait_time)
            else:
                print(f"   ❌ Failed: {e}")
                return None
    
    return None

def get_macro_indicators():
    """
    Download macro-economic indicators.
    REQUIRED: VIX, Treasury_10Y, Dollar_Index must all succeed.
    Raises RuntimeError if any download fails to prevent silently
    saving a CSV without macro features.
    """
    print("   🌍 Downloading macro indicators (proxy must be active)...")

    REQUIRED = {
        'VIX':          '^VIX',
        'Treasury_10Y': '^TNX',
        'Dollar_Index': 'DX-Y.NYB',
    }

    indicators = {}
    failed = []

    for name, symbol in REQUIRED.items():
        try:
            raw = yf.download(symbol, start='2021-01-01', progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if close.empty:
                raise ValueError(f"Empty data for {symbol}")
            indicators[name] = close
            print(f"      ✅ {name} ({symbol}): {len(close)} rows")
        except Exception as e:
            failed.append(name)
            print(f"      ❌ {name} ({symbol}) FAILED: {e}")

    if failed:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"❌ MACRO DOWNLOAD FAILED for: {failed}\n"
            f"   Check proxy is active in config.py (PROXY_URL).\n"
            f"   DO NOT proceed without macro features.\n"
            f"{'='*60}"
        )

    df_macro = pd.DataFrame(indicators)
    df_macro.index = pd.to_datetime(df_macro.index)
    print(f"      ✅ All macro indicators ready: {list(df_macro.columns)}")
    return df_macro

def create_technical_indicators(df):
    """Add technical indicators"""
    print("   📊 Computing technical indicators...")
    
    # Moving averages
    df['MA5'] = ta.SMA(df['Close'], timeperiod=5)
    df['MA10'] = ta.SMA(df['Close'], timeperiod=10)
    df['MA20'] = ta.SMA(df['Close'], timeperiod=20)
    df['MA50'] = ta.SMA(df['Close'], timeperiod=50)
    
    # Momentum indicators
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = ta.MACD(df['Close'])
    df['BB_Upper'], _, df['BB_Lower'] = ta.BBANDS(df['Close'], timeperiod=20)
    
    # Returns and Target
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Target'] = (df['Log_Return'].shift(-1) > 0).astype(int)
    df['Tradable_Return'] = df['Log_Return'].shift(-1)
    
    return df

def create_sentiment_features(df, sentiment_series):
    """Create comprehensive sentiment features"""
    # Join sentiment data
    df = df.join(sentiment_series[['Sentiment_FinBERT', 'News_Count']], how='left')
    
    # Forward fill (news lag effect)
    df['Sentiment_FinBERT'].fillna(method='ffill', inplace=True)
    df['Sentiment_FinBERT'].fillna(0, inplace=True)
    df['News_Count'].fillna(0, inplace=True)
    
    # Derived sentiment features
    base_sent = df['Sentiment_FinBERT']
    
    df['Sentiment_Raw'] = base_sent
    df['Sentiment_EMA'] = base_sent.ewm(span=5, adjust=False).mean()
    df['Sentiment_MA3'] = base_sent.rolling(3).mean()
    df['Sentiment_MA10'] = base_sent.rolling(10).mean()
    
    df['Sentiment_Vol'] = base_sent.rolling(10).std()
    df['Sentiment_Std5'] = base_sent.rolling(5).std()
    df['Sentiment_Std10'] = base_sent.rolling(10).std()
    
    df['Sentiment_Momentum'] = df['Sentiment_EMA'] - df['Sentiment_MA10']
    df['Sentiment_Rate_of_Change'] = base_sent.diff(3)
    
    # Cross features with price
    if 'RSI' in df.columns:
        df['Sentiment_Price_Divergence'] = df['Sentiment_EMA'] - (df['RSI']/100 - 0.5)
    
    # Volume ratio
    if 'Volume' in df.columns:
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df

def add_macro_features(df, df_macro):
    """
    Add macro indicators to feature set
    
    Args:
        df: Main stock data DataFrame
        df_macro: Macro indicators DataFrame
    """
    if df_macro is None or df_macro.empty:
        print("   ⚠️ No macro indicators to add")
        return df
    
    print("   🌍 Adding macro features...")
    
    # Join macro data
    df = df.join(df_macro, how='left')
    
    # Forward fill macro indicators (they update less frequently)
    for col in df_macro.columns:
        if col in df.columns:
            df[col].fillna(method='ffill', inplace=True)
            df[col].fillna(method='bfill', inplace=True)
    
    # Create derived macro features
    if 'VIX' in df.columns:
        df['VIX_MA5'] = df['VIX'].rolling(5).mean()
        df['VIX_Change'] = df['VIX'].diff()
    
    if 'Treasury_10Y' in df.columns:
        df['Treasury_Change'] = df['Treasury_10Y'].diff()
    
    # Interaction: Sentiment × VIX (fear amplifies sentiment?)
    if 'VIX' in df.columns and 'Sentiment_EMA' in df.columns:
        df['Sentiment_VIX_Interaction'] = df['Sentiment_EMA'] * (df['VIX'] / 20)
    
    print(f"      ✅ Added {len(df_macro.columns)} macro indicators")
    print(f"      📊 Total features after macro: {len(df.columns)}")
    
    return df

def generate_features_for_sector(ticker):
    print(f"\n{'='*60}")
    print(f"🔄 PROCESSING: {ticker}")
    print(f"{'='*60}")
    
    # 1. Load trained model
    model = load_trained_model(ticker)
    if model is None:
        return
    
    # 2. Load news data
    news_path = f"data/raw_news_{ticker}_holdings.csv"
    if not os.path.exists(news_path):
        news_path = f"raw_news_{ticker}_holdings.csv"
    
    if not os.path.exists(news_path):
        print(f"   ❌ News file not found")
        return
    
    df_news = pd.read_csv(news_path)
    print(f"   📰 Loaded {len(df_news)} news articles")
    
    # Find columns
    text_col = next((c for c in ['Headline', 'Title', 'Snippet'] if c in df_news.columns), None)
    date_col = next((c for c in ['Date', 'date'] if c in df_news.columns), 'Date')
    
    if not text_col:
        print(f"   ❌ No text column found")
        return
    
    # 3. Generate daily sentiment
    daily_sentiment = aggregate_daily_sentiment(df_news, model, text_col, date_col)
    print(f"   ✅ Generated sentiment for {len(daily_sentiment)} days")
    
    # 4. Load stock data
    df_stock = get_stock_data(ticker)
    if df_stock is None or df_stock.empty:
        print(f"   ❌ No stock data")
        return
    
    # 5. Load macro indicators (NEW!)
    df_macro = get_macro_indicators()
    
    # 6. Create technical indicators
    df_stock = create_technical_indicators(df_stock)
    
    # 7. Merge sentiment features
    df_final = create_sentiment_features(df_stock, daily_sentiment)
    
    # 8. Add macro features (NEW!)
    df_final = add_macro_features(df_final, df_macro)
    
    # 9. Clean and save
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.dropna(inplace=True)
    
    # Save
    output_path = f"data/features_sentiment_macro_{ticker}.csv"
    df_final.to_csv(output_path)
    
    print(f"   ✅ Saved: {output_path}")
    print(f"   📊 Final dataset: {len(df_final)} days")
    print(f"   📋 Total features: {len(df_final.columns)} columns")
    
    # Count feature types
    sentiment_feats = [c for c in df_final.columns if 'Sentiment' in c or 'sentiment' in c]
    macro_feats = [c for c in df_final.columns if c in ['VIX', 'Treasury_10Y', 'Dollar_Index', 'VIX_MA5', 'VIX_Change', 'Treasury_Change', 'Sentiment_VIX_Interaction']]
    
    print(f"\n   📊 Feature Breakdown:")
    print(f"      Sentiment features: {len(sentiment_feats)}")
    print(f"      Macro features: {len(macro_feats)}")
    print(f"      Technical features: {len([c for c in df_final.columns if c in ['MA5','MA10','MA20','MA50','RSI','MACD','BB_Upper','BB_Lower']])}")
    
    # Show sample statistics
    print(f"\n   📊 Sentiment Statistics:")
    print(f"      Mean: {df_final['Sentiment_FinBERT'].mean():.3f}")
    print(f"      Std:  {df_final['Sentiment_FinBERT'].std():.3f}")
    
    if 'VIX' in df_final.columns:
        print(f"\n   📊 VIX Statistics:")
        print(f"      Mean: {df_final['VIX'].mean():.2f}")
        print(f"      Std:  {df_final['VIX'].std():.2f}")
    
    # Save feature list
    features = [c for c in df_final.columns if c not in 
                ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Tradable_Return']]
    
    with open(f"data/sentiment_macro_features_{ticker}.txt", 'w') as f:
        for feat in features:
            f.write(feat + '\n')
    
    print(f"   ✅ Saved feature list: data/sentiment_macro_features_{ticker}.txt")

def main():
    print("\n💡 Enhancement: Adding macro-economic indicators")
    print("   VIX, Treasury Yield, Dollar Index")
    print("="*70 + "\n")
    
    for ticker in TICKERS:
        try:
            generate_features_for_sector(ticker)
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ FEATURE GENERATION COMPLETE (WITH MACRO)")
    print("="*70)
    print("\n📌 Next Steps:")
    print("   1. Retrain models with new features:")
    print("      - Modify step3/4 to use 'features_sentiment_macro_XX.csv'")
    print("   2. Compare accuracy before/after macro indicators")
    print("   3. Check feature importance (do macro features rank high?)")
    print("\n💡 Expected improvement: 1-3% accuracy increase")

if __name__ == "__main__":
    main()
