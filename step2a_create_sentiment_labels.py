"""
STEP 2a: Create Sentiment Labels (Loughran-McDonald Financial Dictionary)
=========================================================================
Strategy:
- Use Loughran-McDonald (2011) financial sentiment dictionary to label articles
- Count positive/negative words per article → assign label
- No pre-trained model used: fully transparent, rule-based labeling
- LM dictionary is domain-specific (finance), superior to general-purpose
  dictionaries for financial news text

Reference:
  Loughran, T. & McDonald, B. (2011). When is a liability not a liability?
  Textual analysis, dictionaries, and 10-Ks. Journal of Finance, 66(1), 35-65.
"""

import pandas as pd
import numpy as np
import os
import re
import urllib.request
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("🏷️  STEP 2a: CREATING SENTIMENT LABELS (Loughran-McDonald Dictionary)")
print("=" * 70 + "\n")

TICKERS = ['XLK', 'XLF']

# =============================================================================
# Loughran-McDonald Word Lists
# Core positive and negative word lists from LM (2011)
# Source: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# =============================================================================

LM_POSITIVE = set([
    'able', 'abundance', 'abundant', 'acclaimed', 'accomplish', 'accomplished',
    'achievement', 'acumen', 'adaptable', 'adequate', 'admirable', 'advancement',
    'advantage', 'affirmative', 'affordable', 'agile', 'amazing', 'ambitious',
    'appealing', 'appreciate', 'appreciation', 'appropriate', 'approval',
    'attractive', 'attain', 'attained', 'award', 'beneficially', 'beneficial',
    'benefit', 'benefits', 'best', 'better', 'boom', 'booming', 'breakthrough',
    'brilliant', 'capability', 'capable', 'celebrated', 'clarity', 'clean',
    'clear', 'collaborate', 'comfort', 'commend', 'commendable', 'committed',
    'competitive', 'complement', 'confidence', 'confident', 'constructive',
    'continued', 'creative', 'decisive', 'dependable', 'diligent', 'distinct',
    'diverse', 'dominant', 'dynamic', 'eager', 'earn', 'earnings', 'effective',
    'efficiency', 'efficient', 'empower', 'enhance', 'enriched', 'ethical',
    'exceed', 'exceeded', 'exceptional', 'exciting', 'expanded', 'expansion',
    'favorable', 'feasible', 'flexible', 'flourish', 'flourishing', 'forward',
    'gain', 'gained', 'gains', 'generate', 'good', 'great', 'grow', 'growing',
    'growth', 'high', 'improve', 'improved', 'improvement', 'increasing',
    'innovative', 'integrity', 'lead', 'leadership', 'maximum', 'modern',
    'momentum', 'new', 'optimal', 'outstanding', 'outperform', 'positive',
    'potential', 'profit', 'profitable', 'profitability', 'progress',
    'promising', 'prosper', 'prosperity', 'raise', 'record', 'reliable',
    'resolve', 'reward', 'rise', 'robust', 'strong', 'strength', 'succeed',
    'success', 'successful', 'superior', 'support', 'surge', 'sustainable',
    'upside', 'upgrade', 'value', 'viable', 'win', 'winning',
])

LM_NEGATIVE = set([
    'abnormal', 'abuse', 'adverse', 'against', 'aggravate', 'allegations',
    'alleged', 'antagonize', 'anxiety', 'bankrupt', 'bankruptcy', 'barrier',
    'below', 'breach', 'burden', 'caution', 'collapse', 'complaint',
    'concern', 'conflict', 'contraction', 'controversial', 'crisis',
    'damage', 'decline', 'decreased', 'deficit', 'delay', 'delisted',
    'deteriorate', 'difficult', 'difficulty', 'disappoint', 'disappointing',
    'disappointment', 'dispute', 'disrupt', 'distress', 'doubt', 'downturn',
    'drop', 'dropped', 'excessive', 'fail', 'failed', 'failure', 'fall',
    'falling', 'fear', 'fine', 'force', 'fraud', 'harm', 'hurt', 'impair',
    'impairment', 'inability', 'inadequate', 'ineffective', 'inefficiency',
    'inferior', 'inflation', 'insufficient', 'investigation', 'issue',
    'lack', 'late', 'layoff', 'liability', 'liquidate', 'liquidity',
    'loss', 'losses', 'lower', 'manipulation', 'misconduct', 'miss',
    'missed', 'negative', 'negligent', 'noncompliance', 'obstacle',
    'overextended', 'penalty', 'poor', 'pressure', 'problem', 'recall',
    'reduce', 'reduction', 'resign', 'restrict', 'risk', 'shortage',
    'significant', 'slump', 'substandard', 'suffer', 'suspension',
    'termination', 'threat', 'troubled', 'uncertain', 'uncertainty',
    'underperform', 'unfavorable', 'unprofitable', 'unstable', 'violation',
    'volatile', 'volatility', 'warn', 'warning', 'weakness', 'writedown',
    'writeoff', 'wrong',
])

print(f"📚 Loughran-McDonald Dictionary loaded:")
print(f"   Positive words: {len(LM_POSITIVE)}")
print(f"   Negative words: {len(LM_NEGATIVE)}")
print(f"   Source: Loughran & McDonald (2011), Journal of Finance\n")


def tokenize(text):
    """简单分词：转小写，只保留字母"""
    return re.findall(r'[a-z]+', str(text).lower())


def get_sentiment_label(text):
    """
    用LM词典给文章打标签
    返回: 0 (negative), 1 (neutral), 2 (positive)

    规则:
    - 统计正面词和负面词数量
    - pos > neg → Positive (2)
    - neg > pos → Negative (0)
    - pos == neg (包括都是0) → Neutral (1)
    """
    words = tokenize(text)
    pos_count = sum(1 for w in words if w in LM_POSITIVE)
    neg_count = sum(1 for w in words if w in LM_NEGATIVE)

    if pos_count > neg_count:
        return 2   # Positive
    elif neg_count > pos_count:
        return 0   # Negative
    else:
        return 1   # Neutral


def get_sentiment_score(text):
    """
    返回情感分数 (pos_count - neg_count) / total_words
    用于调试和分析
    """
    words = tokenize(text)
    if len(words) == 0:
        return 0.0
    pos_count = sum(1 for w in words if w in LM_POSITIVE)
    neg_count = sum(1 for w in words if w in LM_NEGATIVE)
    return (pos_count - neg_count) / len(words)


def create_labels_for_sector(ticker):
    print(f"\n{'='*70}")
    print(f"🎯 Processing {ticker}...")
    print(f"{'='*70}")

    # 读取新闻数据
    news_path = f"data/raw_news_{ticker}_holdings.csv"
    if not os.path.exists(news_path):
        news_path = f"raw_news_{ticker}_holdings.csv"
    if not os.path.exists(news_path):
        print(f"   ❌ News file not found: {news_path}")
        return

    df = pd.read_csv(news_path)

    # 找文本列
    text_col = next((c for c in ['Headline', 'Title', 'Snippet', 'snippet',
                                  'headline', 'title'] if c in df.columns), None)
    if not text_col:
        print(f"   ❌ No text column found. Available: {list(df.columns)}")
        return

    print(f"   📊 Total articles: {len(df)}")
    print(f"   📝 Text column: {text_col}")

    # LM词典可以跑全部文章（不需要采样，速度很快）
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    print(f"   🎲 Using {sample_size} articles for classifier training\n")

    # 打标签
    print("   🏷️  Labeling with Loughran-McDonald dictionary...")
    labels = []
    scores = []

    for text in df_sample[text_col]:
        label = get_sentiment_label(str(text))
        score = get_sentiment_score(str(text))
        labels.append(label)
        scores.append(score)

    df_sample['Sentiment_Label'] = labels
    df_sample['LM_Score'] = scores

    # 保存
    if not os.path.exists('data'):
        os.makedirs('data')

    output_path = f"data/sentiment_labels_{ticker}.csv"
    df_sample[[text_col, 'Sentiment_Label', 'LM_Score']].to_csv(
        output_path, index=False)

    print(f"   ✅ Saved: {output_path}")

    # 统计标签分布
    label_counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    print(f"\n   📊 Label Distribution:")
    print(f"      Negative (0): {label_counts.get(0, 0):4d} "
          f"({label_counts.get(0, 0)/total*100:.1f}%)")
    print(f"      Neutral  (1): {label_counts.get(1, 0):4d} "
          f"({label_counts.get(1, 0)/total*100:.1f}%)")
    print(f"      Positive (2): {label_counts.get(2, 0):4d} "
          f"({label_counts.get(2, 0)/total*100:.1f}%)")

    # 质量检查：展示几个例子
    print(f"\n   🔍 Sample Labels (quality check):")
    samples = df_sample.sample(n=min(5, len(df_sample)), random_state=0)
    label_map = {0: 'NEG', 1: 'NEU', 2: 'POS'}
    for _, row in samples.iterrows():
        text_preview = str(row[text_col])[:80]
        print(f"      [{label_map[row['Sentiment_Label']]}] {text_preview}")

    return df_sample


def main():
    print("📌 METHOD: Loughran-McDonald (2011) Financial Sentiment Dictionary")
    print("   • Rule-based, fully transparent — no pre-trained model")
    print("   • Domain-specific: designed for financial text (10-Ks, news)")
    print("   • Labels used to train custom classifiers in Step 2b")
    print("=" * 70)

    results = {}
    for ticker in TICKERS:
        try:
            result = create_labels_for_sector(ticker)
            if result is not None:
                results[ticker] = result
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("✅ LABELING COMPLETE (Loughran-McDonald Dictionary)")
    print(f"{'='*70}")
    print("\n📌 Files created:")
    for ticker in TICKERS:
        print(f"   • data/sentiment_labels_{ticker}.csv")
    print("\n📌 Next Step:")
    print("   Run step2b_train_sentiment_classifier.py to train custom classifiers!")
    print("\n📌 Report Note:")
    print("   Labels generated using Loughran-McDonald (2011) dictionary —")
    print("   a domain-specific financial lexicon, not a pre-trained neural model.")


if __name__ == "__main__":
    main()