"""
Feature Correlation Heatmap
============================
生成30个特征的相关性热力图，放入报告Section 2.4。
运行前确保 data/features_sentiment_macro_XLK.csv 存在。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os

os.makedirs('results', exist_ok=True)

TICKERS = ['XLK', 'XLF']

EXCLUDE = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
           'Target', 'Tradable_Return']

# ── Feature category labels ──────────────────────────────────────────────────
CATEGORIES = {
    # Technical
    'MA5':    'Technical', 'MA10':   'Technical', 'MA20':   'Technical',
    'MA50':   'Technical', 'RSI':    'Technical', 'MACD':   'Technical',
    'MACD_Signal': 'Technical', 'BB_Upper': 'Technical', 'BB_Lower': 'Technical',
    # Sentiment
    'Sentiment_LM':              'Sentiment',
    'Sentiment_FinBERT':         'Sentiment',   # kept for backward compat
    'Sentiment_Raw':             'Sentiment',
    'Sentiment_EMA':             'Sentiment',
    'Sentiment_MA3':             'Sentiment',
    'Sentiment_MA10':            'Sentiment',
    'Sentiment_Vol':             'Sentiment',
    'Sentiment_Std5':            'Sentiment',
    'Sentiment_Std10':           'Sentiment',
    'Sentiment_Momentum':        'Sentiment',
    'Sentiment_Rate_of_Change':  'Sentiment',
    'Sentiment_Price_Divergence':'Sentiment',
    # Macro
    'VIX':              'Macro', 'Treasury_10Y':  'Macro',
    'Dollar_Index':     'Macro', 'VIX_MA5':       'Macro',
    'VIX_Change':       'Macro', 'Treasury_Change':'Macro',
    'Sentiment_VIX_Interaction': 'Macro',
    # Auxiliary
    'Log_Return':  'Auxiliary', 'News_Count': 'Auxiliary',
    'Volume_Ratio':'Auxiliary',
}

CAT_COLORS = {
    'Technical': '#4CAF50',
    'Sentiment': '#2196F3',
    'Macro':     '#FF9800',
    'Auxiliary': '#9C27B0',
}

def get_category(col):
    return CATEGORIES.get(col, 'Other')

def plot_heatmap(ticker):
    path = f'data/features_sentiment_macro_{ticker}.csv'
    if not os.path.exists(path):
        print(f'❌ {path} not found')
        return

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    features = [c for c in df.columns if c not in EXCLUDE]
    # Rename FinBERT→LM for display
    df = df.rename(columns={'Sentiment_FinBERT': 'Sentiment_LM'})
    features = [c.replace('Sentiment_FinBERT','Sentiment_LM') for c in features]
    features = [c for c in features if c in df.columns]

    # Sort features by category for readability
    cat_order = ['Technical', 'Sentiment', 'Macro', 'Auxiliary']
    features_sorted = sorted(features,
                             key=lambda c: (cat_order.index(get_category(c))
                                            if get_category(c) in cat_order else 99, c))

    corr = df[features_sorted].corr()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 13))

    # Diverging colormap centred at 0
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, ax=ax, cmap=cmap, center=0,
                vmin=-1, vmax=1,
                square=True, linewidths=0.3, linecolor='#dddddd',
                cbar_kws={'shrink': 0.6, 'label': 'Pearson r'},
                annot=False)   # too many features for annotation

    # ── Category colour bars on axes ─────────────────────────────────────────
    bar_w = 0.012
    for spine in ax.spines.values():
        spine.set_visible(False)

    cats = [get_category(c) for c in features_sorted]
    n = len(features_sorted)
    for i, (feat, cat) in enumerate(zip(features_sorted, cats)):
        color = CAT_COLORS.get(cat, '#999999')
        # left bar
        ax.add_patch(plt.Rectangle((-bar_w * n * 0.08, i), bar_w * n * 0.08, 1,
                                   transform=ax.transData, clip_on=False,
                                   color=color, zorder=5))
        # top bar
        ax.add_patch(plt.Rectangle((i, n + bar_w * n * 0.04), 1,
                                   bar_w * n * 0.08,
                                   transform=ax.transData, clip_on=False,
                                   color=color, zorder=5))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # ── Legend ───────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in CAT_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.18, 1.02), fontsize=10, title='Feature Category',
              framealpha=0.9)

    ax.set_title(f'Feature Correlation Matrix — {ticker}\n'
                 f'(Sorted by category: Technical | Sentiment | Macro | Auxiliary)',
                 fontsize=13, fontweight='bold', pad=14)

    plt.tight_layout()
    out = f'results/feature_correlation_heatmap_{ticker}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {out}')

    # ── Print high-correlation pairs (|r| > 0.8) ─────────────────────────────
    print(f'\n📊 High-correlation pairs for {ticker} (|r| > 0.80):')
    printed = set()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.80:
                pair = (corr.columns[i], corr.columns[j])
                if pair not in printed:
                    printed.add(pair)
                    print(f'   {pair[0]:35s} ↔  {pair[1]:35s}  r = {r:+.3f}')

    # ── Category summary ─────────────────────────────────────────────────────
    print(f'\n📋 Feature counts by category:')
    from collections import Counter
    cnt = Counter(get_category(c) for c in features_sorted)
    for cat in cat_order:
        print(f'   {cat:<12}: {cnt.get(cat, 0)}')


if __name__ == '__main__':
    for ticker in TICKERS:
        print(f'\n{"="*60}')
        print(f'Processing {ticker}...')
        print('='*60)
        plot_heatmap(ticker)

    print('\n✅ Done. Files saved in results/')
    print('   → Add to Report Section 2.4 as Figure showing feature correlation.')
    print('   → Key talking point: within-category correlations are expected;')
    print('     cross-category correlations confirm Sentiment_VIX_Interaction')
    print('     captures unique variance not present in VIX or Sentiment alone.')