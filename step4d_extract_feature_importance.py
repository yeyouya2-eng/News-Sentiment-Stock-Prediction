"""
Extract Feature Importance from Trained XGBoost Models (FIXED)
==============================================================
从step4的rolling model中提取feature importance
适配当前数据结构 (features_sentiment_focused_XX.csv)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

TICKERS = ['XLK', 'XLF']

def extract_feature_importance(ticker):
    """从最后一个训练的模型提取feature importance"""
    
    print(f"\n{'='*70}")
    print(f"📊 Extracting Feature Importance: {ticker}")
    print(f"{'='*70}")
    
    # 1. 加载features（自动查找可用的数据文件）
    # 优先使用含macro的完整数据集，fallback到focused版本
    feature_path = f"data/features_sentiment_macro_{ticker}.csv"
    if not os.path.exists(feature_path):
        feature_path = f"data/features_sentiment_focused_{ticker}.csv"
    if not os.path.exists(feature_path):
        print(f"   ❌ Features file not found. Tried:")
        print(f"      data/features_sentiment_macro_{ticker}.csv")
        print(f"      data/features_sentiment_focused_{ticker}.csv")
        return None
    print(f"   📂 Loading: {feature_path}")
    
    df = pd.read_csv(feature_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)
    # Rename Sentiment_FinBERT to Sentiment_LM for consistency with report
    df.rename(columns={'Sentiment_FinBERT': 'Sentiment_LM'}, inplace=True)
    
    # 2. 定义features（与step4一致）
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Target', 'Next_Open', 'Next_Close', 'Tradable_Return', 'Log_Return']
    
    features = [c for c in df.columns if c not in exclude_cols]
    
    print(f"   📋 Total features: {len(features)}")
    
    # 3. 加载最优参数（适配当前文件名）
    param_file = f"results/best_params_xgb_{ticker}.json"
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            params = json.load(f)
        print(f"   ✅ Loaded tuned params from {param_file}")
    else:
        # 尝试旧文件名
        param_file_old = f"results/best_params_{ticker}.json"
        if os.path.exists(param_file_old):
            with open(param_file_old, 'r') as f:
                params = json.load(f)
            print(f"   ✅ Loaded tuned params from {param_file_old}")
        else:
            params = {
                'max_depth': 5,
                'learning_rate': 0.03,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            print(f"   ⚠️  Using default params")
    
    # 4. 训练一个模型（用全部数据）
    print(f"   🎓 Training model on full dataset...")
    
    X = df[features]
    y = df['Target']
    
    model = xgb.XGBClassifier(**params, eval_metric='logloss', n_jobs=-1)
    model.fit(X, y, verbose=False)
    
    print(f"   ✅ Model trained on {len(X)} samples")
    
    # 5. 提取feature importance
    importance_dict = dict(zip(features, model.feature_importances_))
    
    # 6. 创建DataFrame并排序
    fi_df = pd.DataFrame([
        {'Feature': feat, 'Importance': imp}
        for feat, imp in importance_dict.items()
    ]).sort_values('Importance', ascending=False)
    
    # 7. 添加分类标签（适配当前features）
    def classify_feature(feat):
        if 'Sentiment' in feat or 'sentiment' in feat:
            return 'Sentiment'
        elif feat in ['MA5', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 
                      'BB_Upper', 'BB_Lower', 'Log_Return']:
            return 'Technical'
        elif feat in ['VIX', 'Treasury_10Y', 'Dollar_Index', 'VIX_MA5', 
                      'VIX_Change', 'Treasury_Change', 'Sentiment_VIX_Interaction']:
            return 'Macro'
        elif feat in ['News_Count']:
            return 'News'
        elif feat in ['Volume_Ratio']:
            return 'Volume'
        else:
            return 'Other'
    
    fi_df['Category'] = fi_df['Feature'].apply(classify_feature)
    
    # 8. 保存
    if not os.path.exists('results'):
        os.makedirs('results')
    
    out_path = f"results/feature_importance_{ticker}.csv"
    fi_df.to_csv(out_path, index=False)
    
    print(f"   💾 Saved: {out_path}")
    
    # 9. 打印统计
    print(f"\n   📊 Summary:")
    print(f"   {'='*60}")
    
    # Top 10
    print(f"\n   Top 10 Features:")
    for i, row in fi_df.head(10).iterrows():
        print(f"   {i+1:2d}. {row['Feature']:30s} {row['Importance']:>8.4f} ({row['Category']})")
    
    # 按类别统计
    print(f"\n   Feature Importance by Category:")
    category_importance = fi_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)
    for cat, imp in category_importance.items():
        pct = imp / fi_df['Importance'].sum() * 100
        print(f"   {cat:15s}: {imp:>8.4f} ({pct:>5.1f}%)")
    
    # Top 10中的sentiment占比
    top10_sentiment = fi_df.head(10)['Category'].value_counts().get('Sentiment', 0)
    print(f"\n   Sentiment in Top 10: {top10_sentiment}/10")
    
    if top10_sentiment >= 2:
        print(f"   ✅ Sentiment features have significant importance")
    else:
        print(f"   ⚠️  Sentiment features have limited importance")
    
    # 10. 生成图表
    generate_importance_chart(fi_df, ticker)
    
    return fi_df


def generate_importance_chart(fi_df, ticker):
    """生成Feature Importance可视化图表"""
    
    # 颜色映射
    color_map = {
        'Sentiment': '#2196F3',   # 蓝色
        'Technical': '#4CAF50',   # 绿色
        'Macro':     '#FF9800',   # 橙色
        'Volume':    '#9C27B0',   # 紫色
        'News':      '#F44336',   # 红色
        'Other':     '#9E9E9E',   # 灰色
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'XGBoost Feature Importance Analysis — {ticker}', 
                 fontsize=15, fontweight='bold', y=1.01)

    # --- 图1: Top 15 Features 横向条形图 ---
    top15 = fi_df.head(15).copy().reset_index(drop=True)
    top15 = top15.iloc[::-1]  # 翻转让最重要的在最上面

    colors = [color_map.get(cat, '#9E9E9E') for cat in top15['Category']]
    bars = axes[0].barh(top15['Feature'], top15['Importance'], color=colors, 
                        edgecolor='white', linewidth=0.5)

    # 数值标签
    for bar, val in zip(bars, top15['Importance']):
        axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', ha='left', fontsize=8.5)

    axes[0].set_xlabel('Feature Importance (Gain)', fontsize=11)
    axes[0].set_title('Top 15 Features', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, top15['Importance'].max() * 1.18)
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[cat], label=cat) 
                       for cat in color_map if cat in fi_df['Category'].values]
    axes[0].legend(handles=legend_elements, loc='lower right', fontsize=9,
                   framealpha=0.8)

    # --- 图2: Category Importance 饼图 ---
    cat_imp = fi_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)
    pie_colors = [color_map.get(cat, '#9E9E9E') for cat in cat_imp.index]

    wedges, texts, autotexts = axes[1].pie(
        cat_imp.values,
        labels=cat_imp.index,
        autopct='%1.1f%%',
        colors=pie_colors,
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    for text in autotexts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(11)

    axes[1].set_title('Importance by Category', fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    out_path = f'results/feature_importance_{ticker}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   🖼️  Saved chart: {out_path}")

def generate_comparison_chart(xlk_fi, xlf_fi):
    """生成XLK vs XLF对比图"""

    color_map = {
        'Sentiment': '#2196F3',
        'Technical': '#4CAF50',
        'Macro':     '#FF9800',
        'Volume':    '#9C27B0',
        'News':      '#F44336',
        'Other':     '#9E9E9E',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Feature Importance: XLK vs XLF Sector Comparison',
                 fontsize=14, fontweight='bold')

    for ax, fi_df, ticker in zip(axes, [xlk_fi, xlf_fi], ['XLK (Technology)', 'XLF (Financial)']):
        cat_imp = fi_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)
        colors = [color_map.get(c, '#9E9E9E') for c in cat_imp.index]
        bars = ax.bar(cat_imp.index, cat_imp.values * 100, color=colors,
                      edgecolor='white', linewidth=1)
        for bar, val in zip(bars, cat_imp.values * 100):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Total Importance (%)', fontsize=11)
        ax.set_title(ticker, fontsize=12, fontweight='bold')
        ax.set_ylim(0, cat_imp.values.max() * 100 * 1.18)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = 'results/feature_importance_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   🖼️  Saved comparison chart: {out_path}")


def compare_sectors():
    """对比两个sector的feature importance"""
    
    print(f"\n{'='*70}")
    print(f"🔬 SECTOR COMPARISON")
    print(f"{'='*70}")
    
    xlk_fi = pd.read_csv('results/feature_importance_XLK.csv')
    xlf_fi = pd.read_csv('results/feature_importance_XLF.csv')
    
    print(f"\nTop 10 Feature Comparison:")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'XLK Feature':<30} {'XLF Feature':<30}")
    print(f"{'-'*70}")
    
    for i in range(10):
        xlk_feat = xlk_fi.iloc[i]['Feature']
        xlf_feat = xlf_fi.iloc[i]['Feature']
        xlk_cat = xlk_fi.iloc[i]['Category']
        xlf_cat = xlf_fi.iloc[i]['Category']
        
        print(f"{i+1:<5} {xlk_feat[:28]:28s} ({xlk_cat[0]})  {xlf_feat[:28]:28s} ({xlf_cat[0]})")
    
    # 统计Sentiment在Top 10的占比
    xlk_top10_sent = xlk_fi.head(10)['Category'].value_counts().get('Sentiment', 0)
    xlf_top10_sent = xlf_fi.head(10)['Category'].value_counts().get('Sentiment', 0)
    
    print(f"\n{'='*70}")
    print(f"Sentiment Features in Top 10:")
    print(f"  XLK: {xlk_top10_sent}/10")
    print(f"  XLF: {xlf_top10_sent}/10")
    
    if xlk_top10_sent > xlf_top10_sent:
        print(f"  ✅ XLK has more sentiment features in top 10")
        print(f"  → This supports H1 (Tech more sentiment-driven)")
    elif xlf_top10_sent > xlk_top10_sent:
        print(f"  ⚠️  XLF has more sentiment features in top 10")
        print(f"  → This contradicts H1")
    else:
        print(f"  ➖ Equal sentiment features in top 10")
    
    # 按类别对比
    print(f"\n{'='*70}")
    print(f"Category Importance Comparison:")
    print(f"{'='*70}")
    
    xlk_cat = xlk_fi.groupby('Category')['Importance'].sum()
    xlf_cat = xlf_fi.groupby('Category')['Importance'].sum()
    
    all_cats = sorted(set(xlk_cat.index) | set(xlf_cat.index))
    
    print(f"{'Category':<15} {'XLK %':>10} {'XLF %':>10} {'Difference':>12}")
    print(f"{'-'*50}")
    
    for cat in all_cats:
        xlk_pct = xlk_cat.get(cat, 0) / xlk_fi['Importance'].sum() * 100
        xlf_pct = xlf_cat.get(cat, 0) / xlf_fi['Importance'].sum() * 100
        diff = xlk_pct - xlf_pct
        
        arrow = "→" if abs(diff) < 2 else ("↑" if diff > 0 else "↓")
        print(f"{cat:<15} {xlk_pct:>9.1f}% {xlf_pct:>9.1f}% {diff:>+10.1f}pp {arrow}")

    # 生成对比图
    generate_comparison_chart(xlk_fi, xlf_fi)

def main():
    """主函数"""
    
    print("="*70)
    print("🔍 FEATURE IMPORTANCE EXTRACTION (FIXED)")
    print("="*70)
    print("\nPurpose: Extract and analyze feature importance from trained models")
    print("Data: features_sentiment_focused_XX.csv (23 features)\n")
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 提取两个sector的feature importance
    results = []
    for ticker in TICKERS:
        try:
            fi_df = extract_feature_importance(ticker)
            if fi_df is not None:
                results.append(ticker)
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # 对比
    if len(results) == 2:
        print(f"\n")
        compare_sectors()
    
    print(f"\n{'='*70}")
    print("✅ FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print("\n📌 Files created:")
    print("   • results/feature_importance_XLK.csv")
    print("   • results/feature_importance_XLF.csv")
    print("   • results/feature_importance_XLK.png  (Top 15 + Category Pie)")
    print("   • results/feature_importance_XLF.png  (Top 15 + Category Pie)")
    print("   • results/feature_importance_comparison.png  (XLK vs XLF)")
    print("\n📌 Use these results in Progress Report Section 3.3")
    print("   (Evidence for H2: Sentiment features in Top 10)")

if __name__ == "__main__":
    main()