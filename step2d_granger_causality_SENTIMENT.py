"""
STEP 2bb: Granger Causality Analysis (Sentiment Focused)
==========================================================
验证 News Sentiment 是否是 Stock Returns 的"领先指标"

当前数据结构：features_sentiment_focused_XX.csv
测试对象：11个Sentiment features

目的：
• 检验sentiment是否Granger-cause price movements
• 对比XLK vs XLF的因果关系强度
• 支持H1（sector divergence）
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("🔬 GRANGER CAUSALITY: Sentiment → Price")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']
MAX_LAG = 5  # 测试滞后 1-5 天

def load_data(ticker):
    """
    加载当前数据结构
    
    数据：features_sentiment_focused_XX.csv
    特征：23 features (9 technical + 11 sentiment + 3 other)
    """
    data_path = f"data/features_sentiment_macro_{ticker}.csv"
    
    if not os.path.exists(data_path):
        print(f"   ❌ Data not found: {data_path}")
        print(f"   💡 Make sure you have run step2c to generate features")
        return None, []
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)
    
    # 找到所有sentiment features
    sentiment_features = [c for c in df.columns if 'Sentiment' in c or 'sentiment' in c]
    
    if len(sentiment_features) == 0:
        print(f"   ❌ No sentiment features found!")
        print(f"   Available columns: {df.columns.tolist()}")
        return None, []
    
    print(f"   ✅ Loaded {len(df)} days of data")
    print(f"   📊 Found {len(sentiment_features)} sentiment features")
    
    return df, sentiment_features

def run_granger_test(ticker):
    """运行Granger causality测试"""
    
    print(f"\n{'='*60}")
    print(f"🎯 ANALYZING: {ticker}")
    print(f"{'='*60}")
    
    df, sentiment_features = load_data(ticker)
    if df is None or not sentiment_features:
        return []
    
    print(f"\n   📊 Testing Sentiment Features:")
    for i, feat in enumerate(sentiment_features, 1):
        print(f"      {i}. {feat}")
    
    significant_findings = []
    
    # 对每个 Sentiment Feature 单独测试
    for sentiment_feat in sentiment_features:
        
        # 准备数据：[Log_Return, Sentiment_Feature]
        # Granger test format: 第二列是否cause第一列
        if 'Log_Return' not in df.columns:
            print(f"\n   ❌ Log_Return not found in data!")
            return []
        
        test_data = df[['Log_Return', sentiment_feat]].dropna()
        
        if len(test_data) < 50:  # 至少需要50个观测
            print(f"      ⚠️  Insufficient data for {sentiment_feat}")
            continue
        
        try:
            # 运行Granger causality test
            # H0: sentiment does NOT Granger-cause returns
            # H1: sentiment Granger-causes returns
            res = grangercausalitytests(test_data.values, maxlag=MAX_LAG, verbose=False)
            
            # 检查 Lags 1-5 是否有显著的 (p < 0.05)
            is_significant = False
            best_p = 1.0
            best_lag = 0
            
            for lag in range(1, MAX_LAG + 1):
                # 获取 F-test 的 p-value (ssr_ftest)
                # res[lag][0] 是一个字典，包含多个test结果
                p_value = res[lag][0]['ssr_ftest'][1]
                
                if p_value < 0.05:
                    is_significant = True
                    if p_value < best_p:
                        best_p = p_value
                        best_lag = lag
            
            if is_significant:
                print(f"      ✅ {sentiment_feat}: CAUSES Price (Lag {best_lag}, p={best_p:.4f})")
                significant_findings.append({
                    'Ticker': ticker,
                    'Feature': sentiment_feat,
                    'Best_Lag': best_lag,
                    'P_Value': best_p,
                    'Significant': True
                })
            else:
                # 不显著的不输出（太多会很乱）
                pass
                
        except Exception as e:
            print(f"      ❌ Error testing {sentiment_feat}: {e}")
    
    # 总结
    if significant_findings:
        print(f"\n   ✅ Found {len(significant_findings)} significant causal relationships")
    else:
        print(f"\n   ⚠️  No significant causal relationships found")
    
    return significant_findings

def create_visualization(df_results):
    """创建可视化"""
    
    if len(df_results) == 0:
        print("   ⚠️  No results to visualize")
        return
    
    # 按ticker分组
    xlk_count = len(df_results[df_results['Ticker'] == 'XLK'])
    xlf_count = len(df_results[df_results['Ticker'] == 'XLF'])
    
    # 创建图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1: Count comparison
    tickers = ['XLK\n(Tech)', 'XLF\n(Finance)']
    counts = [xlk_count, xlf_count]
    colors = ['#2ca02c', '#ff7f0e']
    
    bars = axes[0].bar(tickers, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Number of Significant Features', fontsize=12, fontweight='bold')
    axes[0].set_title('Granger Causality: Sentiment → Price', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{count}',
                     ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # 图2: P-value distribution
    if len(df_results) > 0:
        # 按ticker着色
        xlk_data = df_results[df_results['Ticker'] == 'XLK']
        xlf_data = df_results[df_results['Ticker'] == 'XLF']
        
        if len(xlk_data) > 0:
            axes[1].scatter(xlk_data['Best_Lag'], xlk_data['P_Value'], 
                          s=100, alpha=0.7, color='#2ca02c', label='XLK', edgecolor='black')
        
        if len(xlf_data) > 0:
            axes[1].scatter(xlf_data['Best_Lag'], xlf_data['P_Value'], 
                          s=100, alpha=0.7, color='#ff7f0e', label='XLF', edgecolor='black')
        
        axes[1].axhline(0.05, color='red', linestyle='--', linewidth=2, label='Significance (p=0.05)')
        axes[1].set_xlabel('Optimal Lag (days)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('P-value', fontsize=12, fontweight='bold')
        axes[1].set_title('Granger Causality P-values', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 0.06])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    out_path = 'results/granger_causality_sentiment.png'
    import os as _os
    if _os.path.exists(out_path):
        _os.remove(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n   🖼️  Saved: results/granger_causality_sentiment.png")

def main():
    """主函数"""
    
    print("📌 GRANGER CAUSALITY ANALYSIS\n")
    print("Test: Does sentiment Granger-cause price movements?\n")
    print("H0: Sentiment does NOT lead price")
    print("H1: Sentiment leads price (causal relationship)\n")
    
    all_results = []
    
    for ticker in TICKERS:
        try:
            results = run_granger_test(ticker)
            all_results.extend(results)
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    print("\n" + "="*70)
    print("🔬 GRANGER CAUSALITY SUMMARY")
    print("="*70)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        print("\n" + df_results.to_string(index=False))
        
        # 保存
        if not os.path.exists('results'):
            os.makedirs('results')
        
        df_results.to_csv('results/granger_causality_sentiment.csv', index=False)
        print(f"\n✅ Saved: results/granger_causality_sentiment.csv")
        
        # 统计
        xlk_sig = len(df_results[df_results['Ticker'] == 'XLK'])
        xlf_sig = len(df_results[df_results['Ticker'] == 'XLF'])
        
        print(f"\n📊 Sector Comparison:")
        print(f"   • XLK: {xlk_sig} significant sentiment features")
        print(f"   • XLF: {xlf_sig} significant sentiment features")
        
        # H1解释
        print(f"\n💡 Interpretation for H1:")
        if xlk_sig > xlf_sig:
            print(f"   ✅ SUPPORTS H1")
            print(f"      XLK shows MORE causal relationships ({xlk_sig} vs {xlf_sig})")
            print(f"      → Tech sector is more sensitive to news sentiment")
        elif xlk_sig < xlf_sig:
            print(f"   ⚠️  CONTRADICTS H1")
            print(f"      XLF shows MORE causal relationships ({xlf_sig} vs {xlk_sig})")
            print(f"      → Finance sector appears more sentiment-driven (unexpected)")
        else:
            print(f"   ➖ NEUTRAL")
            print(f"      Both sectors show equal causal relationships ({xlk_sig})")
        
        # 可视化
        print(f"\n📊 Creating Visualization...")
        create_visualization(df_results)
        
    else:
        print("\n❌ No significant Granger causality found in any sector")
        print("\nPossible reasons:")
        print("   • Sentiment may not lead price (market efficiency)")
        print("   • Lag period (1-5 days) may be too short/long")
        print("   • Sample size may be insufficient")
        print("   • News sentiment already reflected in prices")
    
    print("\n" + "="*70)
    print("✅ GRANGER CAUSALITY ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📌 Next Steps:")
    print("   1. Check results/granger_causality_sentiment.csv")
    print("   2. Check results/granger_causality_sentiment.png")
    print("   3. Add to Progress Report Section 3.0 (before main results)")
    print("   4. Use as supporting evidence for H1")

if __name__ == "__main__":
    main()