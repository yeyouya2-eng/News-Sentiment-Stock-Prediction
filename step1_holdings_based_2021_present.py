"""
STEP 1: Data Collection Based on ETF Holdings (Archive API - Fast Mode)
=======================================================================
核心修改：
1. 【权重更新】按照2025-2026最新数据更新 XLK 和 XLF 前十大持仓。
2. 【提速】改用 NYT Archive API 按月批量下载，不再按天爬取。
3. 【筛选】下载整月数据后，在本地根据公司名筛选，效率提升 100 倍。

老师要求：
"Your contribution is in building the pipeline. Don't just use pre-trained models."
这一步是构建自有数据集的基础。
"""

import time
import pandas as pd
import requests
import datetime
from dateutil.relativedelta import relativedelta
import os
import sys
import json
from tqdm import tqdm

# ================= 配置区域 =================
try:
    import config
    NYT_API_KEY = config.NYT_API_KEY
    print("✅ Loaded NYT_API_KEY from config.py")
except ImportError:
    print("❌ Error: config.py not found!")
    sys.exit(1)

# 时间范围：2021年1月1日 - 昨天
START_DATE = datetime.date(2021, 1, 1)
END_DATE = datetime.date.today() - datetime.timedelta(days=1)

# ================================================================
# [CRITICAL UPDATE] 2025-2026 最新 ETF 持仓权重 (老师反馈)
# ================================================================

XLK_HOLDINGS = {
    # 科技板块：NVDA 超越 AAPL/MSFT 成为权重第一
    'NVIDIA':             ('NVDA', 15.3),
    'Apple':              ('AAPL', 13.3),
    'Microsoft':          ('MSFT', 10.0),
    'Broadcom':           ('AVGO', 5.3),
    'Micron Technology':  ('MU', 4.0),
    'AMD':                ('AMD', 3.0),
    'Cisco':              ('CSCO', 2.8),
    'Palantir':           ('PLTR', 2.6), # 新增
    'Lam Research':       ('LRCX', 2.6), # 新增
    'Oracle':             ('ORCL', 2.3)  # 权重调整
}

XLF_HOLDINGS = {
    # 金融板块：BRK.B 依然稳居第一
    'Berkshire Hathaway': ('BRK.B', 12.1),
    'JPMorgan Chase':     ('JPM', 11.0),
    'Visa':               ('V', 7.3),
    'Mastercard':         ('MA', 5.9),
    'Bank of America':    ('BAC', 4.7),
    'Wells Fargo':        ('WFC', 3.6),
    'Goldman Sachs':      ('GS', 3.5),
    'Morgan Stanley':     ('MS', 2.7),
    'Citigroup':          ('C', 2.6),    # 新增
    'American Express':   ('AXP', 2.5)
}

SECTOR_HOLDINGS = {
    'XLK': XLK_HOLDINGS,
    'XLF': XLF_HOLDINGS
}

# ================= 核心功能函数 =================

def fetch_monthly_archive(year, month):
    """
    使用 Archive API 下载整月所有新闻
    URL: https://api.nytimes.com/svc/archive/v1/{year}/{month}.json
    """
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
    params = {'api-key': NYT_API_KEY}
    
    # 重试机制 (防止网络波动)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                docs = data.get('response', {}).get('docs', [])
                return docs
            elif r.status_code == 429:
                print(f"      ⚠️ Rate limit hit. Waiting 60s... (Attempt {attempt+1})")
                time.sleep(60)
                continue
            else:
                print(f"      ❌ API Error {r.status_code}")
                return None
        except Exception as e:
            print(f"      ❌ Connection error: {e}")
            time.sleep(5)
    return None

def filter_articles_by_holdings(docs, holdings_dict):
    """
    在本地对下载的海量新闻进行筛选
    规则：Headline 或 Snippet 中必须包含持仓公司的名字
    """
    if not docs: return []
    
    filtered_articles = []
    # 预处理：将公司名转为小写，方便匹配
    # 注意：对于 'Visa' 这种词，可能需要更严格的匹配（防止匹配到 visa application），
    # 但为了召回率，先做简单包含匹配，后续清洗再处理。
    target_companies = {name.lower(): code for name, (code, _) in holdings_dict.items()}
    
    # 特殊处理：Visa 和 Apple 这种通用词
    # 可以在这里加正则边界，暂时用简单包含
    
    for doc in docs:
        headline = str(doc.get('headline', {}).get('main', ''))
        snippet = str(doc.get('snippet', ''))
        pub_date = doc.get('pub_date', '')[:10] # YYYY-MM-DD
        
        # 组合文本进行搜索
        full_text = (headline + " " + snippet).lower()
        
        matched_list = []
        for company_name, stock_code in target_companies.items():
            if company_name in full_text:
                matched_list.append(stock_code)
        
        # 如果匹配到了任何一家公司
        if matched_list:
            filtered_articles.append({
                'Date': pub_date,
                'Headline': headline,
                'Snippet': snippet,
                'Matched_Companies': ",".join(matched_list), # 记录是哪家公司的新闻
                'URL': doc.get('web_url', '')
            })
            
    return filtered_articles

def update_sector_data_fast(sector):
    print(f"\n🚀 STARTING FAST SCRAPE FOR: {sector}")
    print(f"   Target: Top 10 Holdings (e.g., {list(SECTOR_HOLDINGS[sector].keys())[:3]}...)")
    
    file_path = f"data/raw_news_{sector}_holdings.csv"
    
    # 1. 检查断点 (从哪个月开始下)
    start_date = START_DATE
    existing_df = pd.DataFrame()
    
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                last_date = existing_df['Date'].max().date()
                # 从最后一个日期的下个月开始
                # 或者简单点：从当前月份重新跑一遍（反正 Archive API 只有几十次请求）
                start_date = last_date.replace(day=1)
                print(f"   📂 Resuming from: {start_date.strftime('%Y-%m')}")
        except:
            pass

    current_date = start_date
    new_articles = []
    
    # 计算总月数用于进度条
    total_months = (END_DATE.year - start_date.year) * 12 + (END_DATE.month - start_date.month) + 1
    
    with tqdm(total=total_months, desc=f"   Downloading {sector}") as pbar:
        while current_date <= END_DATE:
            year = current_date.year
            month = current_date.month
            
            # 1. 下载整月
            docs = fetch_monthly_archive(year, month)
            
            if docs:
                # 2. 本地筛选
                filtered = filter_articles_by_holdings(docs, SECTOR_HOLDINGS[sector])
                new_articles.extend(filtered)
                
                # 更新进度条上的信息
                pbar.set_postfix({'Found': len(filtered)})
            
            # 下个月
            current_date += relativedelta(months=1)
            pbar.update(1)
            
            # Archive API 限制很宽，但还是稍微停顿一下
            time.sleep(1.5)

    # 3. 合并保存
    if new_articles:
        new_df = pd.DataFrame(new_articles)
        
        if not existing_df.empty:
            # 确保列一致
            # 处理日期格式以便合并
            existing_df['Date'] = existing_df['Date'].dt.strftime('%Y-%m-%d')
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
            
        # 去重 (同一天、同一标题)
        before_dedup = len(combined)
        combined.drop_duplicates(subset=['Date', 'Headline'], keep='last', inplace=True)
        
        # 排序
        combined.sort_values('Date', inplace=True)
        
        # 保存
        combined.to_csv(file_path, index=False)
        print(f"\n   ✅ SAVED: {file_path}")
        print(f"   📊 Stats: {len(combined)} total articles (Added {len(new_articles)})")
    else:
        print("\n   ⚠️ No new articles found.")

def main():
    if not os.path.exists('data'):
        os.makedirs('data')
        
    for sector in ['XLK', 'XLF']:
        update_sector_data_fast(sector)

if __name__ == "__main__":
    main()