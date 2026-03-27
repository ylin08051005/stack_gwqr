import pandas as pd
import glob
import os

# 抓取所有 113 年的原始檔案
files = glob.glob('./data/raw/realprice_with_varcount_idw113*.csv')
target_cols = ['土地位置建物門牌', '緯度', '經度', '總價元']

print("="*60)
print(" 🕵️‍♂️ 欄位名稱與空值偵測器")
print("="*60)

for file in files:
    df = pd.read_csv(file)
    print(f"\n📁 檔案: {os.path.basename(file)} (總筆數: {len(df)})")
    
    for col in target_cols:
        # 1. 完全命中
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing == len(df):
                print(f"  ⚠️ 找到 [{col}]，但裡面是全空的 (NaN)！")
            else:
                print(f"  ✅ 找到 [{col}] (空值: {missing} 筆)")
        
        # 2. 找不到，啟動相似詞猜測
        else:
            print(f"  ❌ 找不到精確的 [{col}]！")
            
            # 找找看有沒有混入空白，或名稱相近的
            similar_cols = []
            for c in df.columns:
                if (col in str(c)) or (str(c) in col) or ('門牌' in str(c)) or ('價' in str(c)):
                    similar_cols.append(f"'{c}'")
                    
            if similar_cols:
                print(f"     👉 電腦懷疑其實是這個欄位: {', '.join(similar_cols)}")