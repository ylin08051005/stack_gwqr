"""
data_preparation.py
資料預處理與特徵工程 (包含樓層文字轉數值、113 年過濾、Train/Val/Test 切分、智慧分流標準化)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
import pickle
import glob
import re

sys.path.append('..')
from utils import load_config, create_directories, save_model

EXPERIMENT_LEVEL = 4

# =====================================================================
# 樓層文字轉數值工具集
# =====================================================================
cn_nums = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
    '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25, '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29, '三十': 30,
    '三十一': 31, '三十二': 32, '三十三': 33, '三十四': 34, '三十五': 35, '三十六': 36, '三十七': 37, '三十八': 38, '三十九': 39, '四十': 40
}

def convert_floor(row):
    text = str(row.get('移轉層次', ''))
    total_floor = row.get('總樓層數', '')

    if text == '全':
        if isinstance(total_floor, str):
            match = re.search('[一二三四五六七八九十]+', str(total_floor))
            return cn_nums.get(match.group(), 0) if match else 0
        return total_floor
    
    normal_floors = re.findall(r'(?<!地下)([一二三四五六七八九十]+)層', text)
    num_list = [cn_nums.get(f, 0) for f in normal_floors]
    
    underground_floors = re.findall(r'地下([一二三四五六七八九十]+)層', text)
    num_list += [-cn_nums.get(f, 0) for f in underground_floors]
    
    if num_list:
        return max(num_list)
    
    if '地下' in text:
        return -1
        
    return 0


def get_features_by_level(level):
    base_features = ['建物移轉總面積平方公尺', 'house_age', 'count_floor', 'time_index']
    f_building = ['建物現況格局-廳', '建物現況格局-衛', '電梯']
    f_socio = ['MASTER_UP_rate', 'Sex_ratio', 'popdense']
    f_env = ['dist_to_rail_meter', 'dist_to_mrt_meter', 'count_med_500m', 'count_phar_500m']
    f_weather = ['aqi_idw', 't_mean_24h_idw', 'daily_rain_idw']
    
    if level == 1:
        return base_features + f_building, "Model_1_Building"
    elif level == 2:
        return base_features + f_building + f_socio, "Model_2_Socio"
    elif level == 3:
        return base_features + f_building + f_socio + f_env, "Model_3_Env"
    elif level == 4:
        return base_features + f_building + f_socio + f_env + f_weather, "Model_4_Weather"
    else:
        raise ValueError("EXPERIMENT_LEVEL 必須設定為 1~4 之間")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'config.yaml')
    
    config = load_config(config_path)
    create_directories(config)
    np.random.seed(config['project']['random_seed'])
    
    print("=" * 60)
    print("資料準備與跨期合併階段 (僅限 113 年資料，學術嚴謹三段式切分)")
    print("=" * 60)
    
    features, exp_name = get_features_by_level(EXPERIMENT_LEVEL)
    target = config['data']['target'] 
    
    # 1. 自動抓取並合併 113 年度的 CSV
    raw_dir = os.path.join(config['paths']['data_dir'], 'raw')
    search_pattern = os.path.join(raw_dir, 'realprice_with_varcount_idw*.csv')
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"\n❌ 錯誤: 在 {raw_dir} 找不到符合格式的檔案！")
        return
        
    time_mapping = {
        '1131': 1, 
        '1132': 2, 
        '1133': 3, 
        '1134': 4
    }
    
    df_list = []
    print("\n開始載入與合併各期資料 (過濾 113 年)...")
    for file in all_files:
        filename = os.path.basename(file)
        match = re.search(r'113[1-4]', filename) 
        if match:
            quarter_key = match.group()
            tmp_df = pd.read_csv(file)
            tmp_df['time_index'] = time_mapping.get(quarter_key, 0)
            df_list.append(tmp_df)
            print(f"  - 成功載入 {filename} ({len(tmp_df)} 筆)")
            
    if not df_list:
        print(f"\n❌ 錯誤: 在資料夾中找不到任何 113 年的 CSV 檔案！")
        return

    df = pd.concat(df_list, ignore_index=True)
    
    # 2. 資料清洗與時間過濾
    print("\n執行資料清洗與時間過濾...")
    ghost_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    df.drop(columns=ghost_cols, inplace=True, errors='ignore')
    
    priority_cols = ['土地位置建物門牌', '緯度', '經度', 'time_index', target]
    priority_cols = [c for c in priority_cols if c in df.columns]
    df = df.dropna(subset=priority_cols)
    
    df = df[df['time_index'].between(1, 4)].copy()
    df.reset_index(drop=True, inplace=True)

    # =====================================================================
    # 2.5 執行樓層文字轉數值 (新增 count_floor)
    # =====================================================================
    print("\n執行樓層文字轉換 (產生 count_floor)...")
    df['count_floor'] = df.apply(convert_floor, axis=1)
    
    cols = df.columns.tolist()
    if '移轉層次' in cols:
        idx = cols.index('移轉層次')
        new_cols = cols[:idx+1] + ['count_floor'] + [c for c in cols[idx+1:] if c != 'count_floor']
        df = df[new_cols]

    combined_raw_path = os.path.join(raw_dir, 'realprice_combined_113_all.csv')
    df.to_csv(combined_raw_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 排序並清洗完成的原始資料 (含 count_floor) 已儲存至: {combined_raw_path}")
    
    # 3. 特徵處理 (補值、對數轉換)
    for col in features:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    print(f"\n執行目標變數 ({target}) Log 轉換...")
    df[target] = np.log1p(df[target])
    
    X = df[features].copy()
    y = df[target].copy().values
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"偵測到類別變數: {categorical_cols}，執行 One-hot encoding...")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # =====================================================================
    # 4. ★ 三段式時間序列切分 (僅 113 年) ★
    # =====================================================================
    print("\n執行學術級三段式切分 (113年度):")
    
    train_mask = df['time_index'] <= 2
    val_mask   = df['time_index'] == 3
    test_mask  = df['time_index'] == 4
    
    X_train, y_train = X[train_mask].copy(), y[train_mask]
    X_val, y_val     = X[val_mask].copy(), y[val_mask]
    X_test, y_test   = X[test_mask].copy(), y[test_mask]
    
    print(f"  - [Train] 第 1~2 期 (1131-1132): {X_train.shape[0]} 筆")
    print(f"  - [Val]   第 3 期   (1133)     : {X_val.shape[0]} 筆 (調參用)")
    print(f"  - [Test]  第 4 期   (1134)     : {X_test.shape[0]} 筆 (盲測用)")

   # =====================================================================
    # 5. 特徵標準化與儲存 (★ 學術嚴謹版：避開二元變數)
    # =====================================================================
    print("\n執行特徵標準化 (智慧分流：保留虛擬變數 0/1 原貌)...")
    
    # 自動偵測二元變數 (包含 get_dummies 產生的 True/False 或 1/0)
    binary_cols = []
    for col in X_train.columns:
        unique_vals = set(X_train[col].dropna().unique())
        # 如果該欄位的值只包含 0, 1, True, False 的組合，就判定為二元變數
        if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            binary_cols.append(col)
            
    continuous_cols = [col for col in X_train.columns if col not in binary_cols]
    
    print(f"  - 偵測到 {len(binary_cols)} 個二元變數 (不進行標準化，保持原狀)")
    print(f"  - 偵測到 {len(continuous_cols)} 個連續變數 (進行 Z-score 標準化)")

    scaler = StandardScaler()
    
    # 先複製一份原始資料
    X_train_scaled = X_train.copy()
    X_val_scaled   = X_val.copy()
    X_test_scaled  = X_test.copy()
    
    # 只針對連續變數進行 fit 與 transform
    if continuous_cols:
        X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
        X_val_scaled[continuous_cols]   = scaler.transform(X_val[continuous_cols])
        X_test_scaled[continuous_cols]  = scaler.transform(X_test[continuous_cols])
        
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    
    # 儲存個別檔案
    X_train_scaled.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_val_scaled.to_csv(os.path.join(processed_dir, 'X_val.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    
    # 儲存索引對照
    X_train.index.to_series().to_csv(os.path.join(processed_dir, 'train_index.csv'), index=False)
    X_val.index.to_series().to_csv(os.path.join(processed_dir, 'val_index.csv'), index=False)
    X_test.index.to_series().to_csv(os.path.join(processed_dir, 'test_index.csv'), index=False)
    
    # 儲存 scaler 模型供未來使用 (只包含連續變數的轉換規則)
    save_model(scaler, os.path.join(config['paths']['model_dir'], 'scaler.pkl'))
    
    print("\n" + "=" * 60)
    print(f"🎉 113年度三段式切分與標準化完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()