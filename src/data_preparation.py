"""
資料預處理 (70% Train / 30% Test + Train 5-Fold)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import sys
import os
import glob
import re

sys.path.append('..')
from utils import load_config, create_directories, save_model

EXPERIMENT_LEVEL = 3

TARGET_YEAR = '112' 

# 樓層文字轉數值工具集
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
    f_socio = ['BACHELOR_UP_rate', 'Sex_ratio', 'Dependency_ratio', 'popdense']
    f_env = ['dist_to_rail_meter', 'dist_to_mrt_meter', 'count_med_500m', 'count_phar_500m']
    f_weather = ['aqi_idw', 't_max_idw', 'daily_rain_idw']
    
    if level == 1:
        return base_features + f_building, "Model_1_Building"
    elif level == 2:
        return base_features + f_building + f_socio, "Model_2_Socio"
    elif level == 3:
        return base_features + f_building + f_socio + f_env, "Model_3_Env"
    elif level == 4:
        return base_features + f_building + f_socio + f_env + f_weather, "Model_4_Weather"
    else:
        raise ValueError("EXPERIMENT_LEVEL 錯誤")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'config.yaml')
    
    config = load_config(config_path)
    create_directories(config)
    seed = config['project']['random_seed']
    np.random.seed(seed)
    
    features, exp_name = get_features_by_level(EXPERIMENT_LEVEL)
    target = config['data']['target'] 
    
    raw_dir = os.path.join(config['paths']['data_dir'], 'raw')
    search_pattern = os.path.join(raw_dir, 'realprice_with_varcount_idw*.csv')
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"\n 在 {raw_dir} 找不到符合格式的檔案")
        return
        
    time_mapping = {
        f'{TARGET_YEAR}1': 1, 
        f'{TARGET_YEAR}2': 2, 
        f'{TARGET_YEAR}3': 3, 
        f'{TARGET_YEAR}4': 4
    }
    
    df_list = []
    print(f"\n開始載入與合併{TARGET_YEAR} 年各期資料")
    for file in all_files:
        filename = os.path.basename(file)
        match = re.search(fr'{TARGET_YEAR}[1-4]', filename) 
        if match:
            quarter_key = match.group()
            tmp_df = pd.read_csv(file)
            
            rename_map = {}
            for c in tmp_df.columns:
                c_lower = str(c).lower().strip()
                if c_lower in ['lat', 'latitude']:
                    rename_map[c] = '緯度'
                elif c_lower in ['lng', 'longitude', 'lon']:
                    rename_map[c] = '經度'
            
            if rename_map:
                tmp_df = tmp_df.rename(columns=rename_map)

            tmp_df['time_index'] = time_mapping.get(quarter_key, 0)
            df_list.append(tmp_df)
            print(f"成功載入 {filename} ({len(tmp_df)} 筆)")
            
    if not df_list:
        print(f"\n 在資料夾中找不到任何 {TARGET_YEAR} 年的 CSV 檔案")
        return

    df = pd.concat(df_list, ignore_index=True)
    
    ghost_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    df.drop(columns=ghost_cols, inplace=True, errors='ignore')
    
    priority_cols = ['土地位置建物門牌', '緯度', '經度', 'time_index', target]
    priority_cols = [c for c in priority_cols if c in df.columns]
    df = df.dropna(subset=priority_cols)
    
    df = df[df['time_index'].between(1, 4)].copy()
    df.reset_index(drop=True, inplace=True)

    print("\n 產生 count_floor")
    df['count_floor'] = df.apply(convert_floor, axis=1)
    
    cols = df.columns.tolist()
    if '移轉層次' in cols:
        idx = cols.index('移轉層次')
        new_cols = cols[:idx+1] + ['count_floor'] + [c for c in cols[idx+1:] if c != 'count_floor']
        df = df[new_cols]

    combined_raw_path = os.path.join(raw_dir, f'realprice_combined_{TARGET_YEAR}_all.csv')
    df.to_csv(combined_raw_path, index=False, encoding='utf-8-sig')
    print(f"清洗完成的原始資料已儲存至: {combined_raw_path}")
    print(f"合併後總筆數為: {len(df)} 筆")
    
    for col in features:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    print(f"\n執行目標變數 ({target}) Log 轉換")
    df[target] = np.log1p(df[target])
    
    X = df[features].copy()
    y = df[target].copy().values
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"偵測到類別變數: {categorical_cols}，做 One-hot encoding...")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    print(f"\n執行 70% Train / 30% Test 切分與 Train 內部 5-Fold :")

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=seed)
    
    # 建立 fold_labels 陣列，預設全為 -1 (代表這 30% 是完全獨立的 Test Set)
    fold_labels = np.full(len(X), -1, dtype=int)
    
    # 只針對 70% 的 Train Data 5-Fold 切分
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_idx)):
        real_val_idx = train_idx[val_idx]
        fold_labels[real_val_idx] = fold_idx
        print(f" Train Fold {fold_idx}: 包含 {len(real_val_idx)} 筆驗證資料")
        
    print(f" 保留 30% Test Set: 包含 {len(test_idx)} 筆盲測資料 (標記為 Fold -1)")
    
    X['fold'] = fold_labels 
    
    binary_cols = ['fold'] 
    for col in X.columns:
        if col == 'fold': 
            continue
        unique_vals = set(X[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            binary_cols.append(col)
            
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    
    print(f" 偵測到 {len(binary_cols)-1} 個二元變數 + 1 個 Fold 標籤 (不進行標準化)")
    print(f" 偵測到 {len(continuous_cols)} 個連續變數 (進行標準化)")

    scaler = StandardScaler()
    X_scaled = X.copy()
    
    if continuous_cols:
        train_mask = X['fold'] != -1
        # Train Data (fold 0~4)用fit_transform
        X_scaled.loc[train_mask, continuous_cols] = scaler.fit_transform(X.loc[train_mask, continuous_cols])
        # Test Data (fold -1)只能用transform
        X_scaled.loc[~train_mask, continuous_cols] = scaler.transform(X.loc[~train_mask, continuous_cols])
        
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    
    X_scaled.to_csv(os.path.join(processed_dir, 'X_all.csv'), index=False)
    np.save(os.path.join(processed_dir, 'y_all.npy'), y)
    X.index.to_series().to_csv(os.path.join(processed_dir, 'index_all.csv'), index=False)
    save_model(scaler, os.path.join(config['paths']['model_dir'], 'scaler.pkl'))
    
    print("\n" + "=" * 60)
    print(f"{TARGET_YEAR}年度資料完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()