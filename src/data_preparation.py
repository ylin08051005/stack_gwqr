"""
data_preparation.py
資料預處理與特徵工程 (支援跨期合併與階層式特徵實驗)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os
import pickle
import glob
import re

# 加入上層目錄以匯入 utils
sys.path.append('..')
from utils import load_config, create_directories, save_model

# =====================================================================
# [實驗控制中心] 階層式模型切換開關
# =====================================================================
EXPERIMENT_LEVEL = 1  
# 1 = 基礎變數 + [建物特性]
# 2 = 基礎變數 + [建物特性] + [社經地位]
# 3 = 基礎變數 + [建物特性] + [社經地位] + [環境變項]
# 4 = 基礎變數 + [建物特性] + [社經地位] + [環境變項] + [氣象變項] (全放)


def get_features_by_level(level):
    """根據實驗等級動態組合特徵"""
    
    # 0. 必帶基礎變數
    base_features = [
        '經度', '緯度',                  
        'time_index',                  
        '建物移轉總面積平方公尺',        
        'house_age',                   
        '移轉層次'                       
    ]
    
    f_building = ['建物現況格局-廳', '建物現況格局-衛', '電梯']
    f_socio = ['master_up_rate', 'Dependency_ratio', 'Sex_ratio', 'popdense']
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
    print("資料準備與跨期合併階段")
    print("=" * 60)
    
    features, exp_name = get_features_by_level(EXPERIMENT_LEVEL)
    target = config['data']['target'] 
    
    print(f"\n[目前執行實驗]: {exp_name}")
    print(f"[使用特徵數量]: {len(features)} 個")
    
    # =====================================================================
    # 1. 自動抓取並合併所有季度的 CSV (1122 ~ 1134)
    # =====================================================================
    raw_dir = os.path.join(config['paths']['data_dir'], 'raw')
    search_pattern = os.path.join(raw_dir, 'realprice_with_varcount_idw*.csv')
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"\n❌ 錯誤: 在 {raw_dir} 找不到符合 {search_pattern} 的檔案！")
        return
        
    # 時間轉換字典 (嚴格使用 1~7 以避免數值扭曲)
    time_mapping = {
        '1122': 1, '1123': 2, '1124': 3,
        '1131': 4, '1132': 5, '1133': 6, '1134': 7
    }
    
    df_list = []
    print("\n開始載入與合併各期資料...")
    for file in all_files:
        filename = os.path.basename(file)
        match = re.search(r'11[23][1-4]', filename)
        
        if match:
            quarter_key = match.group()
            tmp_df = pd.read_csv(file)
            tmp_df['time_index'] = time_mapping.get(quarter_key, 0)
            df_list.append(tmp_df)
            print(f"  - 成功載入 {filename} (資料筆數: {len(tmp_df)})")
            
    df = pd.concat(df_list, ignore_index=True)
    
    # ======== 【清除無效的幽靈欄位 (Unnamed)】 ========
    ghost_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if ghost_cols:
        df.drop(columns=ghost_cols, inplace=True)
        print(f"\n  🧹 已經自動清除 {len(ghost_cols)} 個因為地址逗號錯位產生的幽靈欄位！")
    # ==================================================
    
    print(f"✅ 初始合併完成！總資料形狀: {df.shape}")
    
    # =====================================================================
    # 2. 嚴格清理空白與排序欄位
    # =====================================================================
    print("\n執行資料清洗與欄位排序...")
    # 將所有隱形的空白字串轉為 NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    
    # 定義首要保護的欄位
    priority_cols = ['土地位置建物門牌', '緯度', '經度', target]
    priority_cols = [c for c in priority_cols if c in df.columns]
    
    # 刪除關鍵欄位為空的「幽靈房屋資料」
    initial_len = len(df)
    df = df.dropna(subset=priority_cols)
    dropped_len = initial_len - len(df)
    if dropped_len > 0:
        print(f"  ⚠️ 剔除了 {dropped_len} 筆缺少門牌、經緯度或總價的無效資料！")
    
    # ======== 【關鍵修復：重置索引】 ========
    # 確保刪除資料後，列編號依然是連續的，避免預測結果配對錯亂！
    df.reset_index(drop=True, inplace=True)
    # ========================================

    # 重新排序欄位
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]
    
    # 輸出整理好的乾淨資料
    combined_raw_path = os.path.join(raw_dir, 'realprice_combined_all.csv')
    df.to_csv(combined_raw_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 排序並清洗完成的原始資料已儲存至: {combined_raw_path}")
    
    # =====================================================================
    # 3. 機器學習特徵處理 (補值、對數轉換、編碼)
    # =====================================================================
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"\n❌ 錯誤: 以下特徵欄位在 CSV 中不存在:\n{missing_cols}")
        return
    
    # 針對「非經緯度」的其他特徵進行補值
    features_to_fill = [col for col in features if col not in ['經度', '緯度']]
    if df[features_to_fill].isnull().sum().sum() > 0:
        print("\n針對一般特徵補值 (數值補中位數 / 類別補眾數)...")
        for col in features_to_fill:
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
    
    test_size = config['data']['train_test_split']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config['project']['random_seed']
    )
    
    # =====================================================================
    # 4. 特徵標準化與儲存
    # =====================================================================
    print("\n執行特徵標準化 (Z-score Scaling)...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)
    X_full_scaled = pd.DataFrame(scaler.transform(X_full), columns=X_full.columns, index=X_full.index)
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    
    X_train_scaled.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    X_full_scaled.to_csv(os.path.join(processed_dir, 'X_full.csv'), index=False)
    
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(processed_dir, 'y_full.npy'), y_full)
    
    X_train.index.to_series().to_csv(os.path.join(processed_dir, 'train_index.csv'), index=False)
    X_test.index.to_series().to_csv(os.path.join(processed_dir, 'test_index.csv'), index=False)
    X_full.index.to_series().to_csv(os.path.join(processed_dir, 'full_index.csv'), index=False)
    
    save_model(scaler, os.path.join(config['paths']['model_dir'], 'scaler.pkl'))
    
    feature_info = {
        'experiment_name': exp_name,
        'feature_names': X_train_scaled.columns.tolist(),
        'n_features': X_train_scaled.shape[1]
    }
    with open(os.path.join(processed_dir, 'feature_info.pkl'), 'wb') as f:
        pickle.dump(feature_info, f)

    print("\n" + "=" * 60)
    print(f"🎉 資料準備完成！目前的乾淨資料量: {df.shape[0]} 筆")
    print("=" * 60)

if __name__ == "__main__":
    main()