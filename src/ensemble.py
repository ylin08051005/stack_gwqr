"""
06_stage1_ensemble.py
整合第一階段所有模型的預測結果，準備給第二階段 GWQR 使用
"""
import pandas as pd
import numpy as np
import sys
import os

# 加入上層目錄以匯入 utils
sys.path.append('..')
from utils import load_config, evaluate_predictions

def main():
    # 載入配置
    config = load_config('../config.yaml')
    
    print("=" * 50)
    print("第一階段 Ensemble - 整合所有模型預測")
    print("=" * 50)
    
    # 路徑設定
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    
    # 載入真實值 (Log Scale，需要轉回 Real Scale 才能跟預測值比較)
    # 注意：前面的腳本輸出的 predictions 已經是 Real Scale 了
    y_test_log = np.load(os.path.join(processed_dir, 'y_test.npy'))
    y_full_log = np.load(os.path.join(processed_dir, 'y_full.npy'))
    
    y_test_real = np.expm1(y_test_log)
    y_full_real = np.expm1(y_full_log)
    
    # 模型列表 (請確保這些模型的 .csv 都在 results/predictions 下)
    models = ['xgboost', 'rf', 'lgb', 'nn', 'dnn', 'cnn'] 
    # 如果有些模型沒跑，可以註解掉，例如:
    # models = ['xgboost', 'lgb', 'rf']
    
    # 分位數列表
    quantiles = config['data']['quantiles']
    all_quantiles = ['mean'] + quantiles
    
    print(f"\n整合模型: {models}")
    print(f"分位數: {all_quantiles}")

    # =========================================================
    # 核心函數：讀取並整合
    # =========================================================
    def combine_predictions(mode='test'):
        """
        mode: 'test' or 'full'
        """
        print(f"\n正在處理 {mode} 資料集...")
        
        # 1. 先讀取基礎資訊 (地址、經緯度)
        # 我們隨便讀一個已存在的模型輸出來當作基底 (因為大家的順序與地址都是一樣的)
        base_df = None
        for m in models:
            path = os.path.join(result_dir, f'{m}_{mode}_predictions.csv')
            if os.path.exists(path):
                base_df = pd.read_csv(path)
                # 只保留基本資訊欄位
                cols_to_keep = ['土地位置建物門牌', '緯度', '經度']
                # 確保這些欄位存在
                base_df = base_df[[c for c in cols_to_keep if c in base_df.columns]]
                break
        
        if base_df is None:
            print(f"錯誤：找不到任何 {mode} 的預測檔案，無法整合。")
            return None

        # 2. 加入真實值 (y_true)
        if mode == 'test':
            base_df['y_true'] = y_test_real
        else:
            base_df['y_true'] = y_full_real

        # 3. 迴圈讀取每個模型的預測值並合併
        for model_name in models:
            filepath = os.path.join(result_dir, f'{model_name}_{mode}_predictions.csv')
            
            if not os.path.exists(filepath):
                print(f"  警告: 找不到 {model_name} 的預測檔，跳過。")
                continue
            
            print(f"  - 載入 {model_name} ...")
            df = pd.read_csv(filepath)
            
            # 提取預測欄位 (pred_mean, pred_0.1 ...)
            for q in all_quantiles:
                original_col = f"pred_{q}" # 這是前面腳本輸出的欄位名
                new_col = f"{model_name}_q{q}" # 這是我们要改成的欄位名 (給GWQR用)
                
                if original_col in df.columns:
                    base_df[new_col] = df[original_col]
                else:
                    print(f"    缺失欄位: {original_col} in {model_name}")

        return base_df

    # =========================================================
    # 執行整合
    # =========================================================
    test_ensemble_df = combine_predictions('test')
    full_ensemble_df = combine_predictions('full')
    
    # =========================================================
    # 儲存整合結果 (供第二階段R使用)
    # =========================================================
    stage1_output_dir = os.path.join(config['paths']['result_dir'], 'stage1_ensemble')
    os.makedirs(stage1_output_dir, exist_ok=True)
    
    if test_ensemble_df is not None:
        test_path = os.path.join(stage1_output_dir, 'stage1_test_ensemble.csv')
        test_ensemble_df.to_csv(test_path, index=False, encoding='utf-8-sig')
        print(f"\n測試集整合完畢: {test_path} (Shape: {test_ensemble_df.shape})")

    if full_ensemble_df is not None:
        full_path = os.path.join(stage1_output_dir, 'stage1_full_ensemble.csv')
        full_ensemble_df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"完整集整合完畢: {full_path} (Shape: {full_ensemble_df.shape})")

    # =========================================================
    # 評估各模型表現 (Baseline Comparison)
    # =========================================================
    print("\n" + "=" * 50)
    print("第一階段各模型表現總評 (RMSE & Check Loss)")
    print("=" * 50)
    
    if test_ensemble_df is not None:
        all_results = []
        
        # 1. 評估單一模型
        for model_name in models:
            # 建構符合 evaluate_predictions 格式的字典
            preds_dict = {}
            valid_model = True
            for q in all_quantiles:
                col_name = f"{model_name}_q{q}"
                if col_name in test_ensemble_df.columns:
                    preds_dict[q] = test_ensemble_df[col_name].values
                else:
                    valid_model = False
            
            if valid_model:
                res = evaluate_predictions(y_test_real, preds_dict, quantiles, model_name)
                all_results.append(res)
        
        # 2. 評估簡單平均 (Average Ensemble)
        avg_preds_dict = {}
        for q in all_quantiles:
            # 找出所有模型的該分位數欄位
            cols = [f"{m}_q{q}" for m in models if f"{m}_q{q}" in test_ensemble_df.columns]
            if cols:
                avg_preds_dict[q] = test_ensemble_df[cols].mean(axis=1).values
        
        if avg_preds_dict:
            res_avg = evaluate_predictions(y_test_real, avg_preds_dict, quantiles, "Average_Ensemble")
            all_results.append(res_avg)

        # 3. 輸出總表
        if all_results:
            final_eval_df = pd.concat(all_results, ignore_index=True)
            print(final_eval_df.to_string(index=False))
            
            # 儲存
            eval_path = os.path.join(config['paths']['result_dir'], 'evaluation', 'stage1_summary.csv')
            final_eval_df.to_csv(eval_path, index=False)
            print(f"\n評估報告已儲存: {eval_path}")

    print("\n" + "=" * 50)
    print("Stage 1 整合完成，請前往 R 語言環境執行 Stage 2 GWQR。")
    print("=" * 50)

if __name__ == "__main__":
    main()