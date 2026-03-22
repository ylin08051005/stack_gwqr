"""
06_stage1_ensemble.py
整合第一階段所有模型的預測結果，準備給第二階段 GWQR 使用
支援自動掃描多個年份與特徵層級 (如 112_L1_pred, 113_L2_pred)
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('..')
from utils import load_config, evaluate_predictions

def main():
    config = load_config('./config.yaml')
    
    print("=" * 60)
    print("第一階段 Ensemble - 自動批量整合所有模型預測")
    print("=" * 60)
    
    predictions_base_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    ensemble_base_dir = os.path.join(config['paths']['result_dir'], 'stage1_ensemble')
    eval_base_dir = os.path.join(config['paths']['result_dir'], 'evaluation')
    
    os.makedirs(ensemble_base_dir, exist_ok=True)
    os.makedirs(eval_base_dir, exist_ok=True)
    
    # 模型與分位數列表
    models = ['xgboost', 'rf', 'lgb', 'nn'] 
    quantiles = config['data']['quantiles']
    all_quantiles = ['mean'] + quantiles
    
    # 1. 自動偵測 predictions 目錄下所有的 _pred 資料夾
    if not os.path.exists(predictions_base_dir):
        print(f"找不到預測資料夾: {predictions_base_dir}")
        return
        
    pred_folders = sorted([f for f in os.listdir(predictions_base_dir) 
                           if os.path.isdir(os.path.join(predictions_base_dir, f)) and '_pred' in f])
    
    if not pred_folders:
        print("沒有找到任何類似 '112_L1_pred' 的預測資料夾。請先執行模型預測腳本。")
        return

    print(f"偵測到以下預測資料夾將進行整合: {pred_folders}")

    # =========================================================
    # 核心函數：針對特定資料夾讀取並整合
    # =========================================================
    def combine_predictions(folder_path, mode='test'):
        """ mode: 'test' or 'full' """
        base_df = None
        
        # 先找一個有產出的模型來當作基底 (抓取地址、座標、真實房價)
        for m in models:
            path = os.path.join(folder_path, f'{m}_{mode}_predictions.csv')
            if os.path.exists(path):
                base_df = pd.read_csv(path)
                # ★ 改良：直接從 CSV 抓 actual_price，不再依賴 npy，避免行數錯亂
                cols_to_keep = ['土地位置建物門牌', '緯度', '經度', 'actual_price']
                base_df = base_df[[c for c in cols_to_keep if c in base_df.columns]]
                break
        
        if base_df is None:
            return None

        # 迴圈讀取每個模型的預測值並合併
        for model_name in models:
            filepath = os.path.join(folder_path, f'{model_name}_{mode}_predictions.csv')
            if not os.path.exists(filepath):
                continue
            
            df = pd.read_csv(filepath)
            for q in all_quantiles:
                original_col = f"pred_{q}" 
                new_col = f"{model_name}_q{q}" 
                
                if original_col in df.columns:
                    base_df[new_col] = df[original_col]

        # ★ 產生 Ensemble 平均值 (直接給 GWQR 用的最終強特徵)
        for q in all_quantiles:
            model_cols = [f"{m}_q{q}" for m in models if f"{m}_q{q}" in base_df.columns]
            if model_cols:
                base_df[f"Ensemble_q{q}"] = base_df[model_cols].mean(axis=1)

        return base_df

    # =========================================================
    # 批量處理每個資料夾
    # =========================================================
    for folder in pred_folders:
        prefix = folder.replace('_pred', '') # 提取 '112_L1'
        folder_path = os.path.join(predictions_base_dir, folder)
        
        print(f"\n" + "-" * 50)
        print(f"正在處理: {prefix}")
        print("-" * 50)
        
        test_df = combine_predictions(folder_path, 'test')
        full_df = combine_predictions(folder_path, 'full')
        
        # 建立該組態的專屬輸出目錄
        out_dir = os.path.join(ensemble_base_dir, prefix)
        os.makedirs(out_dir, exist_ok=True)
        
        # 儲存 CSV
        if test_df is not None:
            test_path = os.path.join(out_dir, f'{prefix}_stage1_test_ensemble.csv')
            test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
            print(f"✅ 測試集整合完畢: {test_path} (Shape: {test_df.shape})")
            
        if full_df is not None:
            full_path = os.path.join(out_dir, f'{prefix}_stage1_full_ensemble.csv')
            full_df.to_csv(full_path, index=False, encoding='utf-8-sig')
            print(f"✅ 全樣本整合完畢: {full_path} (Shape: {full_df.shape})")

        # ---------------------------------------------------------
        # 評估該層級所有模型表現
        # ---------------------------------------------------------
        if test_df is not None:
            y_true = test_df['actual_price'].values
            all_results = []
            
            # 評估各別單一模型
            for model_name in models:
                preds_dict = {}
                valid_model = True
                for q in all_quantiles:
                    col_name = f"{model_name}_q{q}"
                    if col_name in test_df.columns:
                        preds_dict[q] = test_df[col_name].values
                    else:
                        valid_model = False
                
                if valid_model:
                    res = evaluate_predictions(y_true, preds_dict, quantiles, model_name)
                    all_results.append(res)
            
            # 評估 Ensemble 平均表現
            avg_preds_dict = {}
            for q in all_quantiles:
                if f"Ensemble_q{q}" in test_df.columns:
                    avg_preds_dict[q] = test_df[f"Ensemble_q{q}"].values
            
            if avg_preds_dict:
                res_avg = evaluate_predictions(y_true, avg_preds_dict, quantiles, "Average_Ensemble")
                all_results.append(res_avg)

            if all_results:
                final_eval_df = pd.concat(all_results, ignore_index=True)
                eval_path = os.path.join(eval_base_dir, f'{prefix}_stage1_summary.csv')
                final_eval_df.to_csv(eval_path, index=False)
                print(f"📊 評估報告已儲存: {eval_path}")

    print("\n" + "=" * 60)
    print(" 所有年份與層級整合完成，請前往 R 語言執行 Stage 2 GWQR。")
    print("=" * 60)

if __name__ == "__main__":
    main()