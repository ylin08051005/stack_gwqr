import pandas as pd
import numpy as np
import os
import sys

# 確保能讀取到 utils.py
sys.path.append('.') 
from utils import evaluate_predictions

def main():
    # 1. 定義你要評估的模型與對應的檔案前綴
    models_config = {
        'XGBoost': 'xgboost',
        'RandomForest': 'rf',
        'LightGBM': 'lgb',
        'NN_MLP': 'nn'
    }
    
    # ★ 請根據你當前要評估的資料夾名稱做修改 (例如 112_L3 或 112_L4)
    target_folder = '112_L4_pred' 
    pred_dir = f"results/predictions/{target_folder}"
    
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    all_evaluations = []
    
    print("\n" + "=" * 70)
    print(f" 開始進行全樣本 (Full-Sample) 評估作業 - 目標資料夾: {target_folder}")
    print("=" * 70)

    # 2. 迴圈讀取四個模型的預測檔案
    for model_name, prefix in models_config.items():
        file_path = os.path.join(pred_dir, f"{prefix}_full_predictions.csv")
        
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到 {model_name} 的預測檔案 ({file_path})，跳過此模型。")
            continue
            
        print(f"處理中: {model_name}...")
        df_full = pd.read_csv(file_path)

        # 準備 y_true 與預測值字典
        y_true = df_full['actual_price'].values
        
        preds_dict = {}
        if 'pred_mean' in df_full.columns:
            preds_dict['mean'] = df_full['pred_mean'].values
            
        for q in quantiles:
            if f'pred_{q}' in df_full.columns:
                preds_dict[q] = df_full[f'pred_{q}'].values
            else:
                print(f"  [警告] {model_name} 缺少 {q} 分位數的預測欄位！")

        # 3. 呼叫你的 utils 進行評估，並在模型名稱後加上 _Full 標籤以便識別
        eval_name_label = f"{model_name}_Full"
        results_df = evaluate_predictions(y_true, preds_dict, quantiles, eval_name_label)
        all_evaluations.append(results_df)

    # 如果沒有成功讀取到任何檔案，則終止程式
    if not all_evaluations:
        print("\n 沒有找到任何預測檔案，請檢查 pred_dir 路徑設定是否正確。")
        return

    # 4. 將四個模型的評估結果上下合併成一張大表
    final_full_eval_df = pd.concat(all_evaluations, ignore_index=True)
    
    print("\n" + "=" * 70)
    print(" 四大模型全樣本 (Full-Sample) 評估總表")
    print("=" * 70)
    print(final_full_eval_df.to_string(index=False))

    # 5. 存檔
    output_path = f"results/evaluation/{target_folder}_all_models_full_evaluation.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_full_eval_df.to_csv(output_path, index=False)
    print(f"\n 全樣本評估大表已成功儲存至: {output_path}")

if __name__ == "__main__":
    main()