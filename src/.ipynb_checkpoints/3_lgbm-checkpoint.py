"""
04_stage1_lightgbm.py
LightGBM Quantile Regression 模型
學術三段式切分版 + 全樣本重訓 (供 GWQR 使用)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import sys
import os
import warnings
import time

# 修正：使用絕對路徑以確保能正確匯入 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import (load_config, save_model, evaluate_predictions)

warnings.filterwarnings('ignore')

class LightGBMQuantileRegressor:
    def __init__(self, quantile, params, random_seed=42, n_jobs=-1):
        self.quantile = quantile
        self.params = params
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.model = None
        
    def fit(self, X, y):
        params = self.params.copy()
        params['random_state'] = self.random_seed
        params['n_jobs'] = self.n_jobs
        params['verbose'] = -1
        
        n_estimators = params.pop('n_estimators', 100)
        
        if self.quantile == 'mean':
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        else:
            params['objective'] = 'quantile'
            params['alpha'] = self.quantile
            params['metric'] = 'quantile'
        
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators
        )
        return self
    
    def predict(self, X):
        return self.model.predict(X)

def get_optuna_params(trial, config):
    lgb_config = config['models']['lightgbm']
    return {
        'n_estimators': trial.suggest_int('n_estimators', lgb_config['n_estimators_range'][0], lgb_config['n_estimators_range'][1]),
        'max_depth': trial.suggest_int('max_depth', lgb_config['max_depth_range'][0], lgb_config['max_depth_range'][1]),
        'learning_rate': trial.suggest_float('learning_rate', lgb_config['learning_rate_range'][0], lgb_config['learning_rate_range'][1], log=True),
        'num_leaves': trial.suggest_int('num_leaves', lgb_config['num_leaves_range'][0], lgb_config['num_leaves_range'][1]),
        'subsample': trial.suggest_float('subsample', lgb_config['subsample_range'][0], lgb_config['subsample_range'][1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

def optimize_lgb_hyperparameters(X_train, y_train, X_val, y_val, quantile, config):
    """學術修正：使用單一驗證集 (X_val) 進行調參"""
    def objective(trial):
        params = get_optuna_params(trial, config)
        model = LightGBMQuantileRegressor(quantile, params, config['project']['random_seed'], n_jobs=config['project']['n_jobs'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        if quantile == 'mean':
            return np.sqrt(np.mean((y_val - y_pred) ** 2))
        else:
            error = y_val - y_pred
            return np.mean(np.where(error >= 0, quantile * error, (quantile - 1) * error))
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=config['project']['random_seed']))
    study.optimize(objective, n_trials=config['hyperparameter_tuning']['n_trials'], timeout=config['hyperparameter_tuning']['timeout'], show_progress_bar=True)
    return study.best_params

def save_detailed_predictions(predictions_dict, y_true_real, indices, raw_data_path, output_path):
    """
    將預測結果與原始經緯度、門牌資訊整合儲存，並確保指定的欄位順序。
    (改為傳入 indices 陣列以支援 Test / Full 雙重輸出)
    """
    print(f"正在整合預測資料至: {output_path}")
    try:
        # 1. 讀取 Index 與原始對照資料
        df_output = pd.DataFrame(index=indices)
        df_raw = pd.read_csv(raw_data_path)
        
        # 2. 合併基礎門牌與座標
        raw_cols = ['土地位置建物門牌', '緯度', '經度']
        df_output = df_output.join(df_raw[raw_cols], how='left')

        # ★ 3. 寫入真實房價 (已還原為正常金額)
        df_output['actual_price'] = y_true_real

        # 4. 寫入各分位數預測值 (還原 Log 轉換)
        quantiles = sorted([q for q in predictions_dict.keys() if q != 'mean'])
        if 'mean' in predictions_dict:
            df_output['pred_mean'] = np.expm1(predictions_dict['mean'])
        for q in quantiles:
            df_output[f'pred_{q}'] = np.expm1(predictions_dict[q])

        # 5. 分位數交叉修正 (Rearrangement)
        pred_cols = [f'pred_{q}' for q in quantiles]
        if len(pred_cols) > 1:
            q_values = df_output[pred_cols].values
            q_values.sort(axis=1)
            df_output[pred_cols] = q_values

        # ★ 6. 強制指定欄位輸出順序
        final_cols = ['土地位置建物門牌', '緯度', '經度', 'actual_price']
        if 'mean' in predictions_dict:
            final_cols.append('pred_mean')
        final_cols.extend(pred_cols)
        
        df_output = df_output[final_cols]
        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
        print("✅ 儲存成功")
    except Exception as e:
        print(f"❌ 儲存失敗: {e}")

def main():
    start_time = time.time()
    config = load_config(os.path.join(parent_dir, 'config.yaml'))
    
    print("=" * 60)
    print("LightGBM Quantile Regression — 效能驗證與全樣本推論版")
    print("=" * 60)
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    # ★ 請確認年份與 data_preparation.py 產生的一致
    target_year = '112' # 可修改
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', f'realprice_combined_{target_year}_all.csv')

    # 讀取 Train, Val, Test
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_val   = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_val   = np.load(os.path.join(processed_dir, 'y_val.npy'))
    y_test  = np.load(os.path.join(processed_dir, 'y_test.npy'))

    # 讀取 Index
    idx_train = pd.read_csv(os.path.join(processed_dir, 'train_index.csv')).iloc[:, 0].astype(int).values
    idx_val   = pd.read_csv(os.path.join(processed_dir, 'val_index.csv')).iloc[:, 0].astype(int).values
    idx_test  = pd.read_csv(os.path.join(processed_dir, 'test_index.csv')).iloc[:, 0].astype(int).values

    # ★ 準備全樣本資料 (供階段 B 重訓使用)
    X_all = pd.concat([X_train, X_val, X_test], axis=0).reset_index(drop=True)
    y_all = np.concatenate([y_train, y_val, y_test])
    idx_all = np.concatenate([idx_train, idx_val, idx_test])
    
    all_quantiles = ['mean'] + config['data']['quantiles']
    predictions_test = {}
    predictions_full = {} # ★ 新增

    for q in all_quantiles:
        print(f"\n>>> 訓練 Quantile = {q}")
        best_params = optimize_lgb_hyperparameters(X_train, y_train, X_val, y_val, q, config)
        print(f"最佳參數: {best_params}")
        
        # ---------------- 階段 A：進行盲測評估訓練 ----------------
        print("  - 進行盲測評估訓練...")
        eval_model = LightGBMQuantileRegressor(q, best_params, config['project']['random_seed'], config['project']['n_jobs'])
        eval_model.fit(X_train, y_train)
        predictions_test[q] = eval_model.predict(X_test)
        
        # 儲存評估模型
        model_path = os.path.join(config['paths']['model_dir'], 'stage1', f'lgb_q{q}.txt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        eval_model.model.save_model(model_path)

        # ---------------- 階段 B：進行全樣本重訓 ----------------
        print("  - [GWQR 用] 進行全樣本重新配適...")
        full_model = LightGBMQuantileRegressor(q, best_params, config['project']['random_seed'], config['project']['n_jobs'])
        full_model.fit(X_all, y_all)
        predictions_full[q] = full_model.predict(X_all)

    # 評估與執行時間紀錄 (基於 Test 集)
    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(v) for k, v in predictions_test.items()}
    
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    
    # ★ 輸出 Test 盲測集 CSV
    save_detailed_predictions(
        predictions_test, 
        y_test_real, 
        idx_test, 
        raw_data_path, 
        os.path.join(result_dir, 'lgb_test_predictions.csv')
    )
    
    # ★ 輸出 Full 全樣本 CSV
    y_all_real = np.expm1(y_all)
    save_detailed_predictions(
        predictions_full, 
        y_all_real, 
        idx_all, 
        raw_data_path, 
        os.path.join(result_dir, 'lgb_full_predictions.csv')
    )

    results_df = evaluate_predictions(y_test_real, predictions_test_real, config['data']['quantiles'], "LightGBM")
    
    elapsed_time = time.time() - start_time
    results_df['Execution_Time'] = f"{elapsed_time/60:.2f} 分鐘"
    
    print("\n" + "=" * 60)
    print(" LightGBM 測試集評估報表 (模型效能驗證用)")
    print(results_df.to_string(index=False))
    print("=" * 60)
    
    results_df.to_csv(os.path.join(config['paths']['result_dir'], 'evaluation', 'lgb_evaluation.csv'), index=False)

if __name__ == "__main__":
    main()