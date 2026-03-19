"""
02_stage1_xgboost.py
XGBoost Quantile Regression 模型
學術三段式切分版 + 全樣本重訓 (供 GWQR 使用)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import sys
import os
import warnings
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import (load_config, save_model, evaluate_predictions)

warnings.filterwarnings('ignore')

class XGBoostQuantileRegressor:
    def __init__(self, quantile, params, random_seed=42):
        self.quantile = quantile
        self.params = params
        self.random_seed = random_seed
        self.model = None
        
    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        params = self.params.copy()
        params['seed'] = self.random_seed
        params['objective'] = 'reg:quantileerror'
        params['quantile_alpha'] = self.quantile
        params['tree_method'] = 'hist'
        params['device'] = 'cpu'
        
        n_estimators = params.pop('n_estimators', 100)
        self.model = xgb.train(
            params, dtrain, num_boost_round=n_estimators, verbose_eval=False
        )
        return self
    
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

def get_optuna_params(trial, config):
    xgb_config = config['models']['xgboost']
    return {
        'n_estimators': trial.suggest_int('n_estimators', xgb_config['n_estimators_range'][0], xgb_config['n_estimators_range'][1]),
        'max_depth': trial.suggest_int('max_depth', xgb_config['max_depth_range'][0], xgb_config['max_depth_range'][1]),
        'learning_rate': trial.suggest_float('learning_rate', xgb_config['learning_rate_range'][0], xgb_config['learning_rate_range'][1], log=True),
        'subsample': trial.suggest_float('subsample', xgb_config['subsample_range'][0], xgb_config['subsample_range'][1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', xgb_config['colsample_bytree_range'][0], xgb_config['colsample_bytree_range'][1]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

def optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val, quantile, config):
    def objective(trial):
        params = get_optuna_params(trial, config)
        model = XGBoostQuantileRegressor(quantile, params, config['project']['random_seed'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        error = y_val - y_pred
        check_loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)
        return np.mean(check_loss)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=config['project']['random_seed']))
    study.optimize(objective, n_trials=config['hyperparameter_tuning']['n_trials'], timeout=config['hyperparameter_tuning']['timeout'])
    return study.best_params

def save_detailed_predictions(predictions_dict, y_true_real, indices, raw_data_path, output_path):
    """
    將預測結果與原始經緯度、門牌資訊整合儲存，並確保指定的欄位順序。
    注意：這裡改為直接傳入 indices 陣列，方便處理 Test 集與 Full 集。
    """
    print(f"正在整合詳細資料至: {output_path}")
    try:
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

        # 5. 執行分位數交叉修正 (確保 0.1 <= 0.25 <= 0.5 ...)
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
        print(f"✅ 儲存完成。")
        
    except Exception as e:
        print(f"❌ 儲存預測報表失敗: {e}")

def main():
    start_time = time.time()
    config = load_config(os.path.join(parent_dir, 'config.yaml'))
    
    print("=" * 60)
    print("XGBoost Quantile Regression — 效能驗證與全樣本推論版")
    print("=" * 60)
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    # ★ 請確認這支檔案的年份與 data_preparation.py 產生的一致 (例如 113)
    # 若您在 data_preparation 設定 TARGET_YEAR = '112'，這裡記得改為 112
    target_year = '112' # 可修改
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', f'realprice_combined_{target_year}_all.csv')

    # 讀取特徵與標籤
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

    quantiles = config['data']['quantiles']
    all_quantiles = ['mean'] + quantiles
    
    predictions_test = {}
    predictions_full = {} # ★ 新增儲存全樣本預測結果

    for q in all_quantiles:
        print(f"\n>>> [Quantile = {q}]")
        
        # ---------------- 階段 A：尋找最佳參數並產出 Test 盲測結果 ----------------
        print("  - 尋找最佳參數與進行盲測評估...")
        if q == 'mean':
            def objective_mean(trial):
                params = get_optuna_params(trial, config)
                model = xgb.train({'objective': 'reg:squarederror', **params}, xgb.DMatrix(X_train, label=y_train), num_boost_round=params['n_estimators'])
                return np.sqrt(np.mean((y_val - model.predict(xgb.DMatrix(X_val))) ** 2))
            study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
            study.optimize(objective_mean, n_trials=config['hyperparameter_tuning']['n_trials'])
            best_params = study.best_params
            
            # 使用 best_params 訓練供評估用的模型
            eval_model = xgb.train({'objective': 'reg:squarederror', **best_params}, xgb.DMatrix(X_train, label=y_train), num_boost_round=best_params['n_estimators'])
            predictions_test[q] = eval_model.predict(xgb.DMatrix(X_test))
            
            # ★ 階段 B：使用 best_params 進行全樣本重訓
            print("  - [GWQR 用] 進行全樣本重新配適...")
            full_model = xgb.train({'objective': 'reg:squarederror', **best_params}, xgb.DMatrix(X_all, label=y_all), num_boost_round=best_params['n_estimators'])
            predictions_full[q] = full_model.predict(xgb.DMatrix(X_all))
            
        else:
            best_params = optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val, q, config)
            
            # 供評估用的模型
            eval_model = XGBoostQuantileRegressor(q, best_params).fit(X_train, y_train).model
            predictions_test[q] = eval_model.predict(xgb.DMatrix(X_test))
            
            # ★ 階段 B：使用 best_params 進行全樣本重訓
            print("  - [GWQR 用] 進行全樣本重新配適...")
            full_model = XGBoostQuantileRegressor(q, best_params).fit(X_all, y_all).model
            predictions_full[q] = full_model.predict(xgb.DMatrix(X_all))

    # 1. 輸出評估報表 (基於 Test 集)
    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(v) for k, v in predictions_test.items()}
    results_df = evaluate_predictions(y_test_real, predictions_test_real, quantiles, "XGBoost")
    
    # 2. 儲存預測結果
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    
    # ★ 輸出 Test 盲測集 CSV (供論文效能展示用)
    save_detailed_predictions(
        predictions_test, y_test_real, idx_test, raw_data_path, 
        os.path.join(result_dir, 'xgboost_test_predictions.csv')
    )
    
    # ★ 輸出 Full 全樣本 CSV (供論文第二階段 GWQR 使用)
    y_all_real = np.expm1(y_all)
    save_detailed_predictions(
        predictions_full, y_all_real, idx_all, raw_data_path, 
        os.path.join(result_dir, 'xgboost_full_predictions.csv')
    )

    # 3. 紀錄執行時間
    elapsed_time = time.time() - start_time
    results_df['Execution_Time'] = f"{elapsed_time/60:.2f} 分鐘"
    results_df.to_csv(os.path.join(config['paths']['result_dir'], 'evaluation', 'xgboost_evaluation.csv'), index=False)
    print("\n" + "=" * 60)
    print(" XGBoost 測試集評估報表 (模型效能驗證用)")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()