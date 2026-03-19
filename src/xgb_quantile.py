"""
02_stage1_xgboost.py
XGBoost Quantile Regression 模型
學術三段式切分版 (Train/Val/Test) - 包含預測結果儲存
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

# 修正：使用絕對路徑以確保能正確匯入 utils
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

def save_detailed_predictions(predictions_dict, index_file, raw_data_path, output_path):
    """
    將預測結果與原始經緯度、門牌資訊整合儲存
    """
    print(f"正在整合詳細資料至: {output_path}")
    try:
        indices = pd.read_csv(index_file).iloc[:, 0].astype(int).values
        df_output = pd.DataFrame(index=indices)
        df_raw = pd.read_csv(raw_data_path)
        raw_cols = ['土地位置建物門牌', '緯度', '經度']
        df_output = df_output.join(df_raw[raw_cols], how='left')

        # 排序分位數並還原 Log 轉換
        quantiles = sorted([q for q in predictions_dict.keys() if q != 'mean'])
        if 'mean' in predictions_dict:
            df_output['pred_mean'] = np.expm1(predictions_dict['mean'])
        
        for q in quantiles:
            df_output[f'pred_{q}'] = np.expm1(predictions_dict[q])

        # 執行分位數交叉修正
        pred_cols = [f'pred_{q}' for q in quantiles]
        q_values = df_output[pred_cols].values
        q_values.sort(axis=1)
        df_output[pred_cols] = q_values

        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 儲存完成。")
    except Exception as e:
        print(f"❌ 儲存預測報表失敗: {e}")

def main():
    start_time = time.time()
    config = load_config(os.path.join(parent_dir, 'config.yaml'))
    
    print("=" * 60)
    print("XGBoost Quantile Regression — 三段式預測版")
    print("=" * 60)
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', 'realprice_combined_all.csv')

    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_val   = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_val   = np.load(os.path.join(processed_dir, 'y_val.npy'))
    y_test  = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    quantiles = config['data']['quantiles']
    all_quantiles = ['mean'] + quantiles
    predictions_test = {}

    for q in all_quantiles:
        print(f"\n>>> 訓練 Quantile = {q}")
        if q == 'mean':
            def objective_mean(trial):
                params = get_optuna_params(trial, config)
                model = xgb.train({'objective': 'reg:squarederror', **params}, xgb.DMatrix(X_train, label=y_train), num_boost_round=params['n_estimators'])
                return np.sqrt(np.mean((y_val - model.predict(xgb.DMatrix(X_val))) ** 2))
            study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
            study.optimize(objective_mean, n_trials=config['hyperparameter_tuning']['n_trials'])
            best_params = study.best_params
            final_model = xgb.train({'objective': 'reg:squarederror', **best_params}, xgb.DMatrix(X_train, label=y_train), num_boost_round=best_params['n_estimators'])
        else:
            best_params = optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val, q, config)
            final_model = XGBoostQuantileRegressor(q, best_params).fit(X_train, y_train).model

        # 生成預測 (對象為第 6~7 期的 X_test)
        predictions_test[q] = final_model.predict(xgb.DMatrix(X_test))

    # 1. 輸出評估報表
    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(v) for k, v in predictions_test.items()}
    results_df = evaluate_predictions(y_test_real, predictions_test_real, quantiles, "XGBoost")
    
    # 2. 儲存預測 CSV 檔案 (關鍵新增)
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    save_detailed_predictions(predictions_test, os.path.join(processed_dir, 'test_index.csv'), raw_data_path, os.path.join(result_dir, 'xgboost_test_predictions.csv'))

    # 3. 紀錄執行時間
    elapsed_time = time.time() - start_time
    results_df['Execution_Time'] = f"{elapsed_time/60:.2f} 分鐘"
    results_df.to_csv(os.path.join(config['paths']['result_dir'], 'evaluation', 'xgboost_evaluation.csv'), index=False)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()