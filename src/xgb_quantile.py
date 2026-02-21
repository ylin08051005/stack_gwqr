"""
02_stage1_xgboost.py
XGBoost Quantile Regression 模型
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import TPESampler
import sys
import os
import warnings
import time  # 新增：用於計算執行時間

# 修正：使用絕對路徑以確保能正確匯入 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import (load_config, save_model, save_predictions, 
                   xgb_quantile_obj, evaluate_predictions)

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


def optimize_xgboost_hyperparameters(X_train, y_train, quantile, config):
    def objective(trial):
        params = get_optuna_params(trial, config)
        kf = KFold(n_splits=config['hyperparameter_tuning']['cv_folds'], shuffle=True, random_state=config['project']['random_seed'])
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = XGBoostQuantileRegressor(quantile, params, config['project']['random_seed'])
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            error = y_val - y_pred
            check_loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)
            cv_scores.append(np.mean(check_loss))
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=config['project']['random_seed']))
    study.optimize(
        objective, 
        n_trials=config['hyperparameter_tuning']['n_trials'],
        timeout=config['hyperparameter_tuning']['timeout'],
        n_jobs=config['project']['n_jobs'],
        show_progress_bar=True
    )
    return study.best_params


def save_detailed_predictions(predictions_dict, index_file, raw_data_path, output_path):

    try:
        indices_df = pd.read_csv(index_file)
        indices = indices_df.iloc[:, 0].astype(int).values
    except Exception as e:
        print(f"讀取索引檔失敗 {index_file}: {e}")
        return

    df_output = pd.DataFrame(index=indices)
    raw_cols = ['土地位置建物門牌', '緯度', '經度']

    try:
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"找不到資料 {raw_data_path}")
            
        df_raw = pd.read_csv(raw_data_path)
        df_merged = df_output.join(df_raw[raw_cols], how='left')
        df_output = df_merged
        
    except Exception as e:
        print(f"合併資料錯誤 ({e})")

    quantile_cols = []

    target_quantiles = sorted([q for q in predictions_dict.keys() if q != 'mean'])

    if 'mean' in predictions_dict:
        if len(predictions_dict['mean']) == len(df_output):
            df_output['pred_mean'] = np.expm1(predictions_dict['mean'])

    for q in target_quantiles:
        preds = predictions_dict[q]
        if len(preds) != len(df_output):
            continue
        col_name = f'pred_{q}'
        df_output[col_name] = np.expm1(preds)
        quantile_cols.append(col_name)

    if len(quantile_cols) > 1:
        print("分位數交叉修正")
        q_values = df_output[quantile_cols].values
        q_values.sort(axis=1)
        df_output[quantile_cols] = q_values

    cols = list(df_output.columns)
    priority_cols = ['土地位置建物門牌', '緯度', '經度']
    new_order = [c for c in priority_cols if c in cols] + [c for c in cols if c not in priority_cols]
    df_output = df_output[new_order]

    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')


def main():
    # 記錄程式開始時間
    start_time = time.time()
    
    # 動態取得 config 路徑
    config_path = os.path.join(parent_dir, 'config.yaml')
    config = load_config(config_path)
    
    print("=" * 50)
    print("XGBoost Quantile Regression")
    print("=" * 50)
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')

    # 修正：改為讀取 01_preprocess.py 合併後的總表
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', 'realprice_combined_all.csv')
    if not os.path.exists(raw_data_path):
        print(f"警告: 找不到合併後的原始資料 {raw_data_path}，請確認是否已執行 01_preprocess.py")

    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    X_full = pd.read_csv(os.path.join(processed_dir, 'X_full.csv'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    y_full = np.load(os.path.join(processed_dir, 'y_full.npy'))
    
    print(f"\n訓練集: {X_train.shape}")
    print(f"測試集: {X_test.shape}")
    print(f"完整資料: {X_full.shape}")
    
    quantiles = config['data']['quantiles']
    all_quantiles = ['mean'] + quantiles
    
    predictions_test = {}
    predictions_full = {}
    
    for q in all_quantiles:
        print(f"\n{'='*50}")
        print(f"訓練 Quantile = {q}")
        print(f"{'='*50}")
        
        if q == 'mean':
            print("超參數優化 (Mean)")
            def objective_mean(trial):
                params = get_optuna_params(trial, config)
                kf = KFold(n_splits=config['hyperparameter_tuning']['cv_folds'], shuffle=True, random_state=config['project']['random_seed'])
                cv_scores = []
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval = xgb.DMatrix(X_val, label=y_val)
                    train_params = params.copy()
                    n_estimators = train_params.pop('n_estimators')
                    train_params.update({'objective': 'reg:squarederror', 'tree_method': 'hist'})
                    model = xgb.train(train_params, dtrain, num_boost_round=n_estimators, verbose_eval=False)
                    y_pred = model.predict(dval)
                    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                    cv_scores.append(rmse)
                return np.mean(cv_scores)
            
            study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=config['project']['random_seed']))
            study.optimize(objective_mean, n_trials=config['hyperparameter_tuning']['n_trials'], timeout=config['hyperparameter_tuning']['timeout'], n_jobs=config['project']['n_jobs'], show_progress_bar=True)
            best_params = study.best_params
            print(f"最佳參數: {best_params}")
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            final_params = best_params.copy()
            final_n_estimators = final_params.pop('n_estimators')
            final_params.update({'objective': 'reg:squarederror', 'tree_method': 'hist', 'seed': config['project']['random_seed']})
            final_model = xgb.train(final_params, dtrain, num_boost_round=final_n_estimators, verbose_eval=False)
            
        else:
            print("超參數優化 (Quantile)")
            best_params = optimize_xgboost_hyperparameters(X_train, y_train, q, config)
            print(f"最佳參數: {best_params}")
            
            final_model_wrapper = XGBoostQuantileRegressor(q, best_params, config['project']['random_seed'])
            final_model_wrapper.fit(X_train, y_train)
            final_model = final_model_wrapper.model

        dtest = xgb.DMatrix(X_test)
        predictions_test[q] = final_model.predict(dtest)
        dfull = xgb.DMatrix(X_full)
        predictions_full[q] = final_model.predict(dfull)
        
        model_path = os.path.join(config['paths']['model_dir'], 'stage1', f'xgboost_q{q}.json')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        final_model.save_model(model_path)
        print(f"模型已儲存: {model_path}")

    print(f"\n{'='*50}")
    print("正在生成詳細預測報表...")
    
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    
    save_detailed_predictions(
        predictions_test,
        os.path.join(processed_dir, 'test_index.csv'),
        raw_data_path,
        os.path.join(result_dir, 'xgboost_test_predictions.csv')
    )
    
    save_detailed_predictions(
        predictions_full,
        os.path.join(processed_dir, 'full_index.csv'),
        raw_data_path,
        os.path.join(result_dir, 'xgboost_full_predictions.csv')
    )

    print(f"\n{'='*50}")
    print("測試集評估結果 (已還原至原始房價單位)")
    print(f"{'='*50}")
    
    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(v) for k, v in predictions_test.items()}
    
    results_df = evaluate_predictions(y_test_real, predictions_test_real, quantiles, "XGBoost")
    print(results_df.to_string(index=False))
    
    eval_dir = os.path.join(config['paths']['result_dir'], 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    results_df.to_csv(os.path.join(eval_dir, 'xgboost_evaluation.csv'), index=False)
    
    # 記錄結束時間並計算總耗時
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "=" * 50)
    print(f" XGBoost 訓練完成!")
    print(f" 總執行時間: {int(hours)} 小時 {int(minutes)} 分鐘 {seconds:.2f} 秒")
    print("=" * 50)


if __name__ == "__main__":
    main()