"""
XGBoost Quantile Regression Model
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
import sys
import os
import warnings
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import (load_config, save_model, evaluate_predictions)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        check_loss = np.where(error >= 0, float(quantile) * error, (float(quantile) - 1.0) * error)
        return np.mean(check_loss)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=config['project']['random_seed']))
    study.optimize(
        objective, 
        n_trials=config['hyperparameter_tuning']['n_trials'], 
        timeout=config['hyperparameter_tuning']['timeout'],
        show_progress_bar=True
    )
    return study.best_params

def save_detailed_predictions(predictions_dict, y_true_real, indices, raw_data_path, output_path):
    print(f"儲存報表: {output_path}")
    try:
        df_output = pd.DataFrame(index=indices)
        df_raw = pd.read_csv(raw_data_path)
        
        raw_cols = ['土地位置建物門牌', '緯度', '經度']
        df_output = df_output.join(df_raw[raw_cols], how='left')
        
        df_output['actual_price_million'] = y_true_real

        quantiles = sorted([q for q in predictions_dict.keys() if q != 'mean'])
        if 'mean' in predictions_dict:
            df_output['pred_mean'] = np.exp(predictions_dict['mean'])
        
        for q in quantiles:
            df_output[f'pred_{q}'] = np.exp(predictions_dict[q])

        pred_cols = [f'pred_{q}' for q in quantiles]
        if len(pred_cols) > 1:
            q_values = df_output[pred_cols].values
            q_values.sort(axis=1)
            df_output[pred_cols] = q_values

        final_cols = ['土地位置建物門牌', '緯度', '經度', 'actual_price_million']
        if 'mean' in predictions_dict:
            final_cols.append('pred_mean')
        final_cols.extend(pred_cols) 
        
        df_output = df_output[final_cols]
        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"儲存完成")
    except Exception as e:
        print(f"儲存失敗: {e}")

def main():
    start_time = time.time()
    config = load_config(os.path.join(parent_dir, 'config.yaml'))
    EXP_LEVEL = "L3"
    target_year = '112' 
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', f'realprice_combined_{target_year}_all.csv')

    X_all_raw = pd.read_csv(os.path.join(processed_dir, 'X_all.csv'))
    y_all_old = np.load(os.path.join(processed_dir, 'y_all.npy')) 
    idx_all = pd.read_csv(os.path.join(processed_dir, 'index_all.csv')).iloc[:, 0].values

    y_raw_price = np.expm1(y_all_old) 
    y_price_million = y_raw_price / 1000000.0
    y_all = np.log(y_price_million)

    quantiles = config['data']['quantiles']
    all_quantiles = ['mean'] + quantiles

    is_test_set = (X_all_raw['fold'] == -1)
    
    X_train_70 = X_all_raw[~is_test_set].copy()
    y_train_70 = y_all[~is_test_set]
    idx_train_70 = idx_all[~is_test_set]
    
    X_test_30 = X_all_raw[is_test_set].drop(columns=['fold']).copy()
    y_test_30 = y_all[is_test_set]
    idx_test_30 = idx_all[is_test_set]

    predictions_oof = {q: np.zeros_like(y_train_70) for q in all_quantiles} 
    test_preds_sum = {q: np.zeros_like(y_test_30) for q in all_quantiles} 
    all_fold_evaluations = [] 

    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    eval_dir = os.path.join(config['paths']['result_dir'], 'evaluation')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    for fold_idx in range(5):
        print(f"\n" + "="*40)
        print(f"執行 Train Fold {fold_idx + 1} / 5")
        print("="*40)
        
        val_mask = (X_train_70['fold'] == fold_idx)
        tr_mask = ~val_mask
        
        X_tr_full = X_train_70[tr_mask].drop(columns=['fold'])
        y_tr_full = y_train_70[tr_mask]
        X_val = X_train_70[val_mask].drop(columns=['fold'])
        y_val = y_train_70[val_mask]
        idx_val = idx_train_70[val_mask]

        X_tr_opt, X_va_opt, y_tr_opt, y_va_opt = train_test_split(X_tr_full, y_tr_full, test_size=0.1, random_state=config['project']['random_seed'])
        
        fold_predictions = {}
        
        for q in all_quantiles:
            print(f"\n  > 訓練 Quantile = {q}")
            if q == 'mean':
                def objective_mean(trial):
                    params = get_optuna_params(trial, config)
                    model = xgb.train({'objective': 'reg:squarederror', **params}, xgb.DMatrix(X_tr_opt, label=y_tr_opt), num_boost_round=params['n_estimators'])
                    return np.sqrt(np.mean((y_va_opt - model.predict(xgb.DMatrix(X_va_opt))) ** 2))
                
                study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
                study.optimize(objective_mean, n_trials=config['hyperparameter_tuning']['n_trials'], show_progress_bar=True)
                best_params = study.best_params
                
                eval_model = xgb.train({'objective': 'reg:squarederror', **best_params}, xgb.DMatrix(X_tr_full, label=y_tr_full), num_boost_round=best_params['n_estimators'])
                fold_pred = eval_model.predict(xgb.DMatrix(X_val))
                test_pred = eval_model.predict(xgb.DMatrix(X_test_30))
                
            else:
                best_params = optimize_xgboost_hyperparameters(X_tr_opt, y_tr_opt, X_va_opt, y_va_opt, q, config)
                eval_model = XGBoostQuantileRegressor(q, best_params).fit(X_tr_full, y_tr_full).model
                fold_pred = eval_model.predict(xgb.DMatrix(X_val))
                test_pred = eval_model.predict(xgb.DMatrix(X_test_30))
                
            fold_predictions[q] = fold_pred
            predictions_oof[q][val_mask] = fold_pred
            test_preds_sum[q] += test_pred
            
            print(f" > Quantile {q} 訓練與預測完成")

        y_val_real = np.exp(y_val) 
        fold_preds_real = {k: np.exp(v) for k, v in fold_predictions.items()}
        
        fold_eval_df = evaluate_predictions(y_val_real, fold_preds_real, quantiles, f"XGB_Fold_{fold_idx+1}")
        all_fold_evaluations.append(fold_eval_df)
        
        save_detailed_predictions(
            fold_predictions, y_val_real, idx_val, raw_data_path, 
            os.path.join(result_dir, f'{target_year}_xgboost_{EXP_LEVEL}_fold_{fold_idx+1}_predictions.csv')
        )

    print("\n" + "=" * 60)
    print(" 產出 Train(OOF)、Test報表")
    print("=" * 60)
    
    elapsed_time = time.time() - start_time
    exec_time_str = f"{elapsed_time/60:.2f} 分鐘"

    y_train_real_million = np.exp(y_train_70)
    oof_preds_real = {k: np.exp(v) for k, v in predictions_oof.items()}
    
    save_detailed_predictions(
        predictions_oof, y_train_real_million, idx_train_70, raw_data_path, 
        os.path.join(result_dir, f'{target_year}_xgboost_{EXP_LEVEL}_oof_predictions.csv')
    )
    
    oof_eval = evaluate_predictions(y_train_real_million, oof_preds_real, quantiles, "XGBoost_Train_70_OOF")
    oof_eval['Execution_Time'] = exec_time_str
    train_report_df = pd.concat(all_fold_evaluations + [oof_eval], ignore_index=True)
    train_report_df.to_csv(os.path.join(eval_dir, f'{target_year}_xgboost_{EXP_LEVEL}_5fold_evaluation.csv'), index=False)

    y_test_real_million = np.exp(y_test_30)
    test_preds_avg = {q: (preds / 5.0) for q, preds in test_preds_sum.items()} 
    test_preds_real = {k: np.exp(v) for k, v in test_preds_avg.items()}
    
    save_detailed_predictions(
        test_preds_avg, y_test_real_million, idx_test_30, raw_data_path, 
        os.path.join(result_dir, f'{target_year}_xgboost_{EXP_LEVEL}_test_predictions.csv')
    )

    test_report_df = evaluate_predictions(y_test_real_million, test_preds_real, quantiles, "XGBoost_Test_30_Ensemble")
    test_report_df['Execution_Time'] = exec_time_str
    test_report_df.to_csv(os.path.join(eval_dir, f'{target_year}_xgboost_{EXP_LEVEL}_test_evaluation.csv'), index=False)

    print("\n" + "=" * 80)
    print(f" XGBoost - {EXP_LEVEL}")
    print("=" * 80)
    print("[1]70% Train 5-Fold OOF:")
    print(train_report_df.to_string(index=False))
    print("-" * 80)
    print("[2]30% Test 盲測:")
    print(test_report_df.to_string(index=False))

if __name__ == "__main__":
    main()