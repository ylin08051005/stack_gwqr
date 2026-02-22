"""
03_stage1_randomforest.py
Random Forest Quantile Regression 模型
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import optuna
from optuna.samplers import TPESampler
import sys
import os
import warnings
import time  # 新增：用於計算執行時間

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import (load_config, save_model, save_predictions, evaluate_predictions)

warnings.filterwarnings('ignore')


class RandomForestQuantileRegressor:
    def __init__(self, quantile, params, random_seed=42, n_jobs=-1):
        self.quantile = quantile
        self.params = params
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.model = None
        self.imputer = SimpleImputer(strategy='median')
        
    def fit(self, X, y):
        X_imputed = self.imputer.fit_transform(X)
        if isinstance(X, pd.DataFrame):
            X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

        self.model = RandomForestRegressor(
            **self.params,
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
            bootstrap=True,
        )
        self.model.fit(X_imputed, y)
        return self
    
    def predict(self, X):
        X_imputed = self.imputer.transform(X)
        
        if self.quantile == 'mean':
            return self.model.predict(X_imputed)
        else:
            all_predictions = np.array([tree.predict(X_imputed) for tree in self.model.estimators_])
            quantile_predictions = np.percentile(all_predictions, self.quantile * 100, axis=0)
            return quantile_predictions


def get_optuna_params(trial, config):
    rf_config = config['models']['random_forest']
    return {
        'n_estimators': trial.suggest_int('n_estimators', rf_config['n_estimators_range'][0], rf_config['n_estimators_range'][1]),
        'max_depth': trial.suggest_int('max_depth', rf_config['max_depth_range'][0], rf_config['max_depth_range'][1]),
        'min_samples_split': trial.suggest_int('min_samples_split', rf_config['min_samples_split_range'][0], rf_config['min_samples_split_range'][1]),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', rf_config['min_samples_leaf_range'][0], rf_config['min_samples_leaf_range'][1]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0, 0.1),
    }


def optimize_rf_hyperparameters(X_train, y_train, quantile, config):
    def objective(trial):
        params = get_optuna_params(trial, config)
        kf = KFold(n_splits=config['hyperparameter_tuning']['cv_folds'], shuffle=True, random_state=config['project']['random_seed'])
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = RandomForestQuantileRegressor(
                quantile, params, 
                config['project']['random_seed'],
                n_jobs=config['project']['n_jobs']
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            if quantile == 'mean':
                score = np.sqrt(np.mean((y_val - y_pred) ** 2))
            else:
                error = y_val - y_pred
                check_loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)
                score = np.mean(check_loss)
            cv_scores.append(score)
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=config['project']['random_seed']))
    
    study.optimize(
        objective, 
        n_trials=config['hyperparameter_tuning']['n_trials'],
        timeout=config['hyperparameter_tuning']['timeout'],
        n_jobs=1, 
        show_progress_bar=True
    )
    return study.best_params


def save_detailed_predictions(predictions_dict, index_file, raw_data_path, output_path):
    
    try:
        indices_df = pd.read_csv(index_file)
        indices = indices_df.iloc[:, 0].astype(int).values
    except Exception as e:
        print(f"讀取索引失敗 {index_file}: {e}")
        return

    df_output = pd.DataFrame(index=indices)
    raw_cols = ['土地位置建物門牌', '緯度', '經度']
    
    try:
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"找不到原始資料: {raw_data_path}")
            
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
    print("儲存完成")

def main():
    # 記錄程式開始時間
    start_time = time.time()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'config.yaml')
    config = load_config(config_path)
    
    print("=" * 50)
    print("Random Forest Quantile Regression")
    print("=" * 50)
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    
    # 修正：改為讀取 01_preprocess.py 合併後的總表
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', 'realprice_combined_all.csv')
    if not os.path.exists(raw_data_path):
        print(f"警告: 找不到合併後的原始資料 {raw_data_path}，請確認是否已執行 01_preprocess.py")
    else:
        print(f"原始資料: {raw_data_path}")
    
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
        
        print("超參數優化")
        best_params = optimize_rf_hyperparameters(X_train, y_train, q, config)
        print(f"最佳參數: {best_params}")
        
        print("訓練模型")
        final_model = RandomForestQuantileRegressor(
            q, best_params, 
            config['project']['random_seed'],
            config['project']['n_jobs']
        )
        final_model.fit(X_train, y_train)

        predictions_test[q] = final_model.predict(X_test)
        predictions_full[q] = final_model.predict(X_full)
        
        model_path = os.path.join(config['paths']['model_dir'], 'stage1', f'rf_q{q}.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(final_model.model, model_path)
        print(f"模型已儲存: {model_path}")
    
    
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    
    save_detailed_predictions(
        predictions_test,
        os.path.join(processed_dir, 'test_index.csv'),
        raw_data_path,
        os.path.join(result_dir, 'rf_test_predictions.csv')
    )
    
    save_detailed_predictions(
        predictions_full,
        os.path.join(processed_dir, 'full_index.csv'),
        raw_data_path,
        os.path.join(result_dir, 'rf_full_predictions.csv')
    )
    
    print(f"\n{'='*50}")
    print("測試集評估")
    print(f"{'='*50}")
    
    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(v) for k, v in predictions_test.items()}
    
    results_df = evaluate_predictions(y_test_real, predictions_test_real, quantiles, "RandomForest")
    
    # =========================================================
    # 關鍵修改：提前計算執行時間，並新增為 CSV 的一個新欄位
    # =========================================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)} 小時 {int(minutes)} 分鐘 {seconds:.2f} 秒"
    
    # 將時間寫入 DataFrame
    results_df['Execution_Time'] = time_str
    # =========================================================

    print(results_df.to_string(index=False))
    
    eval_dir = os.path.join(config['paths']['result_dir'], 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    results_df.to_csv(os.path.join(eval_dir, 'rf_evaluation.csv'), index=False)
    
    print("\n" + "=" * 50)
    print(" Random Forest訓練完成")
    print(f" 總執行時間: {time_str}")
    print("=" * 50)


if __name__ == "__main__":
    main()