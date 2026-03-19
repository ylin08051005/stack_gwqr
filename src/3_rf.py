"""
03_stage1_randomforest.py
Random Forest Quantile Regression 模型
學術三段式切分版 + 全樣本重訓 (供 GWQR 使用)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
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
            # 取出所有決策樹的預測值來計算分位數
            all_predictions = np.array([tree.predict(X_imputed) for tree in self.model.estimators_])
            return np.percentile(all_predictions, self.quantile * 100, axis=0)

def get_optuna_params(trial, config):
    rf_config = config['models']['random_forest']
    return {
        'n_estimators': trial.suggest_int('n_estimators', rf_config['n_estimators_range'][0], rf_config['n_estimators_range'][1]),
        'max_depth': trial.suggest_int('max_depth', rf_config['max_depth_range'][0], rf_config['max_depth_range'][1]),
        'min_samples_split': trial.suggest_int('min_samples_split', rf_config['min_samples_split_range'][0], rf_config['min_samples_split_range'][1]),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', rf_config['min_samples_leaf_range'][0], rf_config['min_samples_leaf_range'][1]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }

def save_detailed_predictions(predictions_dict, y_true_real, indices, raw_data_path, output_path):
    """
    將預測結果與原始經緯度、門牌資訊整合儲存，並確保指定的欄位順序。
    (改為直接傳入 indices 以支援 Test 與 Full 雙重輸出)
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

        # 5. 分位數交叉修正
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
        print("✅ 儲存完成")
    except Exception as e:
        print(f"❌ 儲存報表失敗: {e}")

def main():
    start_time = time.time()
    config = load_config(os.path.join(parent_dir, 'config.yaml'))
    
    print("=" * 60)
    print("Random Forest Quantile Regression — 效能驗證與全樣本推論版")
    print("=" * 60)
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    # ★ 請確認這支檔案的年份與 data_preparation.py 產生的一致
    target_year = '112' # 可修改
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', f'realprice_combined_{target_year}_all.csv')

    # 讀取學術切分的三段數據
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
        def objective(trial):
            params = get_optuna_params(trial, config)
            model = RandomForestQuantileRegressor(q, params, config['project']['random_seed'], n_jobs=config['project']['n_jobs'])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            if q == 'mean':
                return np.sqrt(np.mean((y_val - y_pred) ** 2))
            else:
                error = y_val - y_pred
                return np.mean(np.where(error >= 0, float(q) * error, (float(q) - 1) * error))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=config['project']['random_seed']))
        study.optimize(objective, n_trials=config['hyperparameter_tuning']['n_trials'], timeout=config['hyperparameter_tuning']['timeout'], show_progress_bar=True)
        best_params = study.best_params
        print(f"最佳參數: {best_params}")
        
        # ---------------- 階段 A：進行盲測評估訓練 ----------------
        print("  - 進行盲測評估訓練...")
        eval_model = RandomForestQuantileRegressor(q, best_params, config['project']['random_seed'], config['project']['n_jobs'])
        eval_model.fit(X_train, y_train)
        predictions_test[q] = eval_model.predict(X_test)
        
        # 儲存評估模型
        model_path = os.path.join(config['paths']['model_dir'], 'stage1', f'rf_q{q}.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(eval_model.model, model_path)

        # ---------------- 階段 B：進行全樣本重訓 ----------------
        print("  - [GWQR 用] 進行全樣本重新配適...")
        full_model = RandomForestQuantileRegressor(q, best_params, config['project']['random_seed'], config['project']['n_jobs'])
        full_model.fit(X_all, y_all)
        predictions_full[q] = full_model.predict(X_all)

    # 1. 輸出評估報表 (基於 Test 集)
    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(v) for k, v in predictions_test.items()}
    results_df = evaluate_predictions(y_test_real, predictions_test_real, config['data']['quantiles'], "RandomForest")
    
    # 2. 儲存預測 CSV 檔案
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    
    # ★ 輸出 Test 盲測集 CSV
    save_detailed_predictions(
        predictions_test, y_test_real, idx_test, raw_data_path, 
        os.path.join(result_dir, 'rf_test_predictions.csv')
    )
    
    # ★ 輸出 Full 全樣本 CSV
    y_all_real = np.expm1(y_all)
    save_detailed_predictions(
        predictions_full, y_all_real, idx_all, raw_data_path, 
        os.path.join(result_dir, 'rf_full_predictions.csv')
    )
    
    # 3. 紀錄執行時間
    elapsed_time = time.time() - start_time
    results_df['Execution_Time'] = f"{elapsed_time/60:.2f} 分鐘"
    
    print("\n" + "=" * 60)
    print(" Random Forest 測試集評估報表 (模型效能驗證用)")
    print(results_df.to_string(index=False))
    print("=" * 60)
    
    results_df.to_csv(os.path.join(config['paths']['result_dir'], 'evaluation', 'rf_evaluation.csv'), index=False)

if __name__ == "__main__":
    main()