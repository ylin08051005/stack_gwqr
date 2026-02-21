"""
02_stage1_xgboost.py
XGBoost Quantile Regression 模型
修正: 
1. 解決 n_estimators 警告
2. 評估指標 (RMSE) 還原為原始房價單位 (Real Scale)
3. Mean 模型讀取 config 設定
4. [本次修正] 輸出 CSV 包含: [土地位置建物門牌, 緯度, 經度, 還原後的預測價格]
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

# 加入上層目錄以匯入 utils
sys.path.append('..')
from utils import (load_config, save_model, save_predictions, 
                   xgb_quantile_obj, evaluate_predictions)

warnings.filterwarnings('ignore')


class XGBoostQuantileRegressor:
    """XGBoost 分位數迴歸模型"""
    
    def __init__(self, quantile, params, random_seed=42):
        self.quantile = quantile
        self.params = params
        self.random_seed = random_seed
        self.model = None
        
    def fit(self, X, y):
        """訓練模型"""
        dtrain = xgb.DMatrix(X, label=y)
        
        # 設定參數
        params = self.params.copy()
        params['seed'] = self.random_seed
        params['objective'] = 'reg:quantileerror'
        params['quantile_alpha'] = self.quantile
        params['tree_method'] = 'hist'
        params['device'] = 'cpu'
        
        # 分離 n_estimators
        n_estimators = params.pop('n_estimators', 100)
        
        # 訓練模型
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            verbose_eval=False
        )
        
        return self
    
    def predict(self, X):
        """預測"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)


def get_optuna_params(trial, config):
    """從 config.yaml 讀取 XGBoost 參數範圍"""
    xgb_config = config['models']['xgboost']
    
    return {
        'n_estimators': trial.suggest_int(
            'n_estimators',
            xgb_config['n_estimators_range'][0],
            xgb_config['n_estimators_range'][1]
        ),
        'max_depth': trial.suggest_int(
            'max_depth',
            xgb_config['max_depth_range'][0],
            xgb_config['max_depth_range'][1]
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate',
            xgb_config['learning_rate_range'][0],
            xgb_config['learning_rate_range'][1],
            log=True
        ),
        'subsample': trial.suggest_float(
            'subsample',
            xgb_config['subsample_range'][0],
            xgb_config['subsample_range'][1]
        ),
        'colsample_bytree': trial.suggest_float(
            'colsample_bytree',
            xgb_config['colsample_bytree_range'][0],
            xgb_config['colsample_bytree_range'][1]
        ),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }


def optimize_xgboost_hyperparameters(X_train, y_train, quantile, config):
    """使用Optuna優化XGBoost超參數"""
    
    def objective(trial):
        params = get_optuna_params(trial, config)
        
        kf = KFold(n_splits=config['hyperparameter_tuning']['cv_folds'], 
                   shuffle=True, 
                   random_state=config['project']['random_seed'])
        
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
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config['project']['random_seed'])
    )
    
    study.optimize(
        objective, 
        n_trials=config['hyperparameter_tuning']['n_trials'],
        timeout=config['hyperparameter_tuning']['timeout'],
        n_jobs=config['project']['n_jobs'],
        show_progress_bar=True
    )
    
    return study.best_params


def save_detailed_predictions(predictions_dict, index_file, raw_data_path, output_path):
    """
    客製化儲存函數：結合原始資料的地址、經緯度與預測結果
    """
    print(f"正在整合詳細資料至: {output_path}")
    
    # 1. 讀取 Index 對照表 (確保讀取為整數，避免對位錯誤)
    try:
        indices_df = pd.read_csv(index_file)
        indices = indices_df.iloc[:, 0].astype(int).values
    except Exception as e:
        print(f"錯誤：讀取索引檔失敗 {index_file}: {e}")
        return

    # 2. 準備輸出的 DataFrame 框架 (以預測的 Index 為主)
    df_output = pd.DataFrame(index=indices)
    
    # 3. 讀取原始資料並合併
    # 指定需要的原始欄位
    raw_cols = ['土地位置建物門牌', '緯度', '經度']
    
    try:
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"找不到原始資料: {raw_data_path}")
            
        df_raw = pd.read_csv(raw_data_path)
        
        # 確保原始資料的 index 也是整數，這樣 .join 才能正確運作
        # 假設原始資料讀取進來時的 index 就是 0, 1, 2... (即原始行號)
        
        # 使用 join (Left Join) 將原始資料的地址經緯度併進來
        # 這比 .loc 更安全，如果原始資料缺漏也不會報錯 (會填 NaN)
        df_merged = df_output.join(df_raw[raw_cols], how='left')
        
        # 更新 df_output 為合併後的結果
        df_output = df_merged
        
    except KeyError as e:
        print(f"警告：原始資料中找不到指定欄位 {e}，請檢查 CSV 標頭。")
    except Exception as e:
        print(f"警告：合併原始資料時發生錯誤 ({e})，輸出將僅包含預測值。")

    # 4. 將預測結果 (Log Scale) 轉回 Real Scale 並加入 DataFrame
    # 這裡的 key 是 'mean', 0.1, 0.25...
    for q, preds in predictions_dict.items():
        if len(preds) != len(df_output):
            print(f"警告：預測值數量 ({len(preds)}) 與資料行數 ({len(df_output)}) 不符，跳過 {q}")
            continue
            
        # np.expm1 將 log(1+x) 轉回 x (真實房價)
        df_output[f'pred_{q}'] = np.expm1(preds)

    # 5. 調整欄位順序 (美觀用)
    # 嘗試把地址經緯度放到最前面
    cols = list(df_output.columns)
    priority_cols = ['土地位置建物門牌', '緯度', '經度']
    new_order = [c for c in priority_cols if c in cols] + [c for c in cols if c not in priority_cols]
    df_output = df_output[new_order]

    # 6. 儲存 (utf-8-sig 確保中文不亂碼)
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("儲存完成。")


def main():
    config = load_config('../config.yaml')
    
    print("=" * 50)
    print("XGBoost Quantile Regression")
    print("=" * 50)
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    
    # [設定] 原始資料路徑 (根據您的需求設定絕對路徑或相對路徑)
    # 優先嘗試 config 路徑，如果找不到則嘗試您提供的絕對路徑
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', 'realprice_with_aqi_idw1122.csv')
    if not os.path.exists(raw_data_path):
        # 如果 config 路徑找不到，嘗試您提供的絕對路徑
        raw_data_path = "/Users/ylin/Documents/stack_gwqr/src/data/raw/realprice_with_aqi_idw1122.csv"
    
    print(f"原始資料路徑: {raw_data_path}")

    # 載入處理後的特徵
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
            print("執行超參數優化 (Mean - RMSE)...")
            
            def objective_mean(trial):
                params = get_optuna_params(trial, config)
                kf = KFold(n_splits=config['hyperparameter_tuning']['cv_folds'], 
                           shuffle=True, 
                           random_state=config['project']['random_seed'])
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
            print("執行超參數優化 (Quantile - Check Loss)...")
            best_params = optimize_xgboost_hyperparameters(X_train, y_train, q, config)
            print(f"最佳參數: {best_params}")
            
            final_model_wrapper = XGBoostQuantileRegressor(q, best_params, config['project']['random_seed'])
            final_model_wrapper.fit(X_train, y_train)
            final_model = final_model_wrapper.model
        
        # 預測 (此時仍為 Log Scale)
        dtest = xgb.DMatrix(X_test)
        predictions_test[q] = final_model.predict(dtest)
        
        dfull = xgb.DMatrix(X_full)
        predictions_full[q] = final_model.predict(dfull)
        
        # 儲存模型
        model_path = os.path.join(config['paths']['model_dir'], 'stage1', f'xgboost_q{q}.json')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        final_model.save_model(model_path)
        print(f"模型已儲存: {model_path}")
    
    # ==========================================================
    # [修正] 儲存詳細預測結果 (含地址、經緯度、還原後的價格)
    # ==========================================================
    print(f"\n{'='*50}")
    print("正在生成詳細預測報表...")
    print(f"{'='*50}")
    
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    
    # 儲存測試集詳細結果
    save_detailed_predictions(
        predictions_test,
        os.path.join(processed_dir, 'test_index.csv'), # Index 檔案
        raw_data_path,                                 # 原始資料檔案 (從上面變數抓)
        os.path.join(result_dir, 'xgboost_test_predictions.csv') # 輸出路徑
    )
    
    # 儲存完整資料詳細結果
    save_detailed_predictions(
        predictions_full,
        os.path.join(processed_dir, 'full_index.csv'), # Index 檔案
        raw_data_path,                                 # 原始資料檔案 (從上面變數抓)
        os.path.join(result_dir, 'xgboost_full_predictions.csv') # 輸出路徑
    )
    
    # ==========================================================
    # 評估 (維持使用還原後的 Real Scale 進行數學評估)
    # ==========================================================
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
    
    print("\n" + "=" * 50)
    print("XGBoost訓練完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()