"""
05_stage1_cnn.py
CNN Quantile Regression 模型
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
import sys
import os
import pickle
import warnings
import time  # 新增：用於計算執行時間

# 確保能正確抓取上層目錄的 config 與 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import (load_config, save_predictions, evaluate_predictions, check_loss)

warnings.filterwarnings('ignore')

# 設定TensorFlow使用CPU並開啟多線程
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)


def quantile_loss(quantile):
    """分位數損失函數"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
    return loss


def build_cnn_model(input_dim, quantile='mean', learning_rate=0.001):
    """建立1D CNN模型用於表格數據"""
    model = keras.Sequential([
        # 輸入層 - 將特徵reshape為 (batch, features, 1)
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),
        
        # 第一個卷積塊
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # 第二個卷積塊
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # 第三個卷積塊
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # 全局平均池化
        layers.GlobalAveragePooling1D(),
        
        # 全連接層
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # 輸出層
        layers.Dense(1)
    ])
    
    # 選擇損失函數和優化器
    if quantile == 'mean':
        loss_fn = 'mse'
        metrics = ['mae']
    else:
        loss_fn = quantile_loss(quantile)
        metrics = []
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )
    
    return model


class CNNQuantileRegressor:
    """CNN 分位數迴歸模型"""
    
    def __init__(self, quantile, input_dim, config, random_seed=42):
        self.quantile = quantile
        self.input_dim = input_dim
        self.config = config
        self.random_seed = random_seed
        self.model = None
        
        # 設定隨機種子
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
    def fit(self, X, y, validation_split=0.2):
        """訓練模型"""
        # 建立模型
        learning_rate = self.config.get('models', {}).get('cnn', {}).get('learning_rate', 0.001)
        self.model = build_cnn_model(self.input_dim, self.quantile, learning_rate)
        
        # 設定callbacks
        patience = self.config.get('models', {}).get('cnn', {}).get('patience', 15)
        epochs = self.config.get('models', {}).get('cnn', {}).get('epochs', 100)
        batch_size = self.config.get('models', {}).get('cnn', {}).get('batch_size', 64)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # 訓練模型
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        return self, history
    
    def predict(self, X):
        """預測"""
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()


def optimize_cnn_hyperparameters(X_train, y_train, quantile, config):
    """優化CNN超參數"""
    import optuna
    from optuna.samplers import TPESampler
    # 關閉 Optuna 的 INFO 提示以保持畫面乾淨
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        kf = KFold(n_splits=3, shuffle=True, random_state=config['project']['random_seed'])
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = build_cnn_model(X_train.shape[1], quantile, learning_rate)
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
                )
            ]
            
            model.fit(
                X_tr, y_tr,
                epochs=40,  # 減少epochs以加速
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            y_pred = model.predict(X_val, verbose=0).flatten()
            
            if quantile == 'mean':
                score = np.sqrt(np.mean((y_val - y_pred) ** 2))
            else:
                score = check_loss(y_val, y_pred, quantile)
            cv_scores.append(score)
            
            del model
            keras.backend.clear_session()
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config['project']['random_seed'])
    )
    
    # 為了節省時間，CNN 的優化次數設為 15 次
    n_trials = config.get('hyperparameter_tuning', {}).get('n_trials', 15)
    # 限制 1 小時以內
    timeout = config.get('hyperparameter_tuning', {}).get('timeout', 3600)
    
    study.optimize(
        objective, 
        n_trials=min(n_trials, 20), 
        timeout=timeout,
        show_progress_bar=True
    )
    
    return study.best_params


# ============================================================
# 新增：詳細預測報表 (包含分位數交叉修正與經緯度合併)
# ============================================================
def save_detailed_predictions(predictions_dict, index_file, raw_data_path, output_path):
    print(f"整合詳細資料至: {output_path}")
    
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
        print("執行分位數交叉修正 (Rearrangement)...")
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
    
    # 動態載入配置
    config_path = os.path.join(parent_dir, 'config.yaml')
    config = load_config(config_path)
    
    print("=" * 50)
    print("CNN Quantile Regression")
    print("=" * 50)
    
    # 載入資料
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    
    # 設定正確的合併原始資料路徑
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', 'realprice_combined_all.csv')
    if not os.path.exists(raw_data_path):
        print(f"警告: 找不到合併後的原始資料 {raw_data_path}，後續合併經緯度可能會報錯。")
        
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv')).values
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv')).values
    X_full = pd.read_csv(os.path.join(processed_dir, 'X_full.csv')).values
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    y_full = np.load(os.path.join(processed_dir, 'y_full.npy'))
    
    print(f"\n訓練集: {X_train.shape}")
    print(f"測試集: {X_test.shape}")
    print(f"完整資料: {X_full.shape}")
    
    # 分位數列表
    quantiles = config['data']['quantiles']
    all_quantiles = ['mean'] + quantiles
    
    # 初始化 CNN 設定 (防呆設計)
    if 'cnn' not in config.get('models', {}):
        config.setdefault('models', {})['cnn'] = {}
        
    # 儲存所有預測結果
    predictions_test = {}
    predictions_full = {}
    
    # 訓練每個分位數的模型
    for q in all_quantiles:
        print(f"\n{'='*50}")
        print(f"訓練 Quantile = {q}")
        print(f"{'='*50}")
        
        print("執行超參數優化...")
        best_params = optimize_cnn_hyperparameters(X_train, y_train, q, config)
        print(f"最佳參數: {best_params}")
        
        # 更新配置
        config['models']['cnn']['learning_rate'] = best_params.get('learning_rate', 0.001)
        config['models']['cnn']['batch_size'] = best_params.get('batch_size', 32)
        
        # 用最佳參數訓練最終模型
        print("訓練最終模型...")
        final_model = CNNQuantileRegressor(
            q, 
            X_train.shape[1],
            config,
            config['project']['random_seed']
        )
        final_model, history = final_model.fit(X_train, y_train, validation_split=0.2)
        
        print(f"訓練完成 - 最佳 val_loss: {min(history.history['val_loss']):.4f}")
        
        print("在測試集與完整資料上預測...")
        predictions_test[q] = final_model.predict(X_test)
        predictions_full[q] = final_model.predict(X_full)
        
        # 儲存模型 (統一改用 .keras 格式)
        model_path = os.path.join(config['paths']['model_dir'], 'stage1', f'cnn_q{q}.keras')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        final_model.model.save(model_path)
        print(f"模型已儲存: {model_path}")
        
        # 清除session
        keras.backend.clear_session()
    
    print(f"\n{'='*50}")
    print("生成詳細預測報表...")
    print(f"{'='*50}")
    
    # 儲存預測結果 (包含交叉修正與經緯度)
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    
    save_detailed_predictions(
        predictions_test, 
        os.path.join(processed_dir, 'test_index.csv'),
        raw_data_path,
        os.path.join(result_dir, 'cnn_test_predictions.csv')
    )
    
    save_detailed_predictions(
        predictions_full, 
        os.path.join(processed_dir, 'full_index.csv'),
        raw_data_path,
        os.path.join(result_dir, 'cnn_full_predictions.csv')
    )
    
    print(f"\n{'='*50}")
    print("測試集評估結果 (已還原至原始房價單位)")
    print(f"{'='*50}")
    
    # 將 y_test 和 預測結果 從 Log 還原為真實金額
    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(v) for k, v in predictions_test.items()}
    
    results_df = evaluate_predictions(y_test_real, predictions_test_real, quantiles, "CNN")
    print(results_df.to_string(index=False))
    
    # 儲存評估結果
    eval_dir = os.path.join(config['paths']['result_dir'], 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    results_df.to_csv(os.path.join(eval_dir, 'cnn_evaluation.csv'), index=False)
    
    # 記錄結束時間並計算總耗時
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "=" * 50)
    print(" CNN 訓練完成!")
    print(f" 總執行時間: {int(hours)} 小時 {int(minutes)} 分鐘 {seconds:.2f} 秒")
    print("=" * 50)


if __name__ == "__main__":
    main()