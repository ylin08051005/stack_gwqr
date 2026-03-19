"""
05_stage1_nn.py
Simple Neural Network (NN) Quantile Regression 模型
學術三段式切分版 + 全樣本重訓 (供 GWQR 使用)
"""
import os
import sys
import warnings
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler # [新增] 引入標準化工具

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import load_config, save_model, evaluate_predictions

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def smooth_quantile_loss_fn(tau, delta=0.01):
    """
    [新增] 平滑分位數損失函數 (Huberized Quantile Loss)
    結合 Huber Loss 的特性，當誤差極小時平滑梯度，避免極端分位數 (0.1, 0.9) 權重震盪爆炸
    """
    def loss(y_true, y_pred):
        e = y_true - y_pred
        # 判斷是否在平滑區間內
        is_small_error = tf.abs(e) <= delta
        # Huber 轉換
        huber_loss = tf.where(is_small_error, 0.5 * tf.square(e) / delta, tf.abs(e) - 0.5 * delta)
        # 套用分位數權重
        return tf.reduce_mean(tf.where(e >= 0, tau * huber_loss, (1.0 - tau) * huber_loss))
    return loss

def build_nn(input_dim, quantile, lr, y_train_mean, units_1=64, units_2=32,
             dropout_1=0.2, dropout_2=0.2):
    """建立基礎全連接神經網路"""
    
    # [新增] 輸出層偏差初始化技巧 (Bias Initialization)
    # 讓模型一開始就從「平均房價(Log)」起跑，大幅減少前期亂猜導致的梯度爆炸
    output_bias = keras.initializers.Constant(value=y_train_mean)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units_1, activation='relu', kernel_initializer='he_normal', 
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_1),
        layers.Dense(units_2, activation='relu', kernel_initializer='he_normal', 
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_2),
        layers.Dense(1, bias_initializer=output_bias) # 套用初始化
    ])

    # 套用平滑版損失函數
    loss = 'huber' if quantile == 'mean' else smooth_quantile_loss_fn(float(quantile))
    
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0, clipvalue=3.0) 
    model.compile(optimizer=optimizer, loss=loss)
    return model

def optimize_nn(X_train, y_train, X_val, y_val, quantile, config):
    """使用單一驗證集 (X_val) 進行超參數優化，並開啟進度條"""
    seed = config['project']['random_seed']
    input_dim = X_train.shape[1]
    y_train_mean = np.mean(y_train)

    def objective(trial):
        # [修正] 降低學習率上限，0.01 對於極端分位數來說太暴躁了
        lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
        units_1 = trial.suggest_categorical('units_1', [32, 64, 128])
        units_2 = trial.suggest_categorical('units_2', [16, 32, 64])
        dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.4)
        dropout_2 = trial.suggest_float('dropout_2', 0.1, 0.4)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        tf.random.set_seed(seed)
        model = build_nn(input_dim, quantile, lr, y_train_mean, units_1, units_2, dropout_1, dropout_2)

        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=15,
                    restore_best_weights=True, verbose=0)
            ],
            verbose=0
        )

        y_pred = model.predict(X_val, verbose=0).flatten()

        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            keras.backend.clear_session()
            return float('inf')

        if quantile == 'mean':
            score = np.sqrt(np.mean((y_val - y_pred) ** 2))
        else:
            e = y_val - y_pred
            score = np.mean(np.where(e >= 0, float(quantile) * e, (float(quantile) - 1) * e))

        keras.backend.clear_session()
        return score

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
    study.optimize(
        objective, 
        n_trials=config['hyperparameter_tuning']['n_trials'], 
        timeout=config['hyperparameter_tuning']['timeout'],
        show_progress_bar=True
    )
    return study.best_params

def save_detailed_predictions(predictions_dict, y_true_real, indices,
                              raw_data_path, output_path):
    print(f"正在整合詳細預測資料至: {output_path}")
    try:
        df_output = pd.DataFrame(index=indices)
        df_raw = pd.read_csv(raw_data_path)
        
        raw_cols = ['土地位置建物門牌', '緯度', '經度']
        df_output = df_output.join(df_raw[raw_cols], how='left')
        df_output['actual_price'] = y_true_real

        quantiles = sorted([q for q in predictions_dict.keys() if q != 'mean'])
        
        if 'mean' in predictions_dict:
            df_output['pred_mean'] = np.expm1(np.clip(predictions_dict['mean'], -20.0, 20.0))
            
        for q in quantiles:
            df_output[f'pred_{q}'] = np.expm1(np.clip(predictions_dict[q], -20.0, 20.0))

        pred_cols = [f'pred_{q}' for q in quantiles]
        if len(pred_cols) > 1:
            q_values = df_output[pred_cols].values
            q_values.sort(axis=1)
            df_output[pred_cols] = q_values

        final_cols = ['土地位置建物門牌', '緯度', '經度', 'actual_price']
        if 'mean' in predictions_dict:
            final_cols.append('pred_mean')
        final_cols.extend(pred_cols)

        df_output = df_output[final_cols]
        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
        print("✅ 詳細報表儲存完成。")
    except Exception as e:
        print(f"❌ 警告：儲存詳細預測失敗 ({e})")

def main():
    start_time = time.time()
    config = load_config(os.path.join(parent_dir, 'config.yaml'))
    seed = config['project']['random_seed']

    print("=" * 60)
    print(" Simple Neural Network (NN) — 效能驗證與全樣本推論版")
    print("=" * 60)

    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    target_year = '112' # L1 模型年份
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', f'realprice_combined_{target_year}_all.csv')

    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv')).values.astype(np.float32)
    X_val   = pd.read_csv(os.path.join(processed_dir, 'X_val.csv')).values.astype(np.float32)
    X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv')).values.astype(np.float32)
    
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy')).astype(np.float32)
    y_val   = np.load(os.path.join(processed_dir, 'y_val.npy')).astype(np.float32)
    y_test  = np.load(os.path.join(processed_dir, 'y_test.npy')).astype(np.float32)

    # =========================================================
    # [新增] 特徵標準化 (Crucial for NN stability)
    # =========================================================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    idx_train = pd.read_csv(os.path.join(processed_dir, 'train_index.csv')).iloc[:, 0].astype(int).values
    idx_val   = pd.read_csv(os.path.join(processed_dir, 'val_index.csv')).iloc[:, 0].astype(int).values
    idx_test  = pd.read_csv(os.path.join(processed_dir, 'test_index.csv')).iloc[:, 0].astype(int).values

    # 全樣本也需要標準化
    X_all = np.vstack((X_train, X_val, X_test)) 
    y_all = np.concatenate([y_train, y_val, y_test])
    idx_all = np.concatenate([idx_train, idx_val, idx_test])

    print(f"訓練集: {X_train.shape} | 驗證集: {X_val.shape} | 測試集: {X_test.shape} | 全樣本: {X_all.shape}")

    all_quantiles = ['mean'] + config['data']['quantiles']
    predictions_test = {}
    predictions_full = {} 
    
    y_train_mean = np.mean(y_train)

    for q in all_quantiles:
        print(f"\n>>> [Quantile {q}] 階段 1: 超參數優化中...")
        best = optimize_nn(X_train, y_train, X_val, y_val, q, config)
        print(f">>> [Quantile {q}] 最佳參數: {best}")

        # ---------------- 階段 A ----------------
        print(f">>> [Quantile {q}] 階段 2: 訓練評估用模型 (Phase A)...")
        tf.random.set_seed(seed)
        eval_model = build_nn(X_train.shape[1], q, lr=best['lr'], y_train_mean=y_train_mean,
                               units_1=best['units_1'], units_2=best['units_2'], 
                               dropout_1=best['dropout_1'], dropout_2=best['dropout_2'])

        eval_model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=best['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[
                keras.callbacks.TerminateOnNaN(), 
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, 
                                              restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                                  patience=8, min_lr=1e-7, verbose=0)
            ],
            verbose=0
        )

        predictions_test[q] = eval_model.predict(X_test, verbose=0).flatten()
        eval_model.save(os.path.join(config['paths']['model_dir'], 'stage1', f'nn_q{q}.keras'))
        keras.backend.clear_session()

        # ---------------- 階段 B ----------------
        print(f">>> [Quantile {q}] 階段 3: 進行全樣本重新配適 (Phase B, 供 GWQR 用)...")
        tf.random.set_seed(seed)
        full_model = build_nn(X_all.shape[1], q, lr=best['lr'], y_train_mean=y_train_mean,
                               units_1=best['units_1'], units_2=best['units_2'], 
                               dropout_1=best['dropout_1'], dropout_2=best['dropout_2'])

        full_model.fit(
            X_all, y_all,
            epochs=150,
            batch_size=best['batch_size'],
            validation_split=0.1, 
            shuffle=True, 
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, 
                                              restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                                  patience=8, min_lr=1e-7, verbose=0)
            ],
            verbose=0
        )
        predictions_full[q] = full_model.predict(X_all, verbose=0).flatten()
        keras.backend.clear_session()

    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(np.clip(v, -20.0, 20.0)) for k, v in predictions_test.items()}

    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    
    save_detailed_predictions(
        predictions_test, y_test_real, idx_test, raw_data_path, 
        os.path.join(result_dir, 'nn_test_predictions.csv')
    )
    
    y_all_real = np.expm1(y_all)
    save_detailed_predictions(
        predictions_full, y_all_real, idx_all, raw_data_path, 
        os.path.join(result_dir, 'nn_full_predictions.csv')
    )

    results_df = evaluate_predictions(y_test_real, predictions_test_real, config['data']['quantiles'], "NN")

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    
    results_df['Execution_Time'] = time_str
    print("\n" + "=" * 60)
    print(" 神經網路 (NN) 測試集評估報表 (模型效能驗證用)")
    print(results_df.to_string(index=False))
    print("=" * 60)

    results_df.to_csv(os.path.join(config['paths']['result_dir'], 'evaluation', 'nn_evaluation.csv'), index=False)
    print(f"訓練與預測完成！總執行時間: {time_str}")

if __name__ == "__main__":
    main()