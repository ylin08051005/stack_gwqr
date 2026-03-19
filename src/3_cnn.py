"""
05_stage1_cnn.py
CNN Quantile Regression 模型
學術三段式切分版 + 全樣本重訓 (供 GWQR 使用)
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers # ★ 新增 regularizers 模組
import optuna
from optuna.samplers import TPESampler
import sys
import os
import warnings
import time

# 確保能正確抓取上層目錄的 config 與 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# ★ 修正：確保 utils 裡面沒有缺少的函數
from utils import (load_config, evaluate_predictions)

warnings.filterwarnings('ignore')

# 限制線程數以維持穩定性
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def quantile_loss(quantile):
    """分位數損失函數"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
    return loss

def build_cnn_model(input_dim, quantile='mean', learning_rate=0.001):
    """建立1D CNN模型用於表格數據 (加入防爆裝甲 L2 正則化)"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),
        
        # 卷積層捕捉特徵間的局部關聯 (★ 加入 L2 正則化)
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # (★ 加入 L2 正則化)
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.GlobalAveragePooling1D(),
        
        # 全連接層 (★ 加入 L2 正則化)
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1)
    ])
    
    loss_fn = 'mse' if quantile == 'mean' else quantile_loss(quantile)
    # ★ 確保 optimizer 有 clipnorm=1.0 防止梯度爆炸
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model

def optimize_cnn_hyperparameters(X_train, y_train, X_val, y_val, quantile, config):
    """學術修正：使用單一驗證集 (X_val) 進行調參並啟用進度條"""
    seed = config['project']['random_seed']
    
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        tf.random.set_seed(seed)
        model = build_cnn_model(X_train.shape[1], quantile, learning_rate)
        
        model.fit(
            X_train, y_train,
            epochs=40,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0  # 調參時關閉細節輸出
        )
        
        y_pred = model.predict(X_val, verbose=0).flatten()
        if quantile == 'mean':
            score = np.sqrt(np.mean((y_val - y_pred) ** 2))
        else:
            # ★ 修正：直接在這裡計算 pinball loss
            error = y_val - y_pred
            score = np.mean(np.where(error >= 0, float(quantile) * error, (float(quantile) - 1) * error))
        
        keras.backend.clear_session()
        return score

    # 建立 Optuna 研究
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
    
    # 這裡啟用 show_progress_bar=True
    study.optimize(
        objective, 
        n_trials=40, 
        timeout=7200, 
        show_progress_bar=True
    )
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

        # ★ 3. 寫入真實房價
        df_output['actual_price'] = y_true_real

        # 4. 寫入各分位數預測值 (還原 Log 轉換並確保不溢位)
        quantiles = sorted([q for q in predictions_dict.keys() if q != 'mean'])
        if 'mean' in predictions_dict:
            df_output['pred_mean'] = np.expm1(predictions_dict['mean'])
        for q in quantiles:
            # ★ 安全閥：防止 expm1 數值爆炸
            df_output[f'pred_{q}'] = np.expm1(np.clip(predictions_dict[q], None, 20.0))

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
        print("✅ 預測報表儲存完成")
    except Exception as e:
        print(f"❌ 儲存報表失敗: {e}")

def main():
    start_time = time.time()
    config = load_config(os.path.join(parent_dir, 'config.yaml'))
    seed = config['project']['random_seed']
    
    print("=" * 60)
    print("CNN Quantile Regression — 效能驗證與全樣本推論版")
    print("=" * 60)
    
    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    # ★ 請確認年份設定
    target_year = '112' # 可修改
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', f'realprice_combined_{target_year}_all.csv')

    # 讀取 Train, Val, Test
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv')).values.astype(np.float32)
    X_val   = pd.read_csv(os.path.join(processed_dir, 'X_val.csv')).values.astype(np.float32)
    X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv')).values.astype(np.float32)
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy')).astype(np.float32)
    y_val   = np.load(os.path.join(processed_dir, 'y_val.npy')).astype(np.float32)
    y_test  = np.load(os.path.join(processed_dir, 'y_test.npy')).astype(np.float32)
    
    # 讀取 Index
    idx_train = pd.read_csv(os.path.join(processed_dir, 'train_index.csv')).iloc[:, 0].astype(int).values
    idx_val   = pd.read_csv(os.path.join(processed_dir, 'val_index.csv')).iloc[:, 0].astype(int).values
    idx_test  = pd.read_csv(os.path.join(processed_dir, 'test_index.csv')).iloc[:, 0].astype(int).values

    # ★ 準備全樣本資料 (供階段 B 重訓使用)
    X_all = np.vstack((X_train, X_val, X_test))
    y_all = np.concatenate([y_train, y_val, y_test])
    idx_all = np.concatenate([idx_train, idx_val, idx_test])

    print(f"訓練集: {X_train.shape} | 驗證集: {X_val.shape} | 測試集: {X_test.shape} | 全樣本: {X_all.shape}")

    all_quantiles = ['mean'] + config['data']['quantiles']
    predictions_test = {}
    predictions_full = {} # ★ 新增儲存全樣本預測

    for q in all_quantiles:
        print(f"\n>>> [CNN - {q}] 階段 1: 超參數優化 (Optuna)")
        # 調參時會顯示進度條
        best = optimize_cnn_hyperparameters(X_train, y_train, X_val, y_val, q, config)
        print(f">>> [CNN - {q}] 最佳參數: {best}")
        
        # ---------------- 階段 A：進行盲測評估訓練 ----------------
        print(f">>> [CNN - {q}] 階段 2: 訓練評估用模型 (Phase A)...")
        tf.random.set_seed(seed)
        eval_model = build_cnn_model(X_train.shape[1], q, best['learning_rate'])
        
        eval_model.fit(
            X_train, y_train, 
            epochs=100, 
            batch_size=best['batch_size'], 
            validation_data=(X_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
            ], 
            verbose=0
        )

        predictions_test[q] = eval_model.predict(X_test, verbose=0).flatten()
        
        # 儲存評估模型
        model_path = os.path.join(config['paths']['model_dir'], 'stage1', f'cnn_q{q}.keras')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        eval_model.save(model_path)
        keras.backend.clear_session()

        # ---------------- 階段 B：進行全樣本重訓 ----------------
        print(f">>> [CNN - {q}] 階段 3: 進行全樣本重新配適 (Phase B, 供 GWQR 用)...")
        tf.random.set_seed(seed)
        full_model = build_cnn_model(X_all.shape[1], q, best['learning_rate'])

        full_model.fit(
            X_all, y_all, 
            epochs=100, 
            batch_size=best['batch_size'], 
            validation_split=0.1, # ★ CNN 必備：切出 10% 監控全樣本訓練，防止過擬合
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
            ], 
            verbose=0
        )
        predictions_full[q] = full_model.predict(X_all, verbose=0).flatten()
        keras.backend.clear_session()

    # ★ 提前計算真實房價的還原數值
    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(np.clip(v, None, 20.0)) for k, v in predictions_test.items()}
    
    # 產出最終報表
    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)
    
    # ★ 輸出 Test 盲測集 CSV
    save_detailed_predictions(
        predictions_test, 
        y_test_real, 
        idx_test, 
        raw_data_path, 
        os.path.join(result_dir, 'cnn_test_predictions.csv')
    )
    
    # ★ 輸出 Full 全樣本 CSV
    y_all_real = np.expm1(y_all)
    save_detailed_predictions(
        predictions_full, 
        y_all_real, 
        idx_all, 
        raw_data_path, 
        os.path.join(result_dir, 'cnn_full_predictions.csv')
    )

    # 效能評估 (基於 Test 集)
    results_df = evaluate_predictions(y_test_real, predictions_test_real, config['data']['quantiles'], "CNN")
    
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    results_df['Execution_Time'] = time_str
    
    print("\n" + "=" * 60)
    print(" CNN 盲測結果匯總表 (模型效能驗證用)")
    print(results_df.to_string(index=False))
    print("=" * 60)

    # 儲存評估結果
    eval_dir = os.path.join(config['paths']['result_dir'], 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    results_df.to_csv(os.path.join(eval_dir, 'cnn_evaluation.csv'), index=False)

if __name__ == "__main__":
    main()