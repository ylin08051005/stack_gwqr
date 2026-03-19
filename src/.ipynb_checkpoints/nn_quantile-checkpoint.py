"""
05_stage1_nn.py
Simple Neural Network (NN) Quantile Regression 模型
"""
import os
import sys
import warnings
import time  # 新增：用於計算執行時間
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import load_config, save_model, evaluate_predictions

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)


def quantile_loss_fn(tau):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1.0) * e))
    return loss


from tensorflow.keras import regularizers # 新增這行

def build_nn(input_dim, quantile, lr, units_1=64, units_2=32,
             dropout_1=0.2, dropout_2=0.2):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units_1, activation='relu',
                     kernel_initializer='he_normal', 
                     kernel_regularizer=regularizers.l2(1e-4), # ★ 安全閥 1: L2 權重懲罰
                     name='dense_1'),
        layers.BatchNormalization(), # ★ 安全閥 2: 批次標準化，穩定數值
        layers.Dropout(dropout_1, name='dropout_1'),
        layers.Dense(units_2, activation='relu',
                     kernel_initializer='he_normal', 
                     kernel_regularizer=regularizers.l2(1e-4), # ★ 安全閥 1
                     name='dense_2'),
        layers.BatchNormalization(), # ★ 安全閥 2
        layers.Dropout(dropout_2, name='dropout_2'),
        layers.Dense(1, name='output'),
    ])

    loss = 'mse' if quantile == 'mean' else quantile_loss_fn(float(quantile))
    # ★ 安全閥 3: clipnorm=1.0 限制梯度爆炸
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0) 
    model.compile(optimizer=optimizer, loss=loss)
    return model


def optimize_nn(X_train, y_train, quantile, config):
    seed   = config['project']['random_seed']
    n_fold = 3
    n_trials = config['hyperparameter_tuning']['n_trials']
    timeout  = config['hyperparameter_tuning']['timeout']
    input_dim = X_train.shape[1]

    def objective(trial):
        lr       = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        units_1  = trial.suggest_categorical('units_1', [32, 64, 128])
        units_2  = trial.suggest_categorical('units_2', [16, 32, 64])
        dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.4)
        dropout_2 = trial.suggest_float('dropout_2', 0.1, 0.4)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        fold_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            tf.random.set_seed(seed)
            model = build_nn(input_dim, quantile, lr,
                             units_1, units_2, dropout_1, dropout_2)

            model.fit(X_tr, y_tr,
                      epochs=80,
                      batch_size=batch_size,
                      validation_data=(X_val, y_val),
                      callbacks=[
                          keras.callbacks.EarlyStopping(
                              monitor='val_loss', patience=15,
                              restore_best_weights=True, verbose=0)
                      ],
                      verbose=0)

            y_pred = model.predict(X_val, verbose=0).flatten()

            if quantile == 'mean':
                score = np.sqrt(np.mean((y_val - y_pred) ** 2))
            else:
                e = y_val - y_pred
                score = np.mean(np.where(e >= 0,
                                         float(quantile) * e,
                                         (float(quantile) - 1) * e))
            fold_scores.append(score)

            del model
            keras.backend.clear_session()

        return np.mean(fold_scores)

    study = optuna.create_study(direction='minimize',
                                sampler=TPESampler(seed=seed))
    
    study.optimize(objective,
                   n_trials=n_trials,
                   timeout=timeout,
                   n_jobs=1,
                   show_progress_bar=True)
    return study.best_params


def save_detailed_predictions(predictions_dict, index_file,
                              raw_data_path, output_path):
    print(f"整合詳細資料至: {output_path}")

    try:
        indices = pd.read_csv(index_file).iloc[:, 0].astype(int).values
    except Exception as e:
        print(f"錯誤：讀取索引檔失敗 {index_file}: {e}")
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
        print(f"警告：合併原始資料錯誤 ({e})")

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
        # ★ 安全閥: 同樣在這裡限制最大值
        df_output[col_name] = np.expm1(np.clip(preds, a_min=None, a_max=20.0))
        quantile_cols.append(col_name)

    if len(quantile_cols) > 1:
        print("執行分位數交叉修正 (Rearrangement)...")
        q_values = df_output[quantile_cols].values
        q_values.sort(axis=1) 
        df_output[quantile_cols] = q_values

    cols = list(df_output.columns)
    priority = ['土地位置建物門牌', '緯度', '經度']
    new_order = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
    df_output = df_output[new_order]

    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("儲存完成。")


def main():
    # 記錄程式開始時間
    start_time = time.time()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'config.yaml')
    config = load_config(config_path)

    seed = config['project']['random_seed']

    print("=" * 60)
    print(" Simple Neural Network (NN) — Quantile Regression")
    print("=" * 60)

    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    
    # 修正：改為讀取 01_preprocess.py 合併後的總表
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', 'realprice_combined_all.csv')
    if not os.path.exists(raw_data_path):
        print(f"警告: 找不到合併後的原始資料 {raw_data_path}，請確認是否已執行 01_preprocess.py")
    else:
        print(f"原始資料: {raw_data_path}")

    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv')).values.astype(np.float32)
    X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv')).values.astype(np.float32)
    X_full  = pd.read_csv(os.path.join(processed_dir, 'X_full.csv')).values.astype(np.float32)
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy')).astype(np.float32)
    y_test  = np.load(os.path.join(processed_dir, 'y_test.npy')).astype(np.float32)
    y_full  = np.load(os.path.join(processed_dir, 'y_full.npy')).astype(np.float32)

    print(f"\n訓練集: {X_train.shape}  |  測試集: {X_test.shape}  |  完整: {X_full.shape}")

    quantiles     = config['data']['quantiles']
    all_quantiles = ['mean'] + quantiles
    input_dim     = X_train.shape[1]

    predictions_test = {}
    predictions_full = {}

    for q in all_quantiles:
        print(f"\n{'─'*60}")
        print(f" 訓練 Quantile = {q}")
        print(f"{'─'*60}")

        print("  Optuna 超參數搜索中 ...")
        best = optimize_nn(X_train, y_train, q, config)
        print(f"  最佳參數: {best}")

        print("  訓練最終模型 ...")
        np.random.seed(seed)
        tf.random.set_seed(seed)

        final_model = build_nn(
            input_dim, q,
            lr=best['lr'],
            units_1=best['units_1'],
            units_2=best['units_2'],
            dropout_1=best['dropout_1'],
            dropout_2=best['dropout_2']
        )

        final_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=best['batch_size'],
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=20,
                    restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5,
                    patience=8, min_lr=1e-7, verbose=0)
            ],
            verbose=0
        )

        predictions_test[q] = final_model.predict(X_test, verbose=0).flatten()
        predictions_full[q] = final_model.predict(X_full, verbose=0).flatten()

        model_path = os.path.join(config['paths']['model_dir'],
                                  'stage1', f'nn_q{q}.keras')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        final_model.save(model_path)
        print(f"  模型已儲存: {model_path}")

        del final_model
        keras.backend.clear_session()

    print(f"\n{'='*60}")
    print(" 生成詳細預測報表 ...")
    print(f"{'='*60}")

    result_dir = os.path.join(config['paths']['result_dir'], 'predictions')
    os.makedirs(result_dir, exist_ok=True)

    save_detailed_predictions(
        predictions_test,
        os.path.join(processed_dir, 'test_index.csv'),
        raw_data_path,
        os.path.join(result_dir, 'nn_test_predictions.csv')
    )
    save_detailed_predictions(
        predictions_full,
        os.path.join(processed_dir, 'full_index.csv'),
        raw_data_path,
        os.path.join(result_dir, 'nn_full_predictions.csv')
    )

    print(f"\n{'='*60}")
    print(" 測試集評估結果（已還原至原始房價單位）")
    print(f"{'='*60}")

    y_test_real = np.expm1(y_test)
    
    # ★ 安全閥 4: 限制 log 預測值最大不超過 20.0 (約 4.8 億台幣)，防止 expm1 溢位爆炸
    predictions_test_real = {k: np.expm1(np.clip(v, a_min=None, a_max=20.0)) for k, v in predictions_test.items()}

    results_df = evaluate_predictions(y_test_real, predictions_test_real,
                                      quantiles, "NN")
    
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
    results_df.to_csv(os.path.join(eval_dir, 'nn_evaluation.csv'), index=False)

    print("\n" + "=" * 60)
    print(" NN 訓練完成!")
    print(f" 總執行時間: {time_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()