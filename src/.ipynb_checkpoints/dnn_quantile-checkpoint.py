"""
05_stage1_dnn.py
Deep Neural Network (DNN) Quantile Regression 模型
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


def build_dnn(input_dim, quantile, lr,
              units=(256, 128, 64, 32),
              dropouts=(0.3, 0.3, 0.2, 0.2),
              use_bn=True):
    
    inp = layers.Input(shape=(input_dim,), name='input')

    x = layers.Dense(units[0], activation='relu', kernel_initializer='he_normal')(inp)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropouts[0])(x)
    skip = x 

    x = layers.Dense(units[1], activation='relu', kernel_initializer='he_normal')(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropouts[1])(x)

    if units[0] != units[1]:
        skip_proj = layers.Dense(units[1], use_bias=False, kernel_initializer='he_normal')(skip)
    else:
        skip_proj = skip
    x = layers.Add()([x, skip_proj])

    for i in range(2, len(units)):
        x = layers.Dense(units[i], activation='relu', kernel_initializer='he_normal')(x)
        if use_bn: x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropouts[i])(x)

    out = layers.Dense(1, name='output')(x)
    model = keras.Model(inputs=inp, outputs=out)

    loss = 'mse' if quantile == 'mean' else quantile_loss_fn(float(quantile))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
    return model


UNIT_CONFIGS = [
    (256, 128, 64, 32),
    (128, 64,  32, 16),
    (512, 256, 128, 64),
    (128, 128, 64, 32),
]

def optimize_dnn(X_train, y_train, quantile, config):
    seed      = config['project']['random_seed']
    n_trials  = config['hyperparameter_tuning']['n_trials']
    timeout   = config['hyperparameter_tuning']['timeout']
    input_dim = X_train.shape[1]

    def objective(trial):
        lr         = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        unit_idx   = trial.suggest_categorical('unit_config_idx', list(range(len(UNIT_CONFIGS))))
        units      = UNIT_CONFIGS[unit_idx]
        drop_base  = trial.suggest_float('drop_base', 0.1, 0.4)
        dropouts   = (drop_base, drop_base, max(0.05, drop_base - 0.1), max(0.05, drop_base - 0.1))
        use_bn     = trial.suggest_categorical('use_bn', [True, False])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        kf = KFold(n_splits=3, shuffle=True, random_state=seed)
        fold_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            tf.random.set_seed(seed)
            model = build_dnn(input_dim, quantile, lr, units, dropouts, use_bn)

            model.fit(X_tr, y_tr, epochs=80, batch_size=batch_size,
                      validation_data=(X_val, y_val),
                      callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)],
                      verbose=0)

            y_pred = model.predict(X_val, verbose=0).flatten()

            if quantile == 'mean':
                score = np.sqrt(np.mean((y_val - y_pred) ** 2))
            else:
                e = y_val - y_pred
                score = np.mean(np.where(e >= 0, float(quantile) * e, (float(quantile) - 1) * e))
            fold_scores.append(score)

            del model
            keras.backend.clear_session()

        return np.mean(fold_scores)

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1, show_progress_bar=True)
    return study.best_params


def save_detailed_predictions(predictions_dict, index_file, raw_data_path, output_path):
    print(f"整合詳細資料至: {output_path}")

    try:
        indices = pd.read_csv(index_file).iloc[:, 0].astype(int).values
    except Exception as e:
        print(f"錯誤：讀取索引失敗: {e}")
        return

    df_output = pd.DataFrame(index=indices)
    raw_cols  = ['土地位置建物門牌', '緯度', '經度']

    try:
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"找不到原始資料: {raw_data_path}")
        df_raw   = pd.read_csv(raw_data_path)
        df_output = df_output.join(df_raw[raw_cols], how='left')
    except Exception as e:
        print(f"警告：合併原始資料錯誤 ({e})")

    quantile_cols = []
    target_quantiles = sorted([q for q in predictions_dict.keys() if q != 'mean'])

    if 'mean' in predictions_dict:
        if len(predictions_dict['mean']) == len(df_output):
            df_output['pred_mean'] = np.expm1(predictions_dict['mean'])

    for q in target_quantiles:
        preds = predictions_dict[q]
        if len(preds) != len(df_output): continue
        col_name = f'pred_{q}'
        df_output[col_name] = np.expm1(preds)
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
    print(" Deep Neural Network (DNN) — Quantile Regression")
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

    print(f"\n訓練集: {X_train.shape} | 測試集: {X_test.shape} | 完整: {X_full.shape}")

    quantiles     = config['data']['quantiles']
    all_quantiles = ['mean'] + quantiles
    input_dim     = X_train.shape[1]

    predictions_test = {}
    predictions_full = {}

    for q in all_quantiles:
        print(f"\n{'─'*60}")
        print(f" 訓練 Quantile = {q}")
        print(f"{'─'*60}")

        print("  Optuna 搜索中 ...")
        best = optimize_dnn(X_train, y_train, q, config)
        print(f"  最佳參數: {best}")

        units    = UNIT_CONFIGS[best['unit_config_idx']]
        drop_base = best['drop_base']
        dropouts = (drop_base, drop_base, max(0.05, drop_base - 0.1), max(0.05, drop_base - 0.1))

        print("  訓練最終模型 ...")
        np.random.seed(seed)
        tf.random.set_seed(seed)

        final_model = build_dnn(input_dim, q, best['lr'], units, dropouts, best['use_bn'])

        final_model.fit(
            X_train, y_train, epochs=100, batch_size=best['batch_size'], validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0)
            ], verbose=0
        )

        predictions_test[q] = final_model.predict(X_test, verbose=0).flatten()
        predictions_full[q] = final_model.predict(X_full, verbose=0).flatten()

        model_path = os.path.join(config['paths']['model_dir'], 'stage1', f'dnn_q{q}.keras')
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

    save_detailed_predictions(predictions_test, os.path.join(processed_dir, 'test_index.csv'),
                              raw_data_path, os.path.join(result_dir, 'dnn_test_predictions.csv'))
    save_detailed_predictions(predictions_full, os.path.join(processed_dir, 'full_index.csv'),
                              raw_data_path, os.path.join(result_dir, 'dnn_full_predictions.csv'))

    print(f"\n{'='*60}")
    print(" 測試集評估 (還原至原始房價單位)")
    print(f"{'='*60}")

    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(v) for k, v in predictions_test.items()}

    results_df = evaluate_predictions(y_test_real, predictions_test_real, quantiles, "DNN")
    
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
    results_df.to_csv(os.path.join(eval_dir, 'dnn_evaluation.csv'), index=False)

    print("\n" + "=" * 60)
    print(" DNN 訓練完成")
    print(f" 總執行時間: {time_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()