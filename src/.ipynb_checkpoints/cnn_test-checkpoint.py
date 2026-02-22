"""
05_stage1_cnn.py
1-D Convolutional Neural Network (CNN) Quantile Regression Model
Features: Parameter optimization, training, output detailed prediction results (including rearrangement)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold

sys.path.append('..')
from utils import load_config, save_model, evaluate_predictions

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# TF uses all cores for calculation, Optuna limited to single thread
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)


# ============================================================
# Loss Function
# ============================================================
def quantile_loss_fn(tau):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1.0) * e))
    return loss


# ============================================================
# Model Construction (1D CNN)
# ============================================================
def build_cnn(input_dim, quantile, lr,
              conv_filters=(64, 128, 64),
              conv_dropouts=(0.2, 0.2, 0.2),
              dense_units=(128, 64),
              dense_dropouts=(0.3, 0.2),
              kernel_size=3):
    
    inp = layers.Input(shape=(input_dim,), name='input')

    # Reshape for Conv1D: (batch, features) -> (batch, features, 1)
    x = layers.Reshape((input_dim, 1), name='reshape')(inp)

    # Conv Blocks
    for i, (filt, drop) in enumerate(zip(conv_filters, conv_dropouts)):
        x = layers.Conv1D(filt, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)

    # Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dense Blocks
    for i, (units, drop) in enumerate(zip(dense_units, dense_dropouts)):
        x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)

    out = layers.Dense(1, name='output')(x)
    model = keras.Model(inputs=inp, outputs=out)

    loss = 'mse' if quantile == 'mean' else quantile_loss_fn(float(quantile))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
    return model


# ============================================================
# Optuna Optimization
# ============================================================
CONV_CONFIGS = [
    (64,  128, 64),
    (32,  64,  32),
    (128, 256, 128),
    (64,  64,  64),
]

DENSE_CONFIGS = [
    (128, 64),
    (64,  32),
    (256, 128),
    (128, 128),
]

def optimize_cnn(X_train, y_train, quantile, config):
    seed      = config['project']['random_seed']
    n_trials  = config['hyperparameter_tuning']['n_trials']
    timeout   = config['hyperparameter_tuning']['timeout']
    input_dim = X_train.shape[1]

    def objective(trial):
        lr            = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        conv_cfg_idx  = trial.suggest_categorical('conv_cfg_idx', list(range(len(CONV_CONFIGS))))
        dense_cfg_idx = trial.suggest_categorical('dense_cfg_idx', list(range(len(DENSE_CONFIGS))))
        conv_drop     = trial.suggest_float('conv_drop', 0.1, 0.4)
        dense_drop_1  = trial.suggest_float('dense_drop_1', 0.2, 0.5)
        dense_drop_2  = trial.suggest_float('dense_drop_2', 0.1, 0.4)
        batch_size    = trial.suggest_categorical('batch_size', [32, 64, 128])

        conv_filters   = CONV_CONFIGS[conv_cfg_idx]
        dense_units    = DENSE_CONFIGS[dense_cfg_idx]
        conv_dropouts  = tuple([conv_drop] * len(conv_filters))
        dense_dropouts = (dense_drop_1, dense_drop_2)

        kf = KFold(n_splits=3, shuffle=True, random_state=seed)
        fold_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            tf.random.set_seed(seed)
            model = build_cnn(input_dim, quantile, lr, conv_filters, conv_dropouts, dense_units, dense_dropouts)

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
    # n_jobs=1 ensures stability with TF
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1, show_progress_bar=True)
    return study.best_params


# ============================================================
# Detailed Prediction Report (with Rearrangement)
# ============================================================
def save_detailed_predictions(predictions_dict, index_file, raw_data_path, output_path):
    print(f"  整合詳細資料至: {output_path}")

    try:
        indices = pd.read_csv(index_file).iloc[:, 0].astype(int).values
    except Exception as e:
        print(f"  錯誤：讀取索引失敗: {e}")
        return

    df_output = pd.DataFrame(index=indices)
    raw_cols  = ['土地位置建物門牌', '緯度', '經度']

    try:
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"找不到原始資料: {raw_data_path}")
        df_raw   = pd.read_csv(raw_data_path)
        df_output = df_output.join(df_raw[raw_cols], how='left')
    except Exception as e:
        print(f"  警告：合併原始資料錯誤 ({e})")

    quantile_cols = []
    target_quantiles = sorted([q for q in predictions_dict.keys() if q != 'mean'])

    # Handle Mean
    if 'mean' in predictions_dict:
        if len(predictions_dict['mean']) == len(df_output):
            df_output['pred_mean'] = np.expm1(predictions_dict['mean'])

    # Handle Quantiles
    for q in target_quantiles:
        preds = predictions_dict[q]
        if len(preds) != len(df_output): continue
        col_name = f'pred_{q}'
        df_output[col_name] = np.expm1(preds)
        quantile_cols.append(col_name)

    # Rearrangement (Sorting)
    if len(quantile_cols) > 1:
        print("  執行分位數交叉修正 (Rearrangement)...")
        q_values = df_output[quantile_cols].values
        q_values.sort(axis=1)
        df_output[quantile_cols] = q_values

    # Reorder columns
    cols = list(df_output.columns)
    priority = ['土地位置建物門牌', '緯度', '經度']
    new_order = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
    df_output = df_output[new_order]

    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("  儲存完成。")


# ============================================================
# Main
# ============================================================
def main():
    config = load_config('../config.yaml')
    seed   = config['project']['random_seed']

    print("=" * 60)
    print(" 1D CNN — Quantile Regression")
    print("=" * 60)

    processed_dir = os.path.join(config['paths']['data_dir'], 'processed')
    raw_data_path = os.path.join(config['paths']['data_dir'], 'raw', 'realprice_with_aqi_idw1122.csv')
    if not os.path.exists(raw_data_path):
        raw_data_path = "/Users/ylin/Documents/stack_gwqr/src/data/raw/realprice_with_aqi_idw1122.csv"
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
        best = optimize_cnn(X_train, y_train, q, config)
        print(f"  最佳參數: {best}")

        conv_filters   = CONV_CONFIGS[best['conv_cfg_idx']]
        dense_units    = DENSE_CONFIGS[best['dense_cfg_idx']]
        conv_dropouts  = tuple([best['conv_drop']] * len(conv_filters))
        dense_dropouts = (best['dense_drop_1'], best['dense_drop_2'])

        print("  訓練最終模型 ...")
        np.random.seed(seed)
        tf.random.set_seed(seed)

        final_model = build_cnn(input_dim, q, best['lr'], conv_filters, conv_dropouts, dense_units, dense_dropouts)

        final_model.fit(
            X_train, y_train, epochs=100, batch_size=best['batch_size'], validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0)
            ], verbose=0
        )

        predictions_test[q] = final_model.predict(X_test, verbose=0).flatten()
        predictions_full[q] = final_model.predict(X_full, verbose=0).flatten()

        model_path = os.path.join(config['paths']['model_dir'], 'stage1', f'cnn_q{q}.keras')
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
                              raw_data_path, os.path.join(result_dir, 'cnn_test_predictions.csv'))
    save_detailed_predictions(predictions_full, os.path.join(processed_dir, 'full_index.csv'),
                              raw_data_path, os.path.join(result_dir, 'cnn_full_predictions.csv'))

    print(f"\n{'='*60}")
    print(" 測試集評估 (還原至原始房價單位)")
    print(f"{'='*60}")

    y_test_real = np.expm1(y_test)
    predictions_test_real = {k: np.expm1(v) for k, v in predictions_test.items()}

    results_df = evaluate_predictions(y_test_real, predictions_test_real, quantiles, "CNN")
    print(results_df.to_string(index=False))

    eval_dir = os.path.join(config['paths']['result_dir'], 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    results_df.to_csv(os.path.join(eval_dir, 'cnn_evaluation.csv'), index=False)

    print("\n" + "=" * 60)
    print(" CNN 訓練完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()