"""
CNN Quantile Regression Model
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
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

from utils import (load_config, evaluate_predictions)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def quantile_loss(quantile):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(float(quantile) * error, (float(quantile) - 1.0) * error))
    return loss

def build_cnn_model(input_dim, quantile, learning_rate, y_train_mean):

    output_bias = keras.initializers.Constant(value=y_train_mean)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),

        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(1, bias_initializer=output_bias)
    ])
    
    loss_fn = 'mse' if quantile == 'mean' else quantile_loss(quantile)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model

def optimize_cnn_hyperparameters(X_train, y_train, X_val, y_val, quantile, config, seed, y_train_mean):
    input_dim = X_train.shape[1]

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        tf.random.set_seed(seed)
        model = build_cnn_model(input_dim, quantile, learning_rate, y_train_mean)
        
        model.fit(
            X_train, y_train,
            epochs=60,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
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
            error = y_val - y_pred
            score = np.mean(np.where(error >= 0, float(quantile) * error, (float(quantile) - 1.0) * error))
        
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

def save_detailed_predictions(predictions_dict, y_true_real, indices, raw_data_path, output_path):
    print(f"儲存報表至: {output_path}")
    try:
        df_output = pd.DataFrame(index=indices)
        df_raw = pd.read_csv(raw_data_path)
        
        raw_cols = ['土地位置建物門牌', '緯度', '經度']
        df_output = df_output.join(df_raw[raw_cols], how='left')

        df_output['actual_price_million'] = y_true_real

        quantiles = sorted([q for q in predictions_dict.keys() if q != 'mean'])

        if 'mean' in predictions_dict:
            df_output['pred_mean'] = np.exp(np.clip(predictions_dict['mean'], -20.0, 20.0))
        for q in quantiles:
            df_output[f'pred_{q}'] = np.exp(np.clip(predictions_dict[q], -20.0, 20.0))

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
        print("儲存完成")
    except Exception as e:
        print(f"儲存失敗: {e}")

def main():
    start_time = time.time()
    config = load_config(os.path.join(parent_dir, 'config.yaml'))
    seed = config['project']['random_seed']
    EXP_LEVEL = "L3"
    target_year = '113' 
    
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
    
    X_train_70 = X_all_raw[~is_test_set].drop(columns=['fold']).values.astype(np.float32)
    y_train_70 = y_all[~is_test_set].astype(np.float32)
    idx_train_70 = idx_all[~is_test_set]
    train_fold_labels = X_all_raw[~is_test_set]['fold'].values
    
    X_test_30 = X_all_raw[is_test_set].drop(columns=['fold']).values.astype(np.float32)
    y_test_30 = y_all[is_test_set].astype(np.float32)
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
        print(f"執行Train Fold {fold_idx + 1} / 5")
        print("="*40)
        
        val_mask = (train_fold_labels == fold_idx)
        tr_mask = ~val_mask
        
        X_tr_full = X_train_70[tr_mask]
        y_tr_full = y_train_70[tr_mask]
        X_val = X_train_70[val_mask]
        y_val = y_train_70[val_mask]
        idx_val = idx_train_70[val_mask]

        input_dim = X_tr_full.shape[1]
        y_train_mean = np.mean(y_tr_full)

        X_tr_opt, X_va_opt, y_tr_opt, y_va_opt = train_test_split(
            X_tr_full, y_tr_full, test_size=0.1, random_state=seed
        )
        
        fold_predictions = {}

        for q in all_quantiles:
            print(f"\n > [Quantile {q}] 尋找最佳參數")
            best = optimize_cnn_hyperparameters(X_tr_opt, y_tr_opt, X_va_opt, y_va_opt, q, config, seed, y_train_mean)
            
            tf.random.set_seed(seed)
            eval_model = build_cnn_model(input_dim, q, best['learning_rate'], y_train_mean)
            
            eval_model.fit(
                X_tr_full, y_tr_full, 
                epochs=100, 
                batch_size=best['batch_size'], 
                validation_split=0.1, 
                callbacks=[
                    keras.callbacks.TerminateOnNaN(),
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=0)
                ], 
                verbose=0
            )

            fold_pred = eval_model.predict(X_val, verbose=0).flatten()
            test_pred = eval_model.predict(X_test_30, verbose=0).flatten()
            
            fold_predictions[q] = fold_pred
            predictions_oof[q][val_mask] = fold_pred
            test_preds_sum[q] += test_pred

            keras.backend.clear_session()

        y_val_real = np.exp(y_val)
        fold_preds_real = {k: np.exp(np.clip(v, -20.0, 20.0)) for k, v in fold_predictions.items()}
        
        fold_eval_df = evaluate_predictions(y_val_real, fold_preds_real, config['data']['quantiles'], f"CNN_Fold_{fold_idx+1}")
        all_fold_evaluations.append(fold_eval_df)
        
        save_detailed_predictions(
            fold_predictions, y_val_real, idx_val, raw_data_path, 
            os.path.join(result_dir, f'{target_year}_cnn_{EXP_LEVEL}_fold_{fold_idx+1}_predictions.csv')
        )

    print("\n" + "=" * 60)
    print("產出 Train(OOF)、Test 報表")
    print("=" * 60)
    
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    exec_time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    y_train_real_million = np.exp(y_train_70)
    oof_preds_real = {k: np.exp(np.clip(v, -20.0, 20.0)) for k, v in predictions_oof.items()}
    
    save_detailed_predictions(
        predictions_oof, y_train_real_million, idx_train_70, raw_data_path, 
        os.path.join(result_dir, f'{target_year}_cnn_{EXP_LEVEL}_oof_predictions.csv')
    )
    
    oof_eval = evaluate_predictions(y_train_real_million, oof_preds_real, config['data']['quantiles'], "CNN_Train_70_OOF")
    oof_eval['Execution_Time'] = exec_time_str
    train_report_df = pd.concat(all_fold_evaluations + [oof_eval], ignore_index=True)
    train_report_df.to_csv(os.path.join(eval_dir, f'{target_year}_cnn_{EXP_LEVEL}_5fold_evaluation.csv'), index=False)

    y_test_real_million = np.exp(y_test_30)
    test_preds_avg = {q: (preds / 5.0) for q, preds in test_preds_sum.items()}
    test_preds_real = {k: np.exp(np.clip(v, -20.0, 20.0)) for k, v in test_preds_avg.items()}
    
    save_detailed_predictions(
        test_preds_avg, y_test_real_million, idx_test_30, raw_data_path, 
        os.path.join(result_dir, f'{target_year}_cnn_{EXP_LEVEL}_test_predictions.csv')
    )

    test_report_df = evaluate_predictions(y_test_real_million, test_preds_real, config['data']['quantiles'], "CNN_Test_30_Ensemble")
    test_report_df['Execution_Time'] = exec_time_str
    test_report_df.to_csv(os.path.join(eval_dir, f'{target_year}_cnn_{EXP_LEVEL}_test_evaluation.csv'), index=False)

    print("\n" + "=" * 80)
    print(f" CNN 1D - {EXP_LEVEL}")
    print("=" * 80)
    print("[1] 70% Train 5-Fold OOF:")
    print(train_report_df.to_string(index=False))
    print("-" * 80)
    print("[2] 30% Test 盲測:")
    print(test_report_df.to_string(index=False))

if __name__ == "__main__":
    main()