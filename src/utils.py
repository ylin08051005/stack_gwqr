"""
Utility functions for house price prediction project
"""
import numpy as np
import pandas as pd
import yaml
import os
import pickle
from typing import List, Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def load_config(config_path: str = './config.yaml') -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: dict):
    directories = [
        config['paths']['data_dir'],
        os.path.join(config['paths']['data_dir'], 'raw'),
        os.path.join(config['paths']['data_dir'], 'processed'),
        config['paths']['model_dir'],
        os.path.join(config['paths']['model_dir'], 'stage1'),
        os.path.join(config['paths']['model_dir'], 'stage2'),
        config['paths']['result_dir'],
        os.path.join(config['paths']['result_dir'], 'predictions'),
        os.path.join(config['paths']['result_dir'], 'evaluation'),
        os.path.join(config['paths']['result_dir'], 'figures'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def check_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    計算 Check Loss
    
    Parameters:
    y_true : 實際值
    y_pred : 預測值
    quantile

    Returns:
    check_loss : check loss值
    """
    error = y_true - y_pred
    loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)
    return np.mean(loss)


def total_check_loss(y_true: np.ndarray, predictions_dict: dict, quantiles: List[float]) -> float:
    """
    計算check loss total
    
    Parameters:
    y_true : 實際值
    predictions_dict : key為quantile數，value為預測值
    quantiles列表
    
    Returns:
    total_loss
    """
    total_loss = 0
    for q in quantiles:
        if q in predictions_dict:
            total_loss += check_loss(y_true, predictions_dict[q], q)
    return total_loss


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    計算 sMAPE 
    
    Parameters:
    y_true : 實際值
    y_pred : 預測值
    
    Returns:
    sMAPE值
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape


def calculate_pseudo_r2(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    計算 Pseudo R² for quantile regression
    
    Parameters:
    y_true : 實際值
    y_pred : 預測值
    quantile
    
    Returns:
    Pseudo R²值
    """

    model_loss = check_loss(y_true, y_pred, quantile)

    null_pred = np.percentile(y_true, quantile * 100)
    null_loss = check_loss(y_true, np.full_like(y_true, null_pred), quantile)

    pseudo_r2 = 1 - (model_loss / null_loss)
    return pseudo_r2


def evaluate_predictions(y_true: np.ndarray, 
                         predictions_dict: dict, 
                         quantiles: List[float],
                         model_name: str = "Model") -> pd.DataFrame:
    """
    評估預測結果
    
    Parameters:
    y_true : 實際值
    predictions_dict : key是quantile or mean，value是預測值
    quantiles
    model_name
    
    Returns:
    results_df : 評估結果的DataFrame
    """
    results = []

    for q in quantiles:
        if q in predictions_dict:
            y_pred = predictions_dict[q]
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            check_loss_val = check_loss(y_true, y_pred, q)
            pseudo_r2 = calculate_pseudo_r2(y_true, y_pred, q)
            smape = calculate_smape(y_true, y_pred)
            
            results.append({
                'Model': model_name,
                'Quantile': q,
                'RMSE': rmse,
                'Check_Loss': check_loss_val,
                'Pseudo_R2': pseudo_r2,
                'sMAPE': smape
            })
    #for mean ver.
    if 'mean' in predictions_dict:
        y_pred_mean = predictions_dict['mean']
        rmse_mean = np.sqrt(mean_squared_error(y_true, y_pred_mean))
        smape_mean = calculate_smape(y_true, y_pred_mean)
        
        results.append({
            'Model': model_name,
            'Quantile': 'mean',
            'RMSE': rmse_mean,
            'Check_Loss': np.nan,
            'Pseudo_R2': np.nan,
            'sMAPE': smape_mean
        })
    
    #for quantile ver.
    total_loss = total_check_loss(y_true, predictions_dict, quantiles)
    results.append({
        'Model': model_name,
        'Quantile': 'Total',
        'RMSE': np.nan,
        'Check_Loss': total_loss,
        'Pseudo_R2': np.nan,
        'sMAPE': np.nan
    })
    
    results_df = pd.DataFrame(results)
    return results_df


def save_model(model, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已存到{filepath}")


def load_model(filepath: str):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"載入：{filepath}")
    return model


def save_predictions(predictions_dict: dict, filepath: str):
    df = pd.DataFrame(predictions_dict)
    df.to_csv(filepath, index=False)
    print(f"預測結果已存到: {filepath}")


def load_predictions(filepath: str) -> dict:
    df = pd.read_csv(filepath)
    predictions_dict = df.to_dict('list')
    predictions_dict = {k: np.array(v) for k, v in predictions_dict.items()}
    print(f"預測結果載入到: {filepath}")
    return predictions_dict


class QuantileObjective:
    def __init__(self, quantile: float):
        self.quantile = quantile
    
    def __call__(self, y_true, y_pred):
        error = y_true - y_pred
        loss = np.where(error >= 0, self.quantile * error, (self.quantile - 1) * error)
        return loss
    
    def gradient(self, y_true, y_pred):
        return np.where(y_true >= y_pred, -self.quantile, 1 - self.quantile)
    
    def hessian(self, y_true, y_pred):
        return np.ones_like(y_pred)


def xgb_quantile_obj(quantile: float):
    def objective(y_true, y_pred):
        error = y_true - y_pred
        grad = np.where(error > 0, -quantile, 1 - quantile)
        hess = np.ones_like(y_pred)
        return grad, hess
    return objective


def lgb_quantile_obj(quantile: float):
    def objective(y_true, y_pred):
        error = y_true - y_pred
        grad = np.where(error > 0, -quantile, 1 - quantile)
        hess = np.ones_like(y_pred)
        return grad, hess
    return objective