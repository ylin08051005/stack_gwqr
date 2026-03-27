import sys
import os
import pandas as pd
import numpy as np

# 確保可以讀取到上一層的 utils.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import evaluate_predictions

# 1. 讀取剛剛成功存下來的 30% Test 預測檔
df_test = pd.read_csv('results/predictions/113_nn_L4_test_predictions.csv')
y_test_real = df_test['actual_price_million'].values

# =========================================================
# ★ 修正點：分開定義讀檔用(含mean)與算分數用(純數字)的陣列
# =========================================================
all_quantiles = ['mean', 0.1, 0.25, 0.5, 0.75, 0.9]
eval_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9] # 丟給評估函數的只能是純數字

# 2. 準備 pred_dict
test_preds_real = {q: df_test[f'pred_{q}'].values for q in all_quantiles}

# 3. 呼叫你的評估函數算成績 (★ 這裡傳入 eval_quantiles)
test_report_df = evaluate_predictions(y_test_real, test_preds_real, eval_quantiles, "NN_Test_30_Ensemble")

# 4. 印出來看，順便存檔！
print("\n" + "=" * 60)
print(" 神經網路 (NN) 30% 獨立 Test 盲測集成表現 (補算成績單):")
print("=" * 60)
print(test_report_df.to_string(index=False))

test_report_df.to_csv('results/evaluation/113_nn_L4_test_evaluation.csv', index=False)
print("\n✅ 補救存檔成功！")