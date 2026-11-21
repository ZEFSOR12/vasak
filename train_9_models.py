# train_9_models.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Загрузка данных
print("Загрузка данных...")
dfs = []
for year in range(2011, 2016):
    path = f"data/gt_{year}.csv"
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))

df = pd.concat(dfs, ignore_index=True).dropna()
FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "NOX"]

# --- ЧАСТЬ 1: РАСЧЕТ НАУЧНЫХ НОРМ (Mean ± 3 Sigma) ---
print("\n=== РАСЧЕТ НАУЧНЫХ НОРМ (3 СИГМА) ===")
norms_dict = {}
for col in FEATURES:
    mean = df[col].mean()
    std = df[col].std()
    lower = mean - 3 * std
    upper = mean + 3 * std
    norms_dict[col] = (round(lower, 2), round(upper, 2))
    print(f"{col}: [{lower:.2f}, {upper:.2f}]")

print("\nСкопируй этот словарь в app.py вместо старого RANGES:")
print(norms_dict)

# --- ЧАСТЬ 2: ОБУЧЕНИЕ 9 МОДЕЛЕЙ (Virtual Sensors) ---
print("\n=== ОБУЧЕНИЕ 9 МОДЕЛЕЙ ДЛЯ КАЖДОГО ДАТЧИКА ===")
models_bundle = {}

for target_metric in FEATURES:
    # Все остальные датчики становятся признаками
    other_features = [f for f in FEATURES if f != target_metric]
    
    X = df[other_features]
    y = df[target_metric]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    models_bundle[target_metric] = pipeline
    print(f"Модель для {target_metric:4s}: R² = {r2:.3f}, Ошибка (RMSE) = {rmse:.2f}")

# Сохраняем все 9 моделей в один файл
joblib.dump(models_bundle, "sensors_models.pkl")
print("\nВсе 9 моделей сохранены в sensors_models.pkl")
