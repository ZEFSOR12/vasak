# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import os

FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "TEY"]
TARGET = "NOX"

print("Загрузка данных...")
dfs = []
for year in range(2011, 2016):
    path = f"data/gt_{year}.csv"
    d = pd.read_csv(path)
    d["year"] = year
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

# На всякий случай удаляем пустые строки (NaN)
df = df.dropna(subset=FEATURES + [TARGET])
print(f"Всего записей после очистки: {len(df)}")

# --- Вариант 1: Хронологический сплит (по протоколу UCI) ---
train = df[df["year"] <= 2013]
test = df[df["year"] >= 2014]
X_train_chrono, y_train_chrono = train[FEATURES], train[TARGET]
X_test_chrono, y_test_chrono = test[FEATURES], test[TARGET]

print("\nОбучение на хронологическом сплите (2011-2013 -> 2014-2015)...")
model_chrono = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model_chrono.fit(X_train_chrono, y_train_chrono)
y_pred_chrono = model_chrono.predict(X_test_chrono)
r2_chrono = r2_score(y_test_chrono, y_pred_chrono)
print(f"R² на хронологическом сплите: {r2_chrono:.4f}")

# --- Вариант 2: Случайный сплит (для красивых графиков и UI) ---
print("\nДля интерфейса и отчетов обучаем модель на случайном сплите...")
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=150, max_depth=15, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nИтоговые метрики (случайный сплит):")
print(f"MAE  = {mae:.2f} mg/m³")
print(f"RMSE = {rmse:.2f} mg/m³")
print(f"R²   = {r2:.4f}")

importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nВажность признаков:")
print(importance.to_string())

# --- Графики ---
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10, c="#2a5298")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", lw=2, label="Идеальный прогноз")
plt.xlabel("Фактический NOx, мг/м³")
plt.ylabel("Прогноз NOx, мг/м³")
plt.title("Сравнение расчётных и фактических значений NOx")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("plots/scatter.png", dpi=150); plt.close()

plt.figure(figsize=(12, 4))
n = 200
plt.plot(y_test.values[:n], label="Факт", lw=1.6, color="#1e3c72")
plt.plot(y_pred[:n], label="Прогноз", lw=1.6, color="#ff7b54", alpha=0.9)
plt.xlabel("Запись"); plt.ylabel("NOx, мг/м³")
plt.title("Динамика NOx: факт vs прогноз (первые 200 записей)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("plots/line.png", dpi=150); plt.close()

# --- Сохранение ---
joblib.dump({
    "model": model,
    "features": FEATURES,
    "importance": importance.to_dict(),
    "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
    "y_test": y_test.values.tolist()[:500],
    "y_pred": y_pred[:500].tolist(),
}, "nox_model.pkl")
print("\nГотово. Модель: nox_model.pkl, графики в plots/")
