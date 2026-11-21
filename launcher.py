# launcher.py
import webview
import subprocess
import time
import sys
import os

# 1. Запускаем Streamlit в фоне
print("Запуск сервера приложения...")
# Запускаем скрытно, чтобы не открывалось черное окно консоли
streamlit_process = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
)

# 2. Ждем пару секунд, чтобы сервер успел подняться
time.sleep(3)

# 3. Открываем нативное окно программы
print("Открытие окна...")
window = webview.create_window(
    title="NOx Predictor — Газотурбинная установка", 
    url="http://localhost:8501", 
    width=1400, 
    height=900,
    background_color="#f0f2f6"
)

# 4. Запускаем окно
webview.start()

# 5. Когда окно закрывается, принудительно выключаем Streamlit
streamlit_process.terminate()
print("Программа закрыта.")
