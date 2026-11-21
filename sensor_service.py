# sensor_service.py
import csv
import glob
import json
import os
import random
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HOST, PORT = "127.0.0.1", 8000
BASE = os.path.dirname(os.path.abspath(__file__))

FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "TEY"]

# Границы нормы
RANGES = {
    "AT":   (-6.23, 37.10),   "AP":  (985.85, 1036.56), "AH": (24.08, 100.20),
    "AFDP": (2.09, 7.61),     "GTEP": (17.70, 40.72),   "TIT": (1000.85, 1100.89),
    "TAT":  (511.04, 550.61), "CDP": (9.85, 15.16),     "TEY": (100.02, 179.50),
}
UNITS = {"AT": "°C", "AP": "мбар", "AH": "%", "AFDP": "мбар", "GTEP": "мбар",
         "TIT": "°C", "TAT": "°C", "CDP": "мбар", "TEY": "МВт·ч"}
NOISE = 0.01

def load_rows():
    """Читает реальные строки из gt_*.csv (ищет рядом и в ./data)."""
    paths = sorted(glob.glob(os.path.join(BASE, "gt_*.csv")) +
                   glob.glob(os.path.join(BASE, "data", "gt_*.csv")))
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({k: float(r[k]) for k in FEATURES})
                except (ValueError, KeyError):
                    continue
    return rows

ROWS = load_rows()
_pos = random.randrange(len(ROWS)) if ROWS else 0
_lock = threading.Lock()
_state = {k: sum(RANGES[k]) / 2 for k in FEATURES}

def clamp(name, v):
    lo, hi = RANGES[name]
    return min(max(v, lo), hi)

def next_reading():
    """Одно показание, гарантированно в пределах нормы."""
    global _pos
    with _lock:
        if ROWS:
            base = ROWS[_pos]
            _pos = (_pos + 1) % len(ROWS)
        else:
            for k in FEATURES:
                lo, hi = RANGES[k]
                _state[k] = clamp(k, _state[k] + random.gauss(0, (hi - lo) * 0.03))
            base = dict(_state)
    out = {}
    for k in FEATURES:
        lo, hi = RANGES[k]
        out[k] = round(clamp(k, base[k] + random.gauss(0, (hi - lo) * NOISE)), 3)
    return out

# --- Необязательный прогноз NOx ---
_model = None
try:
    import joblib
    import pandas as pd
    _bundle = joblib.load(os.path.join(BASE, "nox_model.pkl"))
    _model = _bundle["model"]
except Exception:
    pass

def add_prediction(readings):
    if _model is None:
        return None
    row = pd.DataFrame([{f: r[f] for f in FEATURES} for r in readings])
    return [round(float(x), 2) for x in _model.predict(row)]

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query) # Исправленная строка

        if url.path == "/health":
            return self._send(200, {"status": "ok", "source": "csv" if ROWS else "random-walk",
                                    "rows_loaded": len(ROWS), "model_loaded": _model is not None})

        if url.path == "/ranges":
            return self._send(200, {k: {"min": RANGES[k][0], "max": RANGES[k][1], "unit": UNITS[k]}
                                    for k in FEATURES})

        if url.path == "/readings":
            try:
                n = int(q.get("n", ["1"])[0])
            except ValueError:
                return self._send(400, {"error": "n должно быть целым числом"})
            if not 1 <= n <= 1000:
                return self._send(400, {"error": "n должно быть от 1 до 1000"})

            readings = [next_reading() for _ in range(n)]
            want_pred = q.get("predict", ["0"])[0] in ("1", "true", "yes")
            preds = add_prediction(readings) if want_pred else None
            if want_pred and preds is None:
                return self._send(503, {"error": "модель недоступна"})

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            items = []
            for i, r in enumerate(readings):
                item = {"timestamp": ts, "readings": r, "status": "НОРМА"}
                if preds:
                    item["nox_predicted"] = preds[i]
                items.append(item)
            return self._send(200, items[0] if n == 1 else {"count": n, "items": items})

        self._send(404, {"error": "не найдено", "endpoints": ["/readings", "/ranges", "/health"]})

    def log_message(self, fmt, *args):
        pass

if __name__ == "__main__":
    src = f"{len(ROWS)} строк из CSV" if ROWS else "CSV не найдены — режим случайного блуждания"
    print(f"Сервис датчиков: http://{HOST}:{PORT}  ({src})")
    print("Пример: http://127.0.0.1:8000/readings?n=5&predict=1")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
