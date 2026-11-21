# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sqlite3
import datetime
import os
import hashlib
import requests

st.set_page_config(page_title="VASAK — NOx Predictor", page_icon="🏭", layout="wide")

DB_FILE = "turbine_data.db"

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С ПАРОЛЯМИ ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# --- ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ В 3НФ ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS readings (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, timestamp TEXT, threshold REAL, FOREIGN KEY(user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_data (id INTEGER PRIMARY KEY AUTOINCREMENT, reading_id INTEGER, AT REAL, AP REAL, AH REAL, AFDP REAL, GTEP REAL, TIT REAL, TAT REAL, CDP REAL, TEY REAL, nox_predicted REAL, FOREIGN KEY(reading_id) REFERENCES readings(id))''')
    c.execute('SELECT COUNT(*) FROM users')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?,?)', ('admin', make_hashes('admin')))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT id, username, password_hash FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    if data and check_hashes(password, data[2]):
        return data[0], data[1]
    return None, None

def save_to_db_3nf(user_id, readings, nox, threshold):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO readings (user_id, timestamp, threshold) VALUES (?, ?, ?)', (user_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), threshold))
        reading_id = c.lastrowid
        c.execute('''INSERT INTO sensor_data (reading_id, AT, AP, AH, AFDP, GTEP, TIT, TAT, CDP, TEY, nox_predicted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                  (reading_id, readings['AT'], readings['AP'], readings['AH'], readings['AFDP'], readings['GTEP'], readings['TIT'], readings['TAT'], readings['CDP'], readings['TEY'], round(nox, 2)))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Ошибка БД: {e}")
        return False
    finally:
        conn.close()

def load_history_3nf():
    conn = sqlite3.connect(DB_FILE)
    query = '''
        SELECT r.id, u.username, r.timestamp, r.threshold, s.AT, s.AP, s.AH, s.AFDP, s.GTEP, s.TIT, s.TAT, s.CDP, s.TEY, s.nox_predicted
        FROM readings r JOIN users u ON r.user_id = u.id JOIN sensor_data s ON r.id = s.reading_id
        ORDER BY r.id DESC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df["Статус"] = np.where(df["nox_predicted"] > df["threshold"], "ПРЕВЫШЕНИЕ", "НОРМА")
    return df

init_db()

# --- РАСЧЕТ ДИАПАЗОНОВ ПО ПРАВИЛУ 3 СИГМА ---
@st.cache_data
def calculate_3_sigma_ranges():
    # Базовые диапазоны (на случай, если папки data нет)
    fallback = {
        'AT': (-4.63, 40.06), 'AP': (993.68, 1032.46), 'AH': (34.48, 121.25),
        'AFDP': (1.60, 6.25), 'GTEP': (12.98, 38.15), 'TIT': (1028.82, 1134.04),
        'TAT': (525.63, 566.69), 'TEY': (86.65, 180.36), 'CDP': (8.79, 15.33)
    }
    
    features = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "TEY"]
    data_path = "data/gt_2011.csv"
    
    if os.path.exists(data_path):
        try:
            dfs = [pd.read_csv(f"data/gt_{year}.csv") for year in range(2011, 2016)]
            df = pd.concat(dfs, ignore_index=True)
            ranges = {}
            for f in features:
                mean = df[f].mean()
                std = df[f].std()
                # Считаем 3 сигма: mu - 3*sigma и mu + 3*sigma
                lower = max(0, mean - 3 * std) # не допускаем отрицательных значений для датчиков
                upper = mean + 3 * std
                ranges[f] = (lower, upper)
            return ranges
        except Exception as e:
            print(f"Ошибка при чтении данных: {e}")
            return fallback
    else:
        return fallback

RANGES = calculate_3_sigma_ranges()

# --- СТИЛИ ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap');

#MainMenu {visibility: hidden;} 
header {visibility: hidden;} 
footer {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;} 
.stDeployButton {display: none;}
a.header-anchor {display: none !important;}
[data-testid="stHeaderActionElements"] {display: none !important;}

@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse-orange {
    0% { box-shadow: 0 0 0 0 rgba(255, 176, 32, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(255, 176, 32, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 176, 32, 0); }
}
@keyframes danger-flash {
    0% { background-color: #FF4444; }
    50% { background-color: #ff2222; box-shadow: 0 0 20px #ff2222; }
    100% { background-color: #FF4444; }
}

.main-header { background-color: #3AE39A; padding: 1.5rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(58, 227, 154, 0.3); animation: fadeInUp 0.5s ease-out; }
.main-header h1 { margin: 0; font-family: 'Montserrat', sans-serif; font-size: 2.5rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #ffffff !important; }
.main-header p { margin: 0.5rem 0 0; font-family: 'Montserrat', sans-serif; font-weight: 600; color: #ffffff !important; opacity: 0.9; font-size: 1.1rem; }

.scada-display { background-color: #182233; border: 2px solid #3AE39A; border-radius: 10px; padding: 8px; text-align: center; font-size: 1.2rem; font-weight: 700; color: #3AE39A; box-shadow: 0 0 10px rgba(58,227,154,0.2); margin-bottom: 5px; }

.status-badge { display: block; padding: 10px 15px; border-radius: 25px; font-size: 0.95rem; font-weight: 700; margin-top: 5px; margin-bottom: 20px; text-align: center; font-family: 'Montserrat', sans-serif; animation: fadeInUp 0.4s ease-out; transition: all 0.3s ease; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
.status-normal { background-color: #3AE39A; color: #ffffff; }
.status-warning { background-color: #FFB020; color: #ffffff; animation: pulse-orange 2s infinite; }
.status-critical { background-color: #FF4444; color: #ffffff; }

.tooltip-container { position: relative; display: block; cursor: help; }
.tooltip-text { visibility: hidden; width: 260px; background-color: #182233; color: #ffffff; text-align: center; border-radius: 10px; padding: 12px; position: absolute; z-index: 1000; bottom: 130%; left: 50%; margin-left: -130px; opacity: 0; transition: opacity 0.3s; border: 1px solid #FFB020; box-shadow: 0 4px 15px rgba(0,0,0,0.5); font-size: 0.85rem; font-weight: 500; line-height: 1.4; }
.tooltip-text::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -6px; border-width: 6px; border-style: solid; border-color: #FFB020 transparent transparent transparent; }
.tooltip-container:hover .tooltip-text { visibility: visible; opacity: 1; }
.tooltip-danger { border: 1px solid #FF4444 !important; }
.tooltip-danger::after { border-color: #FF4444 transparent transparent transparent !important; }

.card { padding: 1.2rem 1.5rem; border-radius: 20px; margin: 1rem 0; color: #ffffff !important; box-shadow: 0 6px 15px rgba(0,0,0,0.3); border: none; animation: fadeInUp 0.6s ease-out; }
.card b, .card span, .card p { color: #ffffff !important; }
.card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.4); }
.card-ok { background-color: #3AE39A; }
.card-warn { background-color: #FFB020; }
.card-danger { background-color: #FF4444; }

.pred-box { border-radius: 20px; padding: 2.5rem; text-align: center; color: #ffffff !important; margin-bottom: 1rem; box-shadow: 0 8px 25px rgba(0,0,0,0.4); animation: fadeInUp 0.8s ease-out; }
.pred-box.good { background-color: #3AE39A; }
.pred-box.pulse-danger { background-color: #FF4444; animation: danger-flash 1.5s infinite !important; }
.pred-box h2 { margin: 0; font-size: 3.5rem; font-weight: 800; color: #ffffff !important; font-family: 'Montserrat', sans-serif; }
.pred-box p { margin: 0.5rem 0 0; font-size: 1.2rem; font-weight: 600; color: #ffffff !important; }

.no-connection-box { background-color: #182233; border: 2px solid #FF4444; border-radius: 20px; padding: 5rem 2rem; text-align: center; color: #ffffff; margin-top: 2rem; box-shadow: 0 10px 30px rgba(255, 68, 68, 0.2); animation: fadeInUp 0.6s ease-out; }
.no-connection-box h2 { font-family: 'Montserrat', sans-serif; color: #FF4444; font-size: 2rem; margin-bottom: 1rem; }
.no-connection-box p { font-size: 1.2rem; opacity: 0.8; }

div[data-baseweb="input"] { background-color: #182233 !important; border: 1px solid #3AE39A !important; border-radius: 10px; }
div[data-baseweb="input"]:focus-within { box-shadow: 0 0 10px rgba(58,227,154,0.5); }
.stButton > button { background-color: #182233 !important; color: #ffffff !important; border: 1px solid #3AE39A !important; border-radius: 20px !important; font-weight: 600; transition: all 0.3s; }
.stButton > button:hover { background-color: #3AE39A !important; color: #0E1522 !important; }

.section-title { font-family: 'Montserrat', sans-serif; font-size: 1.5rem; font-weight: 700; color: #ffffff !important; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #182233; animation: fadeInUp 0.6s ease-out; }
</style>
""", unsafe_allow_html=True)

# --- ЛОГИКА АВТОРИЗАЦИИ ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

if not st.session_state.logged_in:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        lc1, lc2, lc3 = st.columns([1, 2.5, 1])
        with lc2:
            if os.path.exists("logo.png"):
                st.image("logo.png", use_container_width=True)
            else:
                st.markdown("<h1 style='text-align:center; color:#3AE39A;'>VASAK</h1>", unsafe_allow_html=True)
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        login_user_input = st.text_input("Логин", value="admin")
        login_pass_input = st.text_input("Пароль", type="password", value="admin")
        if st.button("Войти", use_container_width=True, type="primary"):
            uid, uname = login_user(login_user_input, login_pass_input)
            if uid:
                st.session_state.logged_in = True
                st.session_state.user_id = uid
                st.session_state.username = uname
                st.rerun()
            else:
                st.error("Неверный логин или пароль")
        st.info("Логин: **admin**, Пароль: **admin**")
    st.stop()

# --- ОСНОВНОЕ ПРИЛОЖЕНИЕ ---
st.sidebar.markdown(f"### Пользователь: **{st.session_state.username}**")
if st.sidebar.button("Выйти"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Источник данных")
data_source = st.sidebar.radio("Режим:", ["Ручной ввод", "Внешний сервис (HTTP)"])

if 'connection_status' not in st.session_state:
    st.session_state.connection_status = 'pending'

if data_source == "Внешний сервис (HTTP)":
    st.sidebar.success("🟢 Ожидание данных от контроллера")
    if st.sidebar.button("🔄 Запросить данные", use_container_width=True):
        try:
            response = requests.get("http://127.0.0.1:8000/readings", timeout=3)
            if response.status_code == 200:
                data = response.json()
                st.session_state.scada_data = data["readings"]
                st.session_state.connection_status = 'success'
                st.sidebar.success("✅ Данные успешно получены!")
                st.rerun()
            else:
                st.session_state.connection_status = 'error'
                st.sidebar.error("Ошибка получения данных с контроллера.")
        except requests.exceptions.ConnectionError:
            st.session_state.connection_status = 'error'
            st.sidebar.error("❌ Отсутствует связь с контроллером.")
        except Exception:
            st.session_state.connection_status = 'error'
            st.sidebar.error("❌ Ошибка при получении данных.")
else:
    st.session_state.connection_status = 'manual'

st.markdown('<div class="main-header"><h1>VASAK</h1><p>Контроль выбросов NOx</p></div>', unsafe_allow_html=True)

if data_source == "Внешний сервис (HTTP)":
    if st.session_state.connection_status == 'error':
        st.markdown("""
        <div class="no-connection-box">
            <h2>⚠️ В настоящий момент показатели датчиков недоступны</h2>
            <p>Отсутствует связь с контроллером оборудования.<br>
            Пожалуйста, проверьте сетевое подключение и нажмите «🔄 Запросить данные».</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    elif st.session_state.connection_status == 'pending':
        st.markdown("""
        <div class="no-connection-box">
            <h2>⏳ Ожидание данных...</h2>
            <p>Нажмите кнопку «🔄 Запросить данные» в левом меню, чтобы получить показания с контроллера.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# --- ЗАГРУЗКА 9 МОДЕЛЕЙ ---
@st.cache_resource
def load_models():
    return joblib.load("sensors_models.pkl")

try:
    models_dict = load_models()
except FileNotFoundError:
    st.error("Файл sensors_models.pkl не найден. Убедись, что он лежит в папке vasak.")
    st.stop()

UNITS = {"AT":"°C","AP":"мбар","AH":"%","AFDP":"мбар","GTEP":"мбар","TIT":"°C","TAT":"°C","CDP":"мбар","TEY":"МВт·ч"}
LABELS = {
    "AT":"Температура окр среды", "AP":"Атмосферное давление", "AH":"Влажность воздуха", "AFDP":"Перепад на фильтре",
    "GTEP":"Давление выхлопных газов", "TIT":"Температура турбины", "TAT":"Температура после турбины",
    "CDP":"Давление на выходе компрессора", "TEY":"Выработка энергии",
}

ACTIONS = {
    "AT": ("Высокая температура воздуха снижает плотность заряда и повышает NOx. Проверьте охлаждение входного воздуха и снизьте нагрузку.", "Низкая температура не критична — следите за стабильностью горения при пуске."),
    "AP": ("Проверьте калибровку датчика; на работу влияет слабо.", "Пониженное давление снижает массовый расход воздуха — контролируйте нагрузку."),
    "AH": ("Влажность снижает NOx, но влияет на стабильность пламени — следите за CO.", "Низкая влажность способствует росту NOx — усильте контроль."),
    "AFDP": ("Признак загрязнения фильтров — очистите или замените фильтроэлементы.", "Возможна утечка или неисправность датчика."),
    "GTEP": ("Проверьте противодавление котла-утилизатора и выхлопной тракт.", "Проверьте герметичность и исправность датчика."),
    "TIT": ("Главный драйвер NOx. Снизьте TIT (топливо/нагрузка), проверьте охлаждение лопаток.", "Снижение эффективности — проверьте подачу топлива."),
    "TAT": ("Признак перегрузки — проверьте разброс температур, снизьте нагрузку.", "Возможна недогрузка или неисправность датчика."),
    "CDP": ("Проверьте проточную часть и калибровку датчика.", "Ранний признак загрязнения компрессора — промойте."),
    "TEY": ("Проверьте калибровку и соответствие нагрузке.", "Признак деградации — проверьте компрессор и топливную систему."),
}

def get_metric_status(val, lo, hi):
    span = hi - lo
    crit_margin = span * 0.40 
    if val > hi + crit_margin: return "Критически высокий", "status-critical"
    elif val > hi: return "Высокий", "status-warning"
    elif val < lo - crit_margin: return "Критически низкий", "status-critical"
    elif val < lo: return "Низкий", "status-warning"
    else: return "Норма", "status-normal"

st.markdown('<div class="section-title">Показания датчиков (Норма: μ ± 3σ)</div>', unsafe_allow_html=True)

defaults = {"AT":17.7,"AP":1013.0,"AH":77.8,"AFDP":3.9,"GTEP":25.6,"TIT":1081.4,"TAT":546.2,"CDP":12.1,"TEY":133.5}
features = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "TEY"]

cols = st.columns(3)
readings = {}
for i, f in enumerate(features):
    lo, hi = RANGES[f]
    span = hi - lo
    step = round(span / 100, 3) if span < 50 else round(span / 200, 2)
    
    with cols[i % 3]:
        st.markdown(f"<p style='color: white; font-weight: 600; margin-bottom: 5px;'>{LABELS[f]}</p>", unsafe_allow_html=True)
        
        if data_source == "Внешний сервис (HTTP)":
            val = float(st.session_state.scada_data[f])
            st.markdown(f"<div class='scada-display'>{val:.2f} {UNITS[f]}</div>", unsafe_allow_html=True)
            readings[f] = val
        else:
            # Для ползунка ставим дефолтные значения, но ограничиваем рамками 3-х сигм (немного расширим для UI)
            readings[f] = st.number_input(f"_{f}", label_visibility="collapsed", min_value=float(lo - span * 0.5), max_value=float(hi + span * 0.5), value=float(defaults[f]), step=float(step), format="%.2f")
        
        status_text, status_class = get_metric_status(readings[f], lo, hi)
        
        if status_text != "Норма":
            action = ACTIONS[f][0] if readings[f] > hi else ACTIONS[f][1]
            tooltip_border = "tooltip-danger" if "Критически" in status_text else ""
            st.markdown(f'''
            <div class="status-badge {status_class} tooltip-container">
                {status_text} (3σ: {lo:.0f} - {hi:.0f})
                <span class="tooltip-text {tooltip_border}">{action}</span>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-badge {status_class}">{status_text} (3σ: {lo:.0f} - {hi:.0f})</div>', unsafe_allow_html=True)

# --- ПРОГНОЗ NOX ---
row = pd.DataFrame([readings])
nox_model = models_dict['NOX']

# Защита от ошибки порядка колонок
if hasattr(nox_model, 'feature_names_in_'):
    row = row[nox_model.feature_names_in_]

nox = float(nox_model.predict(row)[0])

threshold = st.sidebar.slider("Пороговое значение NOx (мг/м³)", 20, 120, 80, 1)
is_ok = nox <= threshold

st.markdown('<div class="section-title">Прогноз концентрации NOx</div>', unsafe_allow_html=True)
color_class = "good" if is_ok else "pulse-danger"
st.markdown(f"""
<div class="pred-box {color_class}">
    <h2>{nox:.1f} <span style="font-size: 1.5rem;">мг/м³</span></h2>
    <p>{'В пределах допустимого' if is_ok else 'Превышение порога'}</p>
</div>
""", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
with col_btn1:
    if st.button("Общий анализ / Сохранить", use_container_width=True):
        if save_to_db_3nf(st.session_state.user_id, readings, nox, threshold):
            st.success("Запись успешно сохранена!")

st.markdown('<div class="section-title">Диагностика и рекомендации</div>', unsafe_allow_html=True)

html_recommendations = ""
problems = 0
for f in features:
    val = readings[f]
    lo, hi = RANGES[f]
    if val > hi or val < lo:
        problems += 1
        status, _ = get_metric_status(val, lo, hi)
        card_class = "card-danger" if "Критически" in status else "card-warn"
        action_text = ACTIONS[f][0] if val > hi else ACTIONS[f][1]
        
        html_recommendations += f'<div class="card {card_class}"><b>{LABELS[f]}</b> — {status} (Фактически: {val:.2f}).<br><b>Действие:</b> {action_text}</div>'

if problems == 0:
    html_recommendations = '<div class="card card-ok"><b>Все датчики в норме.</b> Установка работает в штатном режиме.</div>'
st.markdown(html_recommendations, unsafe_allow_html=True)

st.markdown('<div class="section-title">Аналитика</div>', unsafe_allow_html=True)

fig_dyn = go.Figure(go.Bar(
    x=[(readings[f] - ((RANGES[f][1] + RANGES[f][0]) / 2)) / ((RANGES[f][1] - RANGES[f][0]) / 2) for f in features], 
    y=features, orientation='h', 
    marker_color=["#FF4444" if get_metric_status(readings[f], RANGES[f][0], RANGES[f][1])[1] == "status-critical" else "#FFB020" if get_metric_status(readings[f], RANGES[f][0], RANGES[f][1])[1] == "status-warning" else "#3AE39A" for f in features],
    textposition='auto'
))
fig_dyn.update_layout(
    height=300, 
    margin=dict(l=0, r=0, t=10, b=0), 
    plot_bgcolor="rgba(0,0,0,0)", 
    paper_bgcolor="rgba(0,0,0,0)", 
    font=dict(color="#ffffff")
)
fig_dyn.update_xaxes(showgrid=True, gridcolor='#182233', zeroline=True, zerolinecolor='#ffffff')
st.plotly_chart(fig_dyn, use_container_width=True)

st.markdown('<div class="section-title">История показаний</div>', unsafe_allow_html=True)
df_history = load_history_3nf()
if len(df_history) > 0:
    df_display = df_history.copy()
    df_display.columns = ["ID", "Оператор", "Дата", "Порог", "AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "TEY", "NOx", "Статус"]
    st.dataframe(df_display, use_container_width=True, hide_index=True)
else:
    st.info("База данных пуста.")
