# streamlit_app.py
# Smart Fertilizer Recommender — Streamlit conversion of your Flask app
# Requirements (put in requirements.txt):
# streamlit
# pyserial  # optional, only if you will use an Arduino sensor
# pandas
# (sqlite3 is in stdlib)

import os
import re
import threading
import time
import math
import sqlite3
import socket
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import streamlit as st

# try optional pyserial
try:
    import serial
except Exception:
    serial = None

# ---------------- Configuration ----------------
SERIAL_PORT = os.environ.get("SERIAL_PORT", "COM5")
BAUD_RATE = int(os.environ.get("BAUD_RATE", "9600"))
READ_TIMEOUT_S = 0.5
STALE_AFTER_S = 15

# Database file
DB_DIR = "data"
DB_FILE = os.path.join(DB_DIR, "soil_results.db")

# ---------------- Fertilizer and Soil Data ----------------
FERTILIZERS = {
    "Urea (46-0-0)": (46, 0, 0),
    "Complete (14-14-14)": (14, 14, 14),
    "Organic Compound (5-5-5)": (5, 5, 5),
    "No fertilizer": (0, 0, 0),
}

CROP_TARGETS: Dict[str, Tuple[float, float, float]] = {
    "tomato":      (1.0, 2.0, 2.0),
    "patola":      (1.0, 1.5, 2.0),
    "okra":        (1.0, 1.0, 1.0),
    "ampalaya":    (1.0, 1.5, 2.0),
    "chayote":     (1.0, 1.5, 2.0),
    "carrot":      (1.0, 1.2, 1.2),
    "string bean": (0.5, 1.0, 1.5),
    "cabbage":     (2.0, 1.0, 1.0),
    "chili pepper":(1.0, 2.0, 2.0),
    "eggplant":    (1.0, 1.0, 1.5),
}

CROP_OPTIONS = list(CROP_TARGETS.keys())
SOIL_CHOICES = [f"Soil {i}" for i in range(1, 11)]

# ---------------- Shared Sensor State ----------------
@dataclass
class NPKState:
    N: float = float("nan")
    P: float = float("nan")
    K: float = float("nan")
    ts: float = 0.0
    ok: bool = False
    msg: str = "Waiting for data..."

state = NPKState()

# ---------------- Serial Reader ----------------

def parse_line_for_value(line: str, key: str) -> float:
    if key.lower() not in line.lower():
        return float("nan")
    m = re.search(r"(-?\d+(\.\d+)?)", line)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return float("nan")
    return float("nan")


def serial_reader():
    """Background thread that reads serial lines and updates `state`.
    If `serial` is None, it will set a helpful message and exit.
    """
    if serial is None:
        state.ok = False
        state.msg = "pyserial not installed; running without sensor."
        return
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=READ_TIMEOUT_S) as ser:
            state.msg = f"Connected to {SERIAL_PORT} @ {BAUD_RATE}"
            while True:
                try:
                    line = ser.readline().decode(errors="ignore").strip()
                    if not line:
                        if time.time() - state.ts > STALE_AFTER_S:
                            state.ok = False
                            state.msg = "Stale: no new data."
                        continue

                    n_val = parse_line_for_value(line, "Nitrogen")
                    p_val = parse_line_for_value(line, "Phosphorus")
                    k_val = parse_line_for_value(line, "Potassium")

                    updated = False
                    if not (n_val != n_val):
                        state.N = n_val; updated = True
                    if not (p_val != p_val):
                        state.P = p_val; updated = True
                    if not (k_val != k_val):
                        state.K = k_val; updated = True

                    if updated:
                        state.ts = time.time()
                        if all(not (x != x) for x in (state.N, state.P, state.K)):
                            state.ok = True
                            state.msg = "OK"
                except Exception as e:
                    state.ok = False
                    state.msg = f"Read error: {e}"
                    time.sleep(0.5)
    except Exception as e:
        state.ok = False
        state.msg = f"Open error: {e}"


# start serial thread (harmless if pyserial not installed)
serial_thread_started = False

def start_serial_thread():
    global serial_thread_started
    if serial_thread_started:
        return
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()
    serial_thread_started = True

# ---------------- Fertilizer Recommendation ----------------

def recommend_fertilizer(crop: str, N: float, P: float, K: float):
    crop = (crop or "").lower()
    if crop not in CROP_TARGETS:
        return ("No fertilizer", (0,0,0), {"reason": "Unknown crop", "deficits": {}})

    values = [max(0.0, float(x)) for x in (N, P, K)]
    total_actual = sum(values)
    if total_actual <= 0:
        return ("No fertilizer", (0,0,0), {"reason": "No sensor values", "deficits": {}})

    tn, tp, tk = CROP_TARGETS[crop]
    total_target = tn + tp + tk
    scale = total_actual / total_target if total_target > 0 else 0
    targetN, targetP, targetK = tn * scale, tp * scale, tk * scale

    eps = 1e-6
    def deficit(actual, target):
        if target < eps: return 0.0
        return max(0.0, (target - actual) / target)

    dN, dP, dK = deficit(values[0], targetN), deficit(values[1], targetP), deficit(values[2], targetK)

    mild, strong = 0.10, 0.25
    onlyN = (dN > mild) and (dP <= mild) and (dK <= mild)
    anyPK = (dP > mild) or (dK > mild)
    bigPK = (dP > strong) or (dK > strong)

    if onlyN:
        fert_name = "Urea (46-0-0)"
    elif anyPK:
        fert_name = "Complete (14-14-14)" if bigPK else "Organic Compound (5-5-5)"
    else:
        fert_name = "No fertilizer"

    return (fert_name, FERTILIZERS[fert_name], {
        "deficits": {"N": round(dN,3), "P": round(dP,3), "K": round(dK,3)},
        "targets_scaled": {"N": round(targetN,2), "P": round(targetP,2), "K": round(targetK,2)},
    })

# ---------------- Crop Suitability Logic ----------------

def suitability_score(soil_N, soil_P, soil_K, crop_N, crop_P, crop_K):
    total_soil = soil_N + soil_P + soil_K
    total_crop = crop_N + crop_P + crop_K
    if total_soil == 0 or total_crop == 0:
        return 0
    sN, sP, sK = soil_N / total_soil, soil_P / total_soil, soil_K / total_soil
    cN, cP, cK = crop_N / total_crop, crop_P / total_crop, crop_K / total_crop
    diff = abs(sN - cN) + abs(sP - cP) + abs(sK - cK)
    return round(1 - diff / 2, 3)

# ---------------- SQLite Helpers ----------------

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS soil_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            crop TEXT,
            N REAL,
            P REAL,
            K REAL,
            recommended TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS soil_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            N REAL,
            P REAL,
            K REAL,
            timestamp TEXT,
            last_crop TEXT,
            locked INTEGER DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS soil_selected_crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            soil_id INTEGER,
            crop_name TEXT,
            UNIQUE(soil_id, crop_name),
            FOREIGN KEY(soil_id) REFERENCES soil_samples(id) ON DELETE CASCADE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS soil_test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            soil_id INTEGER,
            crop_name TEXT,
            suitability_score REAL,
            fertilizer_recommendation TEXT,
            rank INTEGER,
            timestamp TEXT,
            FOREIGN KEY(soil_id) REFERENCES soil_samples(id)
        )
    """)
    try:
        c.execute("ALTER TABLE soil_samples ADD COLUMN last_crop TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE soil_samples ADD COLUMN locked INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def save_raw_soil_sample(soil_name: str, N: float, P: float, K: float, crop_name: str = None, force: bool = False):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT locked FROM soil_samples WHERE name = ?", (soil_name,))
    existing = c.fetchone()
    if existing is not None:
        try:
            locked_val = int(existing[0])
        except Exception:
            locked_val = 0
        if locked_val == 1 and not force:
            conn.close()
            return

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if existing is not None:
        c.execute("""
            UPDATE soil_samples
            SET N = ?, P = ?, K = ?, timestamp = ?, last_crop = ?
            WHERE name = ?
        """, (N, P, K, ts, crop_name, soil_name))
    else:
        c.execute("""
            INSERT INTO soil_samples (name, N, P, K, timestamp, last_crop, locked)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (soil_name, N, P, K, ts, crop_name))

    conn.commit()
    c.execute("SELECT id FROM soil_samples WHERE name = ?", (soil_name,))
    row = c.fetchone()
    conn.close()
    soil_id = row[0] if row else None
    if crop_name and soil_id is not None:
        record_selected_crop(soil_id, crop_name)


def save_test_results_for_soil(soil_id: int, results: List[dict]):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM soil_test_results WHERE soil_id = ?", (soil_id,))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for r in results:
        c.execute("""
            INSERT INTO soil_test_results (soil_id, crop_name, suitability_score, fertilizer_recommendation, rank, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (soil_id, r["crop"], r["score"], r["fertilizer"], r["rank"], ts))
    conn.commit()
    conn.close()
    set_soil_locked(soil_id, True)


def set_soil_locked(soil_id: int, locked: bool):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE soil_samples SET locked = ? WHERE id = ?", (1 if locked else 0, soil_id))
    conn.commit()
    conn.close()


def clear_test_results_for_soil(soil_id: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM soil_test_results WHERE soil_id = ?", (soil_id,))
    conn.commit()
    conn.close()


def get_soil_id_by_name(name: str):
    row = get_soil_by_name(name)
    return row["id"] if row else None


def record_selected_crop(soil_id: int, crop_name: str):
    if soil_id is None or not crop_name:
        return
    crop_name = crop_name.lower()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO soil_selected_crops (soil_id, crop_name)
        VALUES (?, ?)
        ON CONFLICT(soil_id, crop_name) DO NOTHING
    """, (soil_id, crop_name))
    conn.commit()
    conn.close()


def get_selected_crops_for_soil(soil_id: int) -> List[str]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT crop_name FROM soil_selected_crops WHERE soil_id = ? ORDER BY id ASC", (soil_id,))
    rows = [r[0] for r in c.fetchall()]
    conn.close()
    return rows


def get_soil_by_name(name: str):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM soil_samples WHERE name = ?", (name,))
    row = c.fetchone()
    conn.close()
    return row


def row_get(row, key, default=None):
    if isinstance(row, sqlite3.Row):
        return row[key] if key in row.keys() else default
    try:
        return row.get(key, default)  # type: ignore[attr-defined]
    except AttributeError:
        try:
            return row[key]
        except Exception:
            return default


def get_test_results_for_soil_id(soil_id: int):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT crop_name, suitability_score, fertilizer_recommendation, rank, timestamp
        FROM soil_test_results
        WHERE soil_id = ?
        ORDER BY rank ASC
    """, (soil_id,))
    rows = c.fetchall()
    conn.close()
    return rows

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="Smart Fertilizer Recommender", layout="wide")

# initialize DB and start serial thread once
init_db()
start_serial_thread()

# Sidebar navigation
page = st.sidebar.selectbox("Page", ["Dashboard", "Select Crop", "Run Tests", "Show Crop", "Manage Samples"])

# Utility: format timestamp

def format_ts(ts: float) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

# Dashboard page
if page == "Dashboard":
    st.title("Smart Fertilizer Recommender — Dashboard")
    st.subheader("Sensor Status")
    cols = st.columns([2,3])
    with cols[0]:
        st.metric("Sensor OK", "Yes" if state.ok else "No", delta=None)
        st.write(state.msg)
        st.write("Last update:", format_ts(state.ts))
        if st.button("Refresh"):
            st.experimental_rerun()
    with cols[1]:
        st.write("### Latest NPK Readings")
        st.write("Nitrogen (N):", None if state.N != state.N else round(state.N,2))
        st.write("Phosphorus (P):", None if state.P != state.P else round(state.P,2))
        st.write("Potassium (K):", None if state.K != state.K else round(state.K,2))

    st.markdown("---")
    st.write("You can configure serial port via environment variable `SERIAL_PORT` and `BAUD_RATE`.")
    st.write("If running without a physical sensor, pyserial is optional and the app will function using manual inputs on the Select Crop page.")

# Select Crop page
elif page == "Select Crop":
    st.title("Select Crop & Save Soil Sample")
    st.write("Use live sensor readings (if connected) or enter N/P/K manually.")

    col1, col2 = st.columns([1,2])
    with col1:
        soil = st.selectbox("Select Soil", SOIL_CHOICES)
        crop = st.selectbox("Select Crop", [c.title() for c in CROP_OPTIONS])
        st.checkbox("Lock sample after saving (finalize)", key="lock_sample")

    with col2:
        st.write("### Sensor / Manual Inputs")
        use_sensor = st.checkbox("Use live sensor readings if available", value=True)
        if use_sensor and state.ok:
            N_val = None if state.N != state.N else round(state.N,2)
            P_val = None if state.P != state.P else round(state.P,2)
            K_val = None if state.K != state.K else round(state.K,2)
            st.write(f"Sensor readings (last: {format_ts(state.ts)})")
            st.write("N:", N_val, "P:", P_val, "K:", K_val)
        else:
            st.write("Enter NPK values manually (mg/kg)")
            N_val = st.number_input("Nitrogen (N)", min_value=0.0, value=0.0, format="%.2f")
            P_val = st.number_input("Phosphorus (P)", min_value=0.0, value=0.0, format="%.2f")
            K_val = st.number_input("Potassium (K)", min_value=0.0, value=0.0, format="%.2f")

    if st.button("Compute Recommendation"):
        if use_sensor and state.ok:
            N = state.N; P = state.P; K = state.K
        else:
            N = N_val; P = P_val; K = K_val
        fert_name, fert_npk, meta = recommend_fertilizer(crop, N, P, K)
        st.success(f"Recommended: {fert_name}")
        st.write("Fertilizer NPK:", fert_npk)
        st.json(meta)

    st.markdown("---")
    with st.form("save_sample"):
        st.write("### Save current sample to database")
        soil_name_input = st.text_input("Soil name", value=soil)
        crop_for_save = st.selectbox("Crop for this sample", [c.title() for c in CROP_OPTIONS], index=CROP_OPTIONS.index(crop.lower()) if crop.lower() in CROP_OPTIONS else 0)
        force_save = st.checkbox("Force save (overwrite locked)")
        submitted = st.form_submit_button("Save Sample")
        if submitted:
            if use_sensor and state.ok:
                N_save = state.N; P_save = state.P; K_save = state.K
            else:
                N_save = N_val; P_save = P_val; K_save = K_val
            if any(math.isnan(x) for x in (N_save, P_save, K_save)):
                st.error("Invalid or missing NPK values. Cannot save.")
            else:
                save_raw_soil_sample(soil_name_input, float(N_save), float(P_save), float(K_save), crop_for_save, force=force_save)
                st.success(f"Saved {soil_name_input} (Crop: {crop_for_save})")

# Run Tests page
elif page == "Run Tests":
    st.title("Run Suitability Tests for a Soil")
    soil_sel = st.selectbox("Choose soil to test", SOIL_CHOICES)
    if st.button("Start Test"):
        soil_row = get_soil_by_name(soil_sel)
        if not soil_row:
            st.error("No saved sample for this soil. Save it first on Select Crop page.")
        else:
            N, P, K = float(soil_row["N"]), float(soil_row["P"]), float(soil_row["K"]) if soil_row["N"] is not None else (None, None, None)
            # Use stored values for test
            if any(v is None for v in (N,P,K)):
                st.error("Saved sample has missing NPK values.")
            else:
                selected_crops = get_selected_crops_for_soil(soil_row["id"]) or [c.title() for c in CROP_OPTIONS]
                crop_scores = []
                for crop in selected_crops:
                    crop_key = crop.lower()
                    if crop_key not in CROP_TARGETS:
                        continue
                    cN, cP, cK = CROP_TARGETS[crop_key]
                    score = suitability_score(N, P, K, cN, cP, cK)
                    fert_name, _, _ = recommend_fertilizer(crop_key, N, P, K)
                    crop_scores.append({"crop": crop.title(), "score": score, "fertilizer": fert_name})

                if not crop_scores:
                    st.warning("No crops to evaluate.")
                else:
                    top_crops = sorted(crop_scores, key=lambda x: x["score"], reverse=True)
                    results = []
                    for idx, r in enumerate(top_crops, start=1):
                        results.append({"crop": r["crop"], "score": r["score"], "fertilizer": r["fertilizer"], "rank": idx})
                    save_test_results_for_soil(soil_row["id"], results)
                    st.success(f"Test completed and saved for {soil_sel}")

# Show Crop page
elif page == "Show Crop":
    st.title("Show Crop Recommendations")
    selected_soil = st.selectbox("Which soil?", [None] + SOIL_CHOICES)
    if not selected_soil:
        st.info("Select a soil to view recommendations and stored results.")
    else:
        soil_row = get_soil_by_name(selected_soil)
        if not soil_row:
            st.warning("No saved sample for this soil.")
        else:
            N = soil_row["N"]; P = soil_row["P"]; K = soil_row["K"]
            st.write("Saved sample (N,P,K):", N, P, K)
            rows = get_test_results_for_soil_id(soil_row["id"]) or []
            if not rows:
                st.info("No test results saved for this soil. Run tests first.")
            else:
                st.write("Test timestamp: ", row_get(rows[0], "timestamp", ""))
                table = []
                for r in rows:
                    table.append({
                        "Rank": r["rank"],
                        "Crop": r["crop_name"].title(),
                        "Score": float(r["suitability_score"]),
                        "Fertilizer": r["fertilizer_recommendation"],
                        "Timestamp": row_get(r, "timestamp", "")
                    })
                st.table(table)

# Manage Samples page
elif page == "Manage Samples":
    st.title("Manage Samples")
    st.write("View, unlock, or delete stored soil samples and selected crops.")
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, name, N, P, K, timestamp, locked, last_crop FROM soil_samples ORDER BY id ASC")
    samples = c.fetchall()
    conn.close()

    if not samples:
        st.info("No saved samples yet.")
    else:
        for s in samples:
            cols = st.columns([3,1,1,1,1])
            with cols[0]:
                st.write(f"**{s['name']}** — saved: {row_get(s,'timestamp','')} — Crop: {row_get(s,'last_crop','')}")
                st.write(f"N:{s['N']} P:{s['P']} K:{s['K']}")
            with cols[1]:
                if st.button(f"Unlock### {s['id']}"):
                    set_soil_locked(s['id'], False)
                    st.experimental_rerun()
            with cols[2]:
                if st.button(f"Clear results### {s['id']}"):
                    clear_test_results_for_soil(s['id'])
                    st.experimental_rerun()
            with cols[3]:
                if st.button(f"Delete sample### {s['id']}"):
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute("DELETE FROM soil_test_results WHERE soil_id = ?", (s['id'],))
                    c.execute("DELETE FROM soil_selected_crops WHERE soil_id = ?", (s['id'],))
                    c.execute("DELETE FROM soil_samples WHERE id = ?", (s['id'],))
                    conn.commit()
                    conn.close()
                    st.experimental_rerun()
            with cols[4]:
                st.write("Locked" if s['locked'] else "Unlocked")

# Footer: tips
st.sidebar.markdown("---")
st.sidebar.write("Tips:")
st.sidebar.write("• To use a physical Arduino sensor, install pyserial and set SERIAL_PORT and BAUD_RATE environment variables.")
st.sidebar.write("• To deploy: put this file in a GitHub repo with requirements.txt and deploy to Streamlit Cloud (share.streamlit.io)")


# End of file
