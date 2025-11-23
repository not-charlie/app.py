# app.py
# Smart Fertilizer Recommender Dashboard + Multi-Soil Crop Selection
import os
import re
import threading
import time
import math
import sqlite3
import socket
from dataclasses import dataclass
from typing import Dict, Tuple, List
from flask import Flask, render_template, request, jsonify, redirect, url_for

try:
    import serial
except ImportError:
    serial = None

# Local network access only

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
    if serial is None:
        state.ok = False
        state.msg = "pyserial not installed; cannot read Arduino."
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
t = threading.Thread(target=serial_reader, daemon=True)
t.start()

# ---------------- Fertilizer Recommendation ----------------
def recommend_fertilizer(crop: str, N: float, P: float, K: float):
    crop = crop.lower()
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
    # Existing aggregated results table (kept for backward compat if needed)
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
    # Table for soil samples
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
    # Table for crops explicitly selected for each soil
    c.execute("""
        CREATE TABLE IF NOT EXISTS soil_selected_crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            soil_id INTEGER,
            crop_name TEXT,
            UNIQUE(soil_id, crop_name),
            FOREIGN KEY(soil_id) REFERENCES soil_samples(id) ON DELETE CASCADE
        )
    """)
    # New table: per-soil per-crop test results (ranked)
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
    """
    Save or update a soil sample. If the soil sample is locked (finalized after testing),
    do not overwrite unless `force=True` (explicit operation such as running tests or user save).
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Check if existing sample is locked
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
    # If exists update, else insert
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
    """
    results: list of dicts with keys: crop (string), score (float), fertilizer (string), rank (int)
    This function deletes any previous test results for the soil and inserts the new ones.
    """
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
    # After saving test results, mark the soil as locked (finalized)
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
    c.execute("""
        SELECT crop_name FROM soil_selected_crops
        WHERE soil_id = ?
        ORDER BY id ASC
    """, (soil_id,))
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
    """Safe helper to read a key from sqlite3.Row or dict-like objects."""
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

# ---------------- Flask App ----------------
app = Flask(__name__)

def get_local_ip():
    """Get the local IP address for network access"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

@app.route("/")
def home():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    local_ip = get_local_ip()
    return render_template("dashboard.html", sensor_ok=state.ok, sensor_msg=state.msg, local_ip=local_ip)

# Select crop page: shows live sensor readings and allows saving raw soil samples
@app.route("/select_crop")
def select_crop():
    crop = request.args.get("crop", "tomato")
    soil_name = request.args.get("soil")
    N, P, K = state.N, state.P, state.K
    fert_name, fert_npk, meta = recommend_fertilizer(crop, N, P, K)
    meta.setdefault("targets_scaled", {"N": 0, "P": 0, "K": 0})
    meta.setdefault("deficits", {"N": 0, "P": 0, "K": 0})

    soil_names = SOIL_CHOICES
    if not soil_name:
        soil_name = soil_names[0]

    return render_template(
        "select_crop.html",
        crop_options=CROP_OPTIONS,
        selected_crop=crop,
        soil_options=soil_names,
        selected_soil=soil_name,
        sensor_ok=state.ok,
        sensor_msg=state.msg,
        N=None if state.N != state.N else round(state.N, 2),
        P=None if state.P != state.P else round(state.P, 2),
        K=None if state.K != state.K else round(state.K, 2),
        recommended=fert_name,
        rec_n=fert_npk[0],
        rec_p=fert_npk[1],
        rec_k=fert_npk[2],
        meta=meta
    )

# Legacy endpoint (kept for compatibility)
@app.route("/test_soil", methods=["POST"])
def test_soil():
    soil_name = request.form.get("soil_name", "").strip()
    crop_name = request.form.get("crop_name", "").strip() or None
    if not soil_name:
        return jsonify({"error": "Missing soil name"}), 400
    N, P, K = state.N, state.P, state.K
    if any(math.isnan(x) for x in (N, P, K)):
        return jsonify({"error": "Invalid or missing NPK readings"}), 400

    save_raw_soil_sample(soil_name, N, P, K, crop_name)
    return jsonify({"success": True, "soil": soil_name, "crop": crop_name, "N": round(N, 2), "P": round(P, 2), "K": round(K, 2)})

# API endpoint for saving soil from select_crop page
@app.route("/api/save_soil", methods=["POST"])
def api_save_soil():
    """Save current live sensor readings for a specific soil and crop"""
    data = request.get_json() or {}
    soil_name = (data.get("soil_name") or "").strip()
    crop_name = (data.get("crop_name") or "").strip()
    
    if not soil_name:
        return jsonify({"error": "Missing soil selection", "msg": "Please select a soil first."}), 400
    if not crop_name:
        return jsonify({"error": "Missing crop selection", "msg": "Please select a crop to test."}), 400
    
    N, P, K = state.N, state.P, state.K
    if any(math.isnan(x) for x in (N, P, K)):
        return jsonify({"error": "Invalid or missing NPK readings", "msg": "No valid sensor readings available."}), 400
    
    # Use force=True to allow saving even if soil is locked (user explicitly saving new data)
    save_raw_soil_sample(soil_name, N, P, K, crop_name, force=True)
    return jsonify({
        "success": True,
        "msg": f"✓ Saved {soil_name} for crop '{crop_name.title()}'! N: {round(N, 2)}, P: {round(P, 2)}, K: {round(K, 2)} mg/kg",
        "soil": soil_name,
        "crop": crop_name,
        "N": round(N, 2),
        "P": round(P, 2),
        "K": round(K, 2)
    })

# Run tests for a single soil (by name) OR all soils (if no 'soil' param)
# This computes top 10 crops per soil and saves them into soil_test_results
@app.route("/run_tests", methods=["POST"])
def run_tests():
    """Run suitability tests for selected soil using current live sensor reading."""
    soil_name = request.form.get("soil")
    if not soil_name:
        return jsonify({"error": "Missing soil parameter"}), 400
    
    # Use live sensor data
    N, P, K = state.N, state.P, state.K
    if any(math.isnan(x) for x in (N, P, K)):
        return jsonify({"error": "Invalid or missing NPK readings from sensor"}), 400

    # Get or create soil entry
    soil_row = get_soil_by_name(soil_name)
    if not soil_row:
        # Create new soil entry with live data
        save_raw_soil_sample(soil_name, N, P, K, None, force=True)
        soil_row = get_soil_by_name(soil_name)
    
    soil_id = soil_row["id"]
    
    # Unlock the soil and clear old test results when starting a new test
    # This allows new data to be saved and ensures fresh test results
    clear_test_results_for_soil(soil_id)
    set_soil_locked(soil_id, False)
    
    # Update soil with latest live sensor readings (force update while running tests)
    save_raw_soil_sample(soil_name, N, P, K, row_get(soil_row, "last_crop"), force=True)

    selected_crops = get_selected_crops_for_soil(soil_id)
    if not selected_crops:
        return jsonify({"error": f"No crops have been saved for {soil_name} yet. Save crops first in Select Crop page."}), 400

    crop_scores = []
    for crop in selected_crops:
        crop_key = crop.lower()
        if crop_key not in CROP_TARGETS:
            continue
        cN, cP, cK = CROP_TARGETS[crop_key]
        score = suitability_score(N, P, K, cN, cP, cK)
        fert_name, _, _ = recommend_fertilizer(crop_key, N, P, K)
        crop_scores.append({
            "crop": crop.title(),
            "score": score,
            "fertilizer": fert_name
        })

    if not crop_scores:
        return jsonify({"error": "Saved crops are not recognized in the current crop list."}), 400

    top_crops = sorted(crop_scores, key=lambda x: x["score"], reverse=True)

    results = []
    for idx, r in enumerate(top_crops, start=1):
        results.append({
            "crop": r["crop"],
            "score": r["score"],
            "fertilizer": r["fertilizer"],
            "rank": idx
        })

    save_test_results_for_soil(soil_id, results)

    return jsonify({"success": True, "processed": [{"soil": soil_name, "tested_count": len(results)}]})

# Show crop recommendations filtered by selected soil
@app.route("/show_crop", methods=["GET"])
def show_crop():
    selected_soil = request.args.get("soil")
    
    soil_options = SOIL_CHOICES
    if not selected_soil:
        return render_template("show_crop.html", soils=soil_options, selected_soil=None)
    if selected_soil not in soil_options:
        return render_template("show_crop.html", error="Soil not recognized. Please choose from Soil 1-10.", soils=soil_options, selected_soil=None)

    soil_row = get_soil_by_name(selected_soil)
    soil_id = soil_row["id"] if soil_row else None
    N = soil_row["N"] if soil_row else None
    P = soil_row["P"] if soil_row else None
    K = soil_row["K"] if soil_row else None

    rows = get_test_results_for_soil_id(soil_id) if soil_id else []
    top_crops = [{
        "crop": r["crop_name"],
        "score": r["suitability_score"],
        "fertilizer": r["fertilizer_recommendation"],
        "rank": r["rank"],
        "timestamp": row_get(r, "timestamp", "")
    } for r in rows] if rows else []
    test_timestamp = row_get(rows[0], "timestamp", "") if rows else ""

    return render_template(
        "show_crop.html",
        soils=soil_options,
        selected_soil=selected_soil,
        top_crops=top_crops,
        N=N,
        P=P,
        K=K,
        has_test_results=len(top_crops) > 0,
        test_timestamp=test_timestamp,
        last_crop=row_get(soil_row, "last_crop") if soil_row else ""
    )

@app.route("/api/top_crops")
def api_top_crops():
    """Return sensor readings and stored rankings for selected soil."""
    soil_name = request.args.get("soil")
    if not soil_name:
        return jsonify({"error": "Missing soil parameter"}), 400
    
    soil_row = get_soil_by_name(soil_name)
    soil_id = soil_row["id"] if soil_row else None
    N = float(soil_row["N"]) if soil_row else None
    P = float(soil_row["P"]) if soil_row else None
    K = float(soil_row["K"]) if soil_row else None
    timestamp = row_get(soil_row, "timestamp", "") if soil_row else ""
    rows = get_test_results_for_soil_id(soil_id) if soil_id else []

    top_crops = []
    test_timestamp = ""
    for r in rows:
        fert_name = r["fertilizer_recommendation"]
        fert_npk = FERTILIZERS.get(fert_name, (0, 0, 0))
        crop_timestamp = row_get(r, "timestamp", "")
        if crop_timestamp and not test_timestamp:
            test_timestamp = crop_timestamp
        top_crops.append({
            "crop": r["crop_name"],
            "score": float(r["suitability_score"]),
            "fertilizer": fert_name,
            "rank": int(r["rank"]),
            "n": fert_npk[0],
            "p": fert_npk[1],
            "k": fert_npk[2],
            "timestamp": crop_timestamp
        })
    # include locked status so UI can show finalised state
    locked_status = bool(row_get(soil_row, "locked", 0)) if soil_row else False

    return jsonify({
        "N": None if N is None else round(N, 2),
        "P": None if P is None else round(P, 2),
        "K": None if K is None else round(K, 2),
        "timestamp": timestamp,
        "test_timestamp": test_timestamp,
        "top_crops": top_crops,
        "has_test_results": len(top_crops) > 0,
        "last_crop": row_get(soil_row, "last_crop") if soil_row else "",
        "locked": locked_status
    })


@app.route("/api/start_new_test", methods=["POST"])
def api_start_new_test():
    """Unlock the soil sample for a new test and clear previous rankings.
    This must be called explicitly to allow the sample to be overwritten again.
    """
    soil_name = request.form.get("soil") or request.json and request.json.get("soil")
    if not soil_name:
        return jsonify({"error": "Missing soil parameter"}), 400
    soil_row = get_soil_by_name(soil_name)
    if not soil_row:
        return jsonify({"error": "Soil not found"}), 404
    soil_id = soil_row["id"]
    # clear previous stored test results and unlock
    clear_test_results_for_soil(soil_id)
    set_soil_locked(soil_id, False)
    return jsonify({"success": True, "msg": f"Unlocked {soil_name} for new testing."})

@app.route("/api/readings")
def api_readings():
    return jsonify({
        "ok": state.ok,
        "msg": state.msg,
        "N": None if state.N != state.N else state.N,
        "P": None if state.P != state.P else state.P,
        "K": None if state.K != state.K else state.K,
        "ts": state.ts
    })


if __name__ == "__main__":
    init_db()
    time.sleep(1.0)
    
    local_ip = get_local_ip()
    
    print(f"\n{'='*60}")
    print(" Smart Fertilizer Recommender - Starting Server")
    print(f"{'='*60}")
    print(f"\n LOCAL NETWORK ACCESS:")
    print(f"   http://localhost:5000 (this computer)")
    print(f"   http://{local_ip}:5000 (same Wi-Fi network)")
    print(f"\n{'='*60}\n")
    
    # Enable threading to support multiple concurrent connections (multiple phones)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
