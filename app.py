"""
Flask-backend for hesteløp-analyseprogrammet.
"""
import datetime
import threading
import time
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from database import get_conn, init_db
from scraper import fetch_all, SOURCES
import scraper as _scraper
import analyzer
import predictor
import ml_model

app = Flask(__name__)
CORS(app)
init_db()

# ── Bakgrunnsjobb-state ───────────────────────────────────────────────────────

_bulk_job = {
    "running":    False,
    "stop":       False,
    "total":      0,
    "done":       0,
    "skipped":    0,
    "fetched":    0,   # løp hentet totalt
    "errors":     0,
    "current":    "",
    "started_at": None,
    "finished_at": None,
    "log":        [],  # siste 50 meldinger
}
_bulk_lock = threading.Lock()

# ── ML-trenings-state ─────────────────────────────────────────────────────────

_ml_job = {
    "running":     False,
    "log":         [],
    "meta":        None,
    "error":       None,
    "started_at":  None,
    "finished_at": None,
}
_ml_lock = threading.Lock()


def _ml_log(msg: str):
    with _ml_lock:
        _ml_job["log"].append(msg)
        if len(_ml_job["log"]) > 30:
            _ml_job["log"].pop(0)


def _ml_worker():
    with _ml_lock:
        _ml_job.update({"running": True, "log": [], "meta": None, "error": None,
                         "started_at": datetime.datetime.now().isoformat(),
                         "finished_at": None})
    try:
        meta = ml_model.train(log_fn=_ml_log)
        with _ml_lock:
            _ml_job["meta"] = meta
    except Exception as e:
        _ml_log(f"❌ Feil: {e}")
        with _ml_lock:
            _ml_job["error"] = str(e)
    finally:
        with _ml_lock:
            _ml_job["running"]     = False
            _ml_job["finished_at"] = datetime.datetime.now().isoformat()


def _log(msg: str):
    with _bulk_lock:
        _bulk_job["log"].append(msg)
        if len(_bulk_job["log"]) > 50:
            _bulk_job["log"].pop(0)


# ── Hjelpefunksjoner ─────────────────────────────────────────────────────────

def _save_races(races: list[dict]) -> int:
    """Lagrer løp og returnerer antall nye løp lagret.

    Duplikatbeskyttelse (tre lag):
      1. races.race_id er UNIQUE → INSERT OR IGNORE hopper over kjente løp.
      2. Resultater skrives bare inn når løpet er nytt (cur.rowcount == 1).
      3. results(race_id, horse_name) har UNIQUE-indeks + INSERT OR IGNORE
         som siste sikkerhetsnett mot delvise gjenhentinger.
    """
    saved = 0
    with get_conn() as conn:
        for race in races:
            if "error" in race:
                continue
            try:
                cur = conn.execute("""
                    INSERT OR IGNORE INTO races
                        (race_id, track, date, distance, surface, race_type, blood_type, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (race["race_id"], race.get("track", ""), race.get("date", ""),
                      race.get("distance"), race.get("surface", "grus"),
                      race.get("race_type", "trav"), race.get("blood_type", "varmblod"),
                      race.get("source", "")))
            except Exception:
                continue

            if not cur.rowcount:
                # Løpet fantes allerede — hopp over resultater
                continue

            saved += 1
            for entry in race.get("results", []):
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO results
                            (race_id, horse_name, position, start_pos,
                             jockey, trainer, odds, time_sec,
                             extra_distance, win_odds, horse_reg_no)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (race["race_id"], entry.get("horse_name", ""),
                          entry.get("position"), entry.get("start_pos"),
                          entry.get("jockey", ""), entry.get("trainer", ""),
                          entry.get("odds"), entry.get("time_sec"),
                          entry.get("extra_distance", 0),
                          entry.get("win_odds"),
                          entry.get("horse_reg_no")))
                except Exception:
                    pass
    return saved


# ── Bakgrunnshenting ──────────────────────────────────────────────────────────

BATCH_DAYS = 30   # antall dager per list-API-kall


def _bulk_worker(
    date_from: str,
    date_to: str,
    countries: list[str],
    delay: float,
    sources: list[str] | None = None,
):
    """
    Henter historiske løp i perioden date_from–date_to.

    Kilder:
      - rikstoto  → JSON-API, støtter NO/SE/DK/FI, mangler tid/distanse/bloddtype
      - travsport → HTML-scraping av travsport.no, kun NO, men har tid/distanse/kald-varm

    Fremgang måles i racedays (stevner), ikke datoer.
    """
    global _bulk_job
    if sources is None:
        sources = ["rikstoto"]

    with _bulk_lock:
        _bulk_job.update({"running": True, "stop": False, "done": 0,
                          "skipped": 0, "fetched": 0, "errors": 0,
                          "total": 0, "current": "Finner stevner…", "log": [],
                          "started_at": datetime.datetime.now().isoformat(),
                          "finished_at": None})

    try:
        # ── Fase 1: samle alle oppgaver fra valgte kilder ─────────────────────
        # Hvert element: {name, date, type, meta}
        all_tasks: list[dict] = []

        # Rikstoto-racedays (batching)
        if "rikstoto" in sources:
            cur = datetime.date.fromisoformat(date_from)
            end = datetime.date.fromisoformat(date_to)
            while cur <= end:
                with _bulk_lock:
                    if _bulk_job["stop"]:
                        break
                batch_end = min(cur + datetime.timedelta(days=BATCH_DAYS - 1), end)
                rds = _scraper.get_racedays_for_period(
                    cur.isoformat(), batch_end.isoformat(),
                    countries=countries, only_finished=True
                )
                for rd in rds:
                    all_tasks.append({
                        "name": rd["raceDayName"],
                        "date": rd["date"],
                        "type": "rikstoto",
                        "meta": rd,
                    })
                _log(f"🔍 Rikstoto {cur} → {batch_end}: {len(rds)} racedays")
                cur = batch_end + datetime.timedelta(days=1)
                time.sleep(0.3)

        # Travsport-stevner (kalender-scraping)
        if "travsport" in sources:
            with _bulk_lock:
                _bulk_job["current"] = "Leser travsport.no-kalender…"
            ts_metas = _scraper.get_travsport_result_urls(date_from, date_to)
            for meta in ts_metas:
                all_tasks.append({
                    "name": meta["track_name"],
                    "date": meta["date"],
                    "type": "travsport",
                    "meta": meta,
                })
            _log(f"🔍 Travsport {date_from} → {date_to}: {len(ts_metas)} stevner")

        # ATG-løp (dag-for-dag via kalender-API)
        if "atg" in sources:
            with _bulk_lock:
                _bulk_job["current"] = "Leser ATG-kalender…"
            cur = datetime.date.fromisoformat(date_from)
            end = datetime.date.fromisoformat(date_to)
            atg_count = 0
            while cur <= end:
                race_ids = _scraper.get_atg_race_ids(cur.isoformat())
                for rid in race_ids:
                    all_tasks.append({
                        "name": f"ATG {rid}",
                        "date": cur.isoformat(),
                        "type": "atg",
                        "meta": {"race_id": rid},
                    })
                    atg_count += 1
                cur += datetime.timedelta(days=1)
                time.sleep(0.2)
            _log(f"🔍 ATG {date_from} → {date_to}: {atg_count} løp")

        with _bulk_lock:
            _bulk_job["total"] = len(all_tasks)
        src_label = " + ".join(sources)
        _log(f"📋 Totalt {len(all_tasks)} stevner å hente ({src_label})")

        # ── Fase 2: hent hvert stevne ─────────────────────────────────────────
        for task in all_tasks:
            with _bulk_lock:
                if _bulk_job["stop"]:
                    _log("⛔ Stoppet av bruker")
                    break
                _bulk_job["current"] = f"{task['name']} ({task['date']})"

            try:
                if task["type"] == "travsport":
                    races = _scraper.fetch_travsport_raceday(task["meta"])
                elif task["type"] == "atg":
                    race = _scraper.fetch_atg_race(task["meta"]["race_id"])
                    races = [race] if race else []
                else:
                    races = _scraper.fetch_raceday(task["meta"])
                total_new = _save_races(races) if races else 0
            except Exception as e:
                with _bulk_lock:
                    _bulk_job["errors"] += 1
                _log(f"❌ {task['name']}: {e}")
                total_new = 0

            with _bulk_lock:
                _bulk_job["done"]    += 1
                _bulk_job["fetched"] += total_new

            if total_new:
                _log(f"✅ {task['name']} {task['date']} – {total_new} nye løp")
            else:
                with _bulk_lock:
                    _bulk_job["skipped"] += 1

            time.sleep(delay)

    finally:
        with _bulk_lock:
            _bulk_job["running"]     = False
            _bulk_job["current"]     = ""
            _bulk_job["finished_at"] = datetime.datetime.now().isoformat()
        _log("🏁 Ferdig")


# ── Sider ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API: enkel scraping ───────────────────────────────────────────────────────

@app.route("/api/fetch")
def api_fetch():
    date      = request.args.get("date", datetime.date.today().isoformat())
    countries = request.args.getlist("countries") or ["NO"]
    sources   = request.args.getlist("sources") or ["rikstoto"]
    data      = fetch_all(date, sources=sources, countries=countries)

    total_new = 0
    for src_races in data.values():
        if isinstance(src_races, list):
            total_new += _save_races(src_races)

    summary = {src: len(v) if isinstance(v, list) else v
               for src, v in data.items()}
    return jsonify({"status": "ok", "date": date, "sources": sources,
                    "countries": countries, "summary": summary})


# ── API: bulk-henting ─────────────────────────────────────────────────────────

@app.route("/api/fetch/bulk", methods=["POST"])
def api_fetch_bulk():
    if _bulk_job["running"]:
        return jsonify({"error": "En jobb kjører allerede"}), 409

    body      = request.get_json(force=True) or {}
    days      = min(int(body.get("days", 365)), 730)
    countries = body.get("countries") or ["NO"]
    sources   = body.get("sources") or ["rikstoto"]
    delay     = float(body.get("delay", 1.0))

    today     = datetime.date.today()
    date_from = (today - datetime.timedelta(days=days)).isoformat()
    date_to   = today.isoformat()

    t = threading.Thread(
        target=_bulk_worker,
        args=(date_from, date_to, countries, delay, sources),
        daemon=True,
    )
    t.start()

    return jsonify({"status": "started", "date_from": date_from,
                    "date_to": date_to, "countries": countries,
                    "sources": sources})


@app.route("/api/fetch/bulk/status")
def api_fetch_bulk_status():
    with _bulk_lock:
        return jsonify(dict(_bulk_job))


@app.route("/api/fetch/bulk/stop", methods=["POST"])
def api_fetch_bulk_stop():
    with _bulk_lock:
        _bulk_job["stop"] = True
    return jsonify({"status": "stop_requested"})


# ── API: statistikk ───────────────────────────────────────────────────────────

@app.route("/api/leaderboard")
def api_leaderboard():
    limit      = int(request.args.get("limit", 20))
    blood_type = request.args.get("blood_type", None)
    return jsonify(analyzer.leaderboard(limit, blood_type))


@app.route("/api/horses")
def api_horses():
    return jsonify(analyzer.all_horses(
        search     = request.args.get("q", ""),
        blood_type = request.args.get("blood_type"),
        sort       = request.args.get("sort", "wins"),
        page       = int(request.args.get("page", 1)),
        per_page   = int(request.args.get("per_page", 50)),
    ))


@app.route("/api/horse/<name>")
def api_horse(name):
    return jsonify(analyzer.horse_stats(name))


@app.route("/api/jockey/<name>")
def api_jockey(name):
    return jsonify(analyzer.jockey_stats(name))


@app.route("/api/track/<name>")
def api_track(name):
    return jsonify(analyzer.track_stats(name))


@app.route("/api/races")
def api_races():
    limit      = int(request.args.get("limit", 50))
    blood_type = request.args.get("blood_type", None)
    return jsonify(analyzer.recent_races(limit, blood_type))


@app.route("/api/race/<race_id>")
def api_race(race_id):
    return jsonify(analyzer.race_detail(race_id))


# ── API: prediksjon ───────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body    = request.get_json(force=True)
    horses  = body.get("horses", [])
    jockeys = body.get("jockeys", {})
    if not horses:
        return jsonify({"error": "Tom hesteliste"}), 400
    ranked  = ml_model.predict_field(horses, jockeys)
    race_id = body.get("race_id")
    if race_id:
        predictor.save_predictions(race_id, ranked)
    return jsonify(ranked)


# ── API: ML-modell ────────────────────────────────────────────────────────────

@app.route("/api/ml/train", methods=["POST"])
def api_ml_train():
    with _ml_lock:
        if _ml_job["running"]:
            return jsonify({"error": "Trening kjører allerede"}), 409
    t = threading.Thread(target=_ml_worker, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/ml/status")
def api_ml_status():
    with _ml_lock:
        job = dict(_ml_job)
    job["is_trained"] = ml_model.is_trained()
    if not job["meta"]:
        job["meta"] = ml_model.get_meta()
    return jsonify(job)


# ── API: heste-aliaser / sammenslåing ────────────────────────────────────────

@app.route("/api/horse/duplicates")
def api_horse_duplicates():
    limit = int(request.args.get("limit", 30))
    return jsonify(analyzer.suggest_duplicates(limit))


@app.route("/api/horse/merge", methods=["POST"])
def api_horse_merge():
    body = request.get_json(force=True) or {}
    alias = body.get("alias", "").strip()
    canonical = body.get("canonical", "").strip()
    if not alias or not canonical:
        return jsonify({"error": "Mangler 'alias' eller 'canonical'"}), 400
    result = analyzer.merge_horses(alias, canonical)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/horse/aliases")
def api_horse_aliases():
    return jsonify(analyzer.list_aliases())


@app.route("/api/horse/alias/<path:alias_name>", methods=["DELETE"])
def api_horse_alias_delete(alias_name):
    result = analyzer.delete_alias(alias_name)
    return jsonify(result)


@app.route("/api/search/horses")
def api_search_horses():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT horse_name FROM results
            WHERE LOWER(horse_name) LIKE LOWER(?)
            ORDER BY horse_name LIMIT 20
        """, (f"%{q}%",)).fetchall()
    return jsonify([r["horse_name"] for r in rows])


@app.route("/api/stats/bloodtype")
def api_bloodtype_stats():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT rc.blood_type,
                   COUNT(DISTINCT rc.race_id) AS races,
                   COUNT(DISTINCT LOWER(r.horse_name)) AS horses
            FROM races rc
            LEFT JOIN results r ON rc.race_id = r.race_id
            GROUP BY rc.blood_type
        """).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/stats/overview")
def api_overview():
    with get_conn() as conn:
        races   = conn.execute("SELECT COUNT(*) AS c FROM races").fetchone()["c"]
        results = conn.execute("SELECT COUNT(*) AS c FROM results").fetchone()["c"]
        horses  = conn.execute("SELECT COUNT(DISTINCT horse_name) AS c FROM results").fetchone()["c"]
        latest  = conn.execute("SELECT MAX(date) AS d FROM races").fetchone()["d"]
        earliest = conn.execute("SELECT MIN(date) AS d FROM races").fetchone()["d"]
    return jsonify({
        "total_races":    races,
        "total_results":  results,
        "unique_horses":  horses,
        "latest_date":    latest,
        "earliest_date":  earliest,
    })


@app.route("/api/demo")
def api_demo():
    days = int(request.args.get("days", 365))
    from demo_data import generate_demo
    generate_demo(days=days)
    with get_conn() as conn:
        races = conn.execute("SELECT COUNT(*) AS c FROM races").fetchone()["c"]
    return jsonify({"status": "ok", "message": f"Demo-data lastet: {races} løp i databasen"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
