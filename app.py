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

def _save_horse_stats_atg(conn, horse_stats: list[dict]) -> None:
    """
    Oppdaterer horse_stats_atg-tabellen.
    Bruker UPSERT for å oppdatere eksisterende rader med nyere data.
    """
    for hs in horse_stats:
        if not hs.get("horse_reg_no"):
            continue
        try:
            conn.execute("""
                INSERT INTO horse_stats_atg
                    (horse_reg_no, name, age, sex, color, money,
                     life_starts, life_wins, life_2nd, life_3rd,
                     life_win_pct, life_place_pct, best_time_sec,
                     father_name, mother_name, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(horse_reg_no) DO UPDATE SET
                    name           = excluded.name,
                    age            = excluded.age,
                    sex            = excluded.sex,
                    color          = excluded.color,
                    money          = excluded.money,
                    life_starts    = excluded.life_starts,
                    life_wins      = excluded.life_wins,
                    life_2nd       = excluded.life_2nd,
                    life_3rd       = excluded.life_3rd,
                    life_win_pct   = excluded.life_win_pct,
                    life_place_pct = excluded.life_place_pct,
                    best_time_sec  = CASE
                        WHEN excluded.best_time_sec IS NULL THEN horse_stats_atg.best_time_sec
                        WHEN horse_stats_atg.best_time_sec IS NULL THEN excluded.best_time_sec
                        WHEN excluded.best_time_sec < horse_stats_atg.best_time_sec
                            THEN excluded.best_time_sec
                        ELSE horse_stats_atg.best_time_sec
                    END,
                    father_name    = COALESCE(excluded.father_name, horse_stats_atg.father_name),
                    mother_name    = COALESCE(excluded.mother_name, horse_stats_atg.mother_name),
                    updated_at     = datetime('now')
            """, (hs["horse_reg_no"], hs.get("name"), hs.get("age"),
                  hs.get("sex"), hs.get("color"), hs.get("money"),
                  hs.get("life_starts"), hs.get("life_wins"),
                  hs.get("life_2nd"), hs.get("life_3rd"),
                  hs.get("life_win_pct"), hs.get("life_place_pct"),
                  hs.get("best_time_sec"),
                  hs.get("father_name"), hs.get("mother_name")))
        except Exception:
            pass


def _save_races(races: list[dict]) -> int:
    """Lagrer løp og returnerer antall nye løp lagret.

    Duplikatbeskyttelse (tre lag):
      1. races.race_id er UNIQUE → INSERT OR IGNORE hopper over kjente løp.
      2. Resultater skrives bare inn når løpet er nytt (cur.rowcount == 1).
      3. results(race_id, horse_name) har UNIQUE-indeks + INSERT OR IGNORE
         som siste sikkerhetsnett mot delvise gjenhentinger.

    horse_stats (fra ATG) lagres alltid – også når løpet er kjent fra før –
    siden vi vil oppdatere heste-berikelse hver gang vi ser hesten.
    """
    saved = 0
    with get_conn() as conn:
        for race in races:
            if "error" in race:
                continue

            # ATG-berikelse: lagres alltid (uavhengig av om løpet er nytt)
            if race.get("horse_stats"):
                _save_horse_stats_atg(conn, race["horse_stats"])

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


def _known_race_ids(prefix: str | None = None) -> set:
    """Returnerer alle race_id som allerede ligger i DB (valgfri prefiks-filter)."""
    with get_conn() as conn:
        if prefix:
            rows = conn.execute(
                "SELECT race_id FROM races WHERE race_id LIKE ?",
                (f"{prefix}%",),
            ).fetchall()
        else:
            rows = conn.execute("SELECT race_id FROM races").fetchall()
    return {r["race_id"] for r in rows}


def _bulk_worker(
    date_from:    str,
    date_to:      str,
    countries:    list[str],
    delay:        float,
    sources:      list[str] | None = None,
    max_workers:  int = 5,
):
    """
    Henter historiske løp parallelt.

    Optimaliseringer:
      - HTTP keep-alive via requests.Session (~1.5x raskere)
      - Parallell henting med ThreadPoolExecutor (5x raskere)
      - Skip-if-exists: hopper over kjente race_id FØR HTTP-kallet
      - Batch DB-skriving når flere løp returneres samtidig

    `delay` brukes nå som "throttle mellom batcher" (ikke per-løp).
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

    def _bump(new_count: int):
        with _bulk_lock:
            _bulk_job["done"]    += 1
            _bulk_job["fetched"] += new_count
            if new_count == 0:
                _bulk_job["skipped"] += 1

    def _stopped() -> bool:
        with _bulk_lock:
            return _bulk_job["stop"]

    try:
        # ── Fase 1: oppdage stevner/løp og filtrer bort allerede-kjente ─────
        rikstoto_metas: list[dict] = []
        travsport_metas: list[dict] = []
        atg_ids: list[str] = []

        # 1a. Rikstoto-racedays (1 calendar-API-kall per 30 dager)
        if "rikstoto" in sources:
            cur = datetime.date.fromisoformat(date_from)
            end = datetime.date.fromisoformat(date_to)
            while cur <= end:
                if _stopped(): break
                batch_end = min(cur + datetime.timedelta(days=BATCH_DAYS - 1), end)
                rds = _scraper.get_racedays_for_period(
                    cur.isoformat(), batch_end.isoformat(),
                    countries=countries, only_finished=True,
                )
                rikstoto_metas.extend(rds)
                _log(f"🔍 Rikstoto {cur} → {batch_end}: {len(rds)} racedays")
                cur = batch_end + datetime.timedelta(days=1)

        # 1b. Travsport-stevner (1 calendar-scrape per måned)
        if "travsport" in sources:
            with _bulk_lock:
                _bulk_job["current"] = "Leser travsport.no-kalender…"
            travsport_metas = _scraper.get_travsport_result_urls(date_from, date_to)
            _log(f"🔍 Travsport {date_from} → {date_to}: {len(travsport_metas)} stevner")

        # 1c. ATG-løp (1 calendar-kall per dag)
        if "atg" in sources:
            with _bulk_lock:
                _bulk_job["current"] = "Leser ATG-kalender…"
            cur = datetime.date.fromisoformat(date_from)
            end = datetime.date.fromisoformat(date_to)
            while cur <= end:
                if _stopped(): break
                ids = _scraper.get_atg_race_ids(cur.isoformat())
                atg_ids.extend(ids)
                cur += datetime.timedelta(days=1)
            _log(f"🔍 ATG {date_from} → {date_to}: {len(atg_ids)} løp")

        # ── Skip-if-exists: bygg sett av kjente race_id én gang ─────────────
        known = _known_race_ids()

        # Rikstoto race_id = "rikstoto_{rdk}_{race_num}" — vi vet ikke race_num
        # før vi henter, så vi sjekker per raceday: hvis ALLE løp er kjent,
        # er hele raceday-en typisk komplett. Enkel heuristikk: sjekk om
        # "rikstoto_{rdk}_1" er kjent → hopp over hele raceday-en.
        rikstoto_metas_new = [m for m in rikstoto_metas
                              if f"rikstoto_{m['raceDay']}_1" not in known]
        rikstoto_skipped = len(rikstoto_metas) - len(rikstoto_metas_new)

        # Travsport: race_id = "travsport_{slug}_{date}_{race_num}"
        # Sjekk om første løp finnes
        travsport_metas_new = [m for m in travsport_metas
                               if f"travsport_{m['track_slug']}_{m['date']}_1" not in known]
        travsport_skipped = len(travsport_metas) - len(travsport_metas_new)

        # ATG: race_id = "atg_{atg_id}"
        atg_ids_new = [rid for rid in atg_ids if f"atg_{rid}" not in known]
        atg_skipped = len(atg_ids) - len(atg_ids_new)

        total_new = len(rikstoto_metas_new) + len(travsport_metas_new) + len(atg_ids_new)
        total_skipped = rikstoto_skipped + travsport_skipped + atg_skipped

        with _bulk_lock:
            _bulk_job["total"]   = total_new
            _bulk_job["skipped"] = total_skipped

        _log(f"⏭ Hopper over {total_skipped} allerede-hentede løp")
        _log(f"📋 {total_new} nye løp å hente — kjører {max_workers} parallelle workers")

        # ── Fase 2: parallell henting per kilde ─────────────────────────────

        def _progress(kind: str):
            def cb(done, total, payload):
                if _stopped():
                    return
                if isinstance(payload, list):
                    n_new = _save_races(payload)
                else:
                    n_new = _save_races([payload]) if payload else 0
                _bump(n_new)
                if done % 10 == 0 or done == total:
                    with _bulk_lock:
                        _bulk_job["current"] = f"{kind}: {done}/{total}"
                if n_new:
                    _log(f"✅ {kind} ({done}/{total}) +{n_new} løp")
            return cb

        # 2a. Rikstoto parallelt (4 workers — APIet tåler ikke aggressivt med)
        if rikstoto_metas_new and not _stopped():
            _scraper.fetch_rikstoto_racedays_parallel(
                rikstoto_metas_new,
                max_workers=min(max_workers, 4),
                progress_cb=_progress("Rikstoto"),
            )
            time.sleep(delay)

        # 2b. Travsport sekvensielt (HTML-scraping, server-throttling-fare)
        if travsport_metas_new and not _stopped():
            for m in travsport_metas_new:
                if _stopped(): break
                races = _scraper.fetch_travsport_raceday(m)
                n_new = _save_races(races) if races else 0
                _bump(n_new)
                with _bulk_lock:
                    _bulk_job["current"] = f"Travsport: {m['track_name']} {m['date']}"
                if n_new:
                    _log(f"✅ Travsport {m['track_name']} {m['date']} +{n_new}")
            time.sleep(delay)

        # 2c. ATG parallelt (5 workers er trygt)
        if atg_ids_new and not _stopped():
            _scraper.fetch_atg_races_parallel(
                atg_ids_new,
                max_workers=max_workers,
                progress_cb=_progress("ATG"),
            )

    except Exception as e:
        _log(f"❌ Uventet feil: {e}")
        with _bulk_lock:
            _bulk_job["errors"] += 1

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

    body        = request.get_json(force=True) or {}
    days        = min(int(body.get("days", 365)), 730)
    countries   = body.get("countries") or ["NO"]
    sources     = body.get("sources") or ["rikstoto"]
    delay       = float(body.get("delay", 0.3))           # default ned fra 1.0
    max_workers = min(int(body.get("max_workers", 5)), 10) # cap på 10 for høflighet

    today     = datetime.date.today()
    date_from = (today - datetime.timedelta(days=days)).isoformat()
    date_to   = today.isoformat()

    t = threading.Thread(
        target=_bulk_worker,
        args=(date_from, date_to, countries, delay, sources, max_workers),
        daemon=True,
    )
    t.start()

    return jsonify({"status": "started", "date_from": date_from,
                    "date_to": date_to, "countries": countries,
                    "sources": sources, "max_workers": max_workers})


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
    """
    Ranger et felt av hester med 9 vektede komponenter.

    Body:
      horses:          ["Hest A", "Hest B", ...]
      jockeys:         {"Hest A": "Kusk X", ...}        – valgfritt
      odds:            {"Hest A": 3.45, ...}            – valgfritt (pre-race odds)
      start_positions: {"Hest A": 4, ...}               – valgfritt (startnummer)
      extra_distances: {"Hest A": 20, ...}              – valgfritt (handicap-meter)
      trainers:        {"Hest A": "Trener Y", ...}      – valgfritt
      race_id:         "atg_2026-05-24_17_1"            – valgfritt (auto-fyller + lagrer)

    Hvis race_id er gitt, henter vi automatisk odds, start_positions,
    extra_distances og trainers fra databasen (results-tabellen).
    """
    body            = request.get_json(force=True)
    horses          = body.get("horses", [])
    jockeys         = body.get("jockeys", {})
    odds            = body.get("odds", {})
    start_positions = body.get("start_positions", {})
    extra_distances = body.get("extra_distances", {})
    trainers        = body.get("trainers", {})
    race_id         = body.get("race_id")

    if not horses:
        return jsonify({"error": "Tom hesteliste"}), 400

    # Auto-fyll fra databasen når race_id er gitt og verdier mangler
    if race_id:
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT horse_name, odds, start_pos, extra_distance, trainer
                FROM results WHERE race_id = ?
            """, (race_id,)).fetchall()
        for r in rows:
            n = r["horse_name"]
            if n not in odds            and r["odds"]:            odds[n]            = r["odds"]
            if n not in start_positions and r["start_pos"]:       start_positions[n] = r["start_pos"]
            if n not in extra_distances and r["extra_distance"] is not None:
                extra_distances[n] = r["extra_distance"]
            if n not in trainers        and r["trainer"]:         trainers[n]        = r["trainer"]

    ranked = ml_model.predict_field(
        horses, jockeys, odds,
        start_positions=start_positions,
        extra_distances=extra_distances,
        trainers=trainers,
    )
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


@app.route("/api/horse/auto_merge", methods=["POST"])
def api_horse_auto_merge():
    """Slår automatisk sammen alle hester med samme horse_reg_no."""
    return jsonify(analyzer.auto_merge_by_reg_no())


# ── API: Kommende løp + prediksjoner ─────────────────────────────────────────

@app.route("/api/upcoming")
def api_upcoming():
    """
    Lister kommende ATG-løp med pre-race odds og våre prediksjoner.

    Query-parametere:
      date_from: ISO-dato (default: i dag)
      date_to:   ISO-dato (default: i morgen)
      predict:   '1' for å auto-predikere alle (default: '1')
    """
    date_from = request.args.get("date_from", datetime.date.today().isoformat())
    date_to   = request.args.get("date_to",
                                  (datetime.date.today() + datetime.timedelta(days=1)).isoformat())
    do_predict = request.args.get("predict", "1") != "0"

    races = _scraper.fetch_atg_upcoming(date_from, date_to)
    if not races or (len(races) == 1 and "error" in races[0]):
        return jsonify({"races": [], "error": races[0].get("error") if races else "Ingen data"})

    # Lagre racene + horse-stats i DB (samme som vanlig henting),
    # slik at /api/race/<race_id> kan hente detaljer senere.
    _save_races(races)

    out = []
    for race in races:
        starters = race.get("results", [])
        race_block = {
            "race_id":    race["race_id"],
            "atg_id":     race["race_id"].replace("atg_", ""),
            "track":      race["track"],
            "date":       race["date"],
            "start_time": race.get("start_time"),
            "status":     race.get("status"),
            "distance":   race["distance"],
            "blood_type": race["blood_type"],
            "race_type":  race["race_type"],
            "n_starters": sum(1 for s in starters if not s.get("scratched")),
            "starters":   [],
            "prediction": [],
        }

        if do_predict:
            horses = [s["horse_name"] for s in starters if not s.get("scratched")]
            jockeys         = {s["horse_name"]: s["jockey"]         for s in starters}
            trainers        = {s["horse_name"]: s["trainer"]        for s in starters}
            odds            = {s["horse_name"]: s["odds"]           for s in starters if s["odds"]}
            start_positions = {s["horse_name"]: s["start_pos"]      for s in starters}
            extra_distances = {s["horse_name"]: s["extra_distance"] for s in starters}

            ranked = ml_model.predict_field(
                horses, jockeys, odds,
                start_positions=start_positions,
                extra_distances=extra_distances,
                trainers=trainers,
            )
            race_block["prediction"] = ranked
            predictor.save_predictions(race["race_id"], ranked)

        out.append(race_block)

    return jsonify({"races": out, "count": len(out),
                    "date_from": date_from, "date_to": date_to})


@app.route("/api/predictions/accuracy")
def api_prediction_accuracy():
    """Sammenligner lagrede prediksjoner mot faktiske resultater."""
    limit      = int(request.args.get("limit", 200))
    blood_type = request.args.get("blood_type", None)
    return jsonify(analyzer.prediction_accuracy(limit=limit, blood_type=blood_type))


@app.route("/api/horse/<name>/enrichment")
def api_horse_enrichment(name):
    """Returnerer ATG-berikelse for en hest (alder, kjønn, inntekter, PR)."""
    return jsonify(analyzer.horse_enrichment(name))


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
