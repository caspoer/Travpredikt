"""
Forbedret ML-modell for hesteloepsprediksjon.

Viktige forbedringer over v1:
  - Temporal treningsdata: stats beregnes kun fra loep FOER gjeldende dato
    (eliminerer data-lekkasje som ga train AUC 0.77 vs test AUC 0.59)
  - Bayesiansk smoothing: trekker win_rate mot prior (14%) for hester med faa loep
    (løser problemet med 0%/100% win_rate for hester med 1 loep)
  - Temporal train/test-split: test paa siste 20% av datoer, ikke tilfeldig
    (mer aerlg AUC-estimat for real-world bruk)
  - Sterkere regularisering: max_depth=2, min_samples_leaf=25
  - Fjernet win_vs_field som eneste feature (var 84% av alt)
  - Lagt til field_size og won_last som nye features

Features (12 totalt):
  win_rate_smooth   - Bayesiansk win-rate (stabil ved faa loep)
  top3_rate_smooth  - Bayesiansk top3-rate
  form_norm         - 1 - snittposisjon_siste5 / 10
  experience_norm   - min(antall_loep / 30, 1.0)
  jockey_wr         - jockey-vinnerprosent
  win_rank_norm     - rang i felt basert paa win_rate (1=best)
  top3_rank_norm    - rang i felt basert paa top3_rate (1=best)
  win_vs_field      - win_rate / feltsnitt (cap 3.0)
  has_data          - 1 hvis hesten finnes i DB
  field_size_norm   - feltets stoerrelse / 14
  won_last          - 1 hvis vant siste loep i DB
  time_rank_norm    - rang i felt basert paa beste km-tid (1=raskest, 0.5=ukjent)
"""
import os
import json
import datetime
import numpy as np
from collections import defaultdict
from database import get_conn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "horse_model.joblib")
META_PATH  = os.path.join(os.path.dirname(__file__), "horse_model_meta.json")

# ─── Eksplisitte prediksjonsvekter ───────────────────────────────────────────
# Justeres fritt – summen bør være 1.0.
# Anbefalt oppsett basert på forskning + profesjonell handicapping:
W_CAPACITY      = 0.35   # personlig rekord / hest-kapasitet (PR-tid)
W_FORM_TOP3     = 0.25   # topp-3-plasseringer siste 5 løp (recency)
W_TOP3_CAREER   = 0.20   # topp-3% karriere (konsistens – jevn hest)
W_WIN_TOTAL     = 0.10   # vunnede løp totalt (Bayesiansk win-rate)
W_JOCKEY        = 0.10   # jockey-vinnerprosent siste 40 løp

JOCKEY_RECENT_N = 40   # antall løp brukt til jockey-statistikk

# Prior: ~14% vinnersannsynlighet, ekvivalent med 10 "fantom-loep"
PRIOR_WIN_RATE  = 0.138
PRIOR_TOP3_RATE = 0.414
PRIOR_N         = 10.0
ALPHA_WIN  = PRIOR_WIN_RATE  * PRIOR_N   # 1.38
BETA_WIN   = (1 - PRIOR_WIN_RATE)  * PRIOR_N   # 8.62
ALPHA_TOP3 = PRIOR_TOP3_RATE * PRIOR_N   # 4.14
BETA_TOP3  = (1 - PRIOR_TOP3_RATE) * PRIOR_N   # 5.86

FEATURE_COLS = [
    "win_rate_smooth", "top3_rate_smooth", "form_norm", "experience_norm",
    "jockey_wr", "win_rank_norm", "top3_rank_norm",
    "win_vs_field", "has_data", "field_size_norm", "won_last",
    "time_rank_norm",
]


# ─── Eksplisitte prediksjonsvekter ────────────────────────────────────────────
# Forskningsforankret oppsett basert på Benter (1994), BetMix-data,
# EquinEdge handicapping og brukerens egne justeringer.
W_MARKET        = 0.22   # pre-race vinnerodds (markedskonsensus)
W_CLASS         = 0.18   # klasse: livstidsinntekt per start (ATG)
W_CAPACITY      = 0.25   # personlig rekord (km-tid)
W_FORM_BEST2OF3 = 0.12   # form: best 2-av-3 siste løp (filtrerer dud-løp)
W_JOCKEY        = 0.08   # jockey-vinnerprosent siste 40 løp
W_POST_POS      = 0.06   # startspor (inside-fordel, trav)
W_TOP3_CAREER   = 0.04   # konsistens (top3% karriere)
W_HANDICAP      = 0.03   # extra_distance som klasse-rating (kaldblod)
W_TRAINER       = 0.02   # trener-vinnerprosent
# Sum: 0.22+0.18+0.25+0.12+0.08+0.06+0.04+0.03+0.02 = 1.00

# Favoritt-langskudd-korrigering (Benter-stil shrinkage)
# Markedssannsynlighet løftes svakt opp på favoritter, ned på langskudd.
ODDS_SHRINKAGE_EXP = 0.94

# Trener-stats: minst så mange løp før vi stoler på vinnerprosenten
TRAINER_MIN_RACES = 10


# ── Bulk-statistikk (brukes til prediksjon) ───────────────────────────────────

def _precompute_horse_stats(conn) -> dict:
    """Beregner aggregert statistikk for alle hester med Bayesiansk smoothing."""
    rows = conn.execute("""
        SELECT LOWER(r.horse_name) AS name,
               COUNT(*) AS races,
               SUM(CASE WHEN r.position = 1 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN r.position <= 3 THEN 1 ELSE 0 END) AS top3
        FROM results r
        WHERE r.position IS NOT NULL
        GROUP BY LOWER(r.horse_name)
    """).fetchall()

    stats = {}
    for r in rows:
        races = r["races"]
        wins  = r["wins"]
        top3  = r["top3"]
        stats[r["name"]] = {
            "has_data":        1,
            "races":           races,          # faktisk loepstall (ikke normalisert)
            "win_rate_smooth":  (wins  + ALPHA_WIN)  / (races + ALPHA_WIN  + BETA_WIN),
            "top3_rate_smooth": (top3  + ALPHA_TOP3) / (races + ALPHA_TOP3 + BETA_TOP3),
            "experience_norm":  min(races / 30.0, 1.0),
            "avg_pos_5":        5.0,
            "won_last":         0,
        }

    # Siste-5-form og siste vinner via window-funksjon (SQLite 3.25+)
    try:
        pos_rows = conn.execute("""
            WITH ranked AS (
                SELECT LOWER(r.horse_name) AS name,
                       r.position,
                       ROW_NUMBER() OVER (
                           PARTITION BY LOWER(r.horse_name)
                           ORDER BY rc.date DESC, r.id DESC
                       ) AS rn
                FROM results r JOIN races rc ON r.race_id = rc.race_id
                WHERE r.position IS NOT NULL
            )
            SELECT name, position, rn FROM ranked WHERE rn <= 5
        """).fetchall()

        last5  = defaultdict(list)
        for r in pos_rows:
            last5[r["name"]].append((r["rn"], r["position"]))
        for name, ps in last5.items():
            if name in stats:
                ps_sorted = [p for _, p in sorted(ps)]
                stats[name]["avg_pos_5"]     = sum(ps_sorted) / len(ps_sorted)
                stats[name]["won_last"]      = 1 if ps_sorted[0] == 1 else 0
                stats[name]["top3_recent_5"] = sum(1 for p in ps_sorted if p <= 3) / len(ps_sorted)

                # Best 2-av-3: ta siste 3 løp, kast dårligste, snitt resterende 2
                # Filtrerer ut "dud"-løp (sykdom, dårlig pace, feilstart)
                last3 = ps_sorted[:3]
                if len(last3) >= 3:
                    best2 = sorted(last3)[:2]   # to laveste posisjoner = best
                    stats[name]["best2of3_pos"] = sum(best2) / 2.0
                elif len(last3) >= 1:
                    stats[name]["best2of3_pos"] = sum(last3) / len(last3)
    except Exception:
        pass

    # Beste km-tid per hest (distanse-normalisert personlig rekord)
    try:
        time_rows = conn.execute("""
            SELECT LOWER(r.horse_name) AS name,
                   MIN(r.time_sec / (rc.distance / 1000.0)) AS best_km_time
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE r.time_sec IS NOT NULL AND r.time_sec > 0
              AND rc.distance IS NOT NULL AND rc.distance >= 500
            GROUP BY LOWER(r.horse_name)
        """).fetchall()
        for row in time_rows:
            if row["name"] in stats:
                stats[row["name"]]["best_km_time"] = float(row["best_km_time"])
    except Exception:
        pass

    # ATG-berikelse: PR-tid og livstidsstatistikk fra horse_stats_atg-tabellen.
    # Disse overrider/supplerer egne tall siden ATG ofte har mer komplette data
    # (livstid, ikke bare det vi har scrapet).
    try:
        enrich_rows = conn.execute("""
            SELECT LOWER(name)     AS name,
                   best_time_sec,
                   life_win_pct,
                   life_place_pct,
                   life_starts,
                   age,
                   money
            FROM horse_stats_atg
            WHERE name IS NOT NULL AND name != ''
        """).fetchall()
        for row in enrich_rows:
            key = row["name"]
            if key not in stats:
                # Hesten finnes i ATG-berikelse men ikke i egne resultater ennå
                # → opprett basis-record så vi får nytte av berikelsen
                stats[key] = {
                    "has_data": 1,
                    "races":    row["life_starts"] or 0,
                    "win_rate_smooth":  PRIOR_WIN_RATE,
                    "top3_rate_smooth": PRIOR_TOP3_RATE,
                    "experience_norm":  min((row["life_starts"] or 0) / 30.0, 1.0),
                    "avg_pos_5": 5.0,
                    "won_last":  0,
                }

            # Beste km-tid: bruk ATG-verdien hvis den finnes (mer pålitelig)
            if row["best_time_sec"]:
                cur = stats[key].get("best_km_time")
                atg_kmt = float(row["best_time_sec"])
                if cur is None or atg_kmt < cur:
                    stats[key]["best_km_time"] = atg_kmt

            # Livstids-statistikk: oppdater hvis vi har færre data-punkter
            life_n = row["life_starts"] or 0
            if life_n > stats[key].get("races", 0):
                wins_life = round((row["life_win_pct"] or 0) / 100.0 * life_n)
                top3_life = round((row["life_place_pct"] or 0) / 100.0 * life_n)
                stats[key]["win_rate_smooth"]  = (wins_life + ALPHA_WIN)  / (life_n + ALPHA_WIN  + BETA_WIN)
                stats[key]["top3_rate_smooth"] = (top3_life + ALPHA_TOP3) / (life_n + ALPHA_TOP3 + BETA_TOP3)
                stats[key]["experience_norm"]  = min(life_n / 30.0, 1.0)
                stats[key]["races"]            = life_n

            # Berikelses-info for visning
            stats[key]["atg_age"]   = row["age"]
            stats[key]["atg_money"] = row["money"]

            # Klasse-indikator: inntekt per start (klassenivå)
            # Forskningens enkelt-sterkeste fundamental ("avg money per race").
            # Bruker hele livstid for stabilitet; måles i øre per start.
            if life_n > 0 and row["money"]:
                stats[key]["earnings_per_start"] = row["money"] / life_n
    except Exception:
        pass

    # Trener-mapping: siste trener per hest (for trener-stats-oppslag)
    try:
        tr_rows = conn.execute("""
            SELECT LOWER(r.horse_name) AS name,
                   r.trainer           AS trainer_orig,
                   LOWER(r.trainer)    AS trainer_key
            FROM results r JOIN races rc ON r.race_id = rc.race_id
            WHERE r.trainer IS NOT NULL AND r.trainer != ''
            ORDER BY rc.date DESC, r.id DESC
        """).fetchall()
        seen: set = set()
        for r in tr_rows:
            if r["name"] not in seen:
                seen.add(r["name"])
                if r["name"] in stats:
                    stats[r["name"]]["last_trainer"]      = r["trainer_key"]
                    stats[r["name"]]["last_trainer_disp"] = r["trainer_orig"]
    except Exception:
        pass

    # Siste jockey per hest (original stor/liten bokstav + lowercase-noekkel)
    try:
        jk_rows = conn.execute("""
            SELECT LOWER(r.horse_name) AS name,
                   r.jockey           AS jockey_orig,
                   LOWER(r.jockey)    AS jockey_key
            FROM results r JOIN races rc ON r.race_id = rc.race_id
            WHERE r.jockey IS NOT NULL AND r.jockey != ''
            ORDER BY rc.date DESC, r.id DESC
        """).fetchall()
        seen: set = set()
        for r in jk_rows:
            if r["name"] not in seen:
                seen.add(r["name"])
                if r["name"] in stats:
                    stats[r["name"]]["last_jockey"]      = r["jockey_key"]   # for oppslag
                    stats[r["name"]]["last_jockey_disp"] = r["jockey_orig"]  # for visning
    except Exception:
        pass

    return stats


def _precompute_jockey_stats(conn) -> dict:
    """Vinnerprosent per jockey med Bayesiansk smoothing (min 3 loep)."""
    rows = conn.execute("""
        SELECT LOWER(jockey) AS j,
               COUNT(*) AS races,
               SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) AS wins
        FROM results
        WHERE position IS NOT NULL
          AND jockey IS NOT NULL AND jockey != ''
        GROUP BY LOWER(jockey)
        HAVING races >= 3
    """).fetchall()
    return {r["j"]: (r["wins"] + ALPHA_WIN) / (r["races"] + ALPHA_WIN + BETA_WIN)
            for r in rows}


def _precompute_trainer_stats(conn) -> dict:
    """Vinnerprosent per trener med Bayesiansk smoothing (min TRAINER_MIN_RACES løp)."""
    rows = conn.execute("""
        SELECT LOWER(trainer) AS t,
               COUNT(*) AS races,
               SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) AS wins
        FROM results
        WHERE position IS NOT NULL
          AND trainer IS NOT NULL AND trainer != ''
        GROUP BY LOWER(trainer)
        HAVING races >= ?
    """, (TRAINER_MIN_RACES,)).fetchall()
    return {r["t"]: (r["wins"] + ALPHA_WIN) / (r["races"] + ALPHA_WIN + BETA_WIN)
            for r in rows}


def _precompute_jockey_recent_stats(conn, n: int = JOCKEY_RECENT_N) -> dict:
    """Vinnerprosent per jockey beregnet fra de siste N løpene (Bayesiansk smoothing)."""
    rows = conn.execute("""
        WITH ranked AS (
            SELECT LOWER(r.jockey) AS j,
                   r.position,
                   ROW_NUMBER() OVER (
                       PARTITION BY LOWER(r.jockey)
                       ORDER BY rc.date DESC, r.id DESC
                   ) AS rn
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE r.position IS NOT NULL
              AND r.jockey IS NOT NULL AND r.jockey != ''
        )
        SELECT j,
               COUNT(*)                                          AS races,
               SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END)   AS wins
        FROM ranked
        WHERE rn <= ?
        GROUP BY j
        HAVING races >= 5
    """, (n,)).fetchall()
    return {
        r["j"]: (r["wins"] + ALPHA_WIN) / (r["races"] + ALPHA_WIN + BETA_WIN)
        for r in rows
    }


# ── Feature-matrise ───────────────────────────────────────────────────────────

def _build_feature_matrix(
    horse_names: list,
    horse_stats: dict,
    jockey_stats: dict,
    race_jockeys: dict | None = None,
) -> np.ndarray:
    """Returnerer (n_hester x 11) feature-matrise for ett felt."""
    if race_jockeys is None:
        race_jockeys = {}

    n   = len(horse_names)
    raw = []

    for name in horse_names:
        key = name.lower()
        s   = horse_stats.get(key, {})
        jk  = (race_jockeys.get(name) or s.get("last_jockey") or "").lower()
        jwr = jockey_stats.get(jk, PRIOR_WIN_RATE)

        raw.append({
            "win_rate_smooth":  s.get("win_rate_smooth",  PRIOR_WIN_RATE),
            "top3_rate_smooth": s.get("top3_rate_smooth", PRIOR_TOP3_RATE),
            "form_norm":        1.0 - min(s.get("avg_pos_5", 5.0), 10.0) / 10.0,
            "experience_norm":  s.get("experience_norm", 0.0),
            "jockey_wr":        jwr,
            "has_data":         float(s.get("has_data", 0)),
            "won_last":         float(s.get("won_last", 0)),
            "best_km_time":     s.get("best_km_time"),   # None = ingen data
        })

    wrs      = [r["win_rate_smooth"]  for r in raw]
    t3s      = [r["top3_rate_smooth"] for r in raw]
    field_wr = max(sum(wrs) / n, 1e-6)
    field_t3 = max(sum(t3s) / n, 1e-6)

    wr_order = sorted(range(n), key=lambda i: wrs[i], reverse=True)
    t3_order = sorted(range(n), key=lambda i: t3s[i], reverse=True)
    wr_rank  = [0.0] * n
    t3_rank  = [0.0] * n
    div = max(n - 1, 1)
    for rank, idx in enumerate(wr_order):
        wr_rank[idx] = 1.0 - rank / div
    for rank, idx in enumerate(t3_order):
        t3_rank[idx] = 1.0 - rank / div

    # Rangering etter beste km-tid (lavere = raskere = bedre rang).
    # Hester uten tidsdata får nøytral verdi 0.5.
    time_rank = [0.5] * n
    timed = [(i, r["best_km_time"]) for i, r in enumerate(raw) if r["best_km_time"] is not None]
    if len(timed) >= 2:
        timed_sorted = sorted(timed, key=lambda x: x[1])   # lavest tid = raskest = rang 1
        t_div = max(len(timed) - 1, 1)
        for t_rank, (idx, _) in enumerate(timed_sorted):
            time_rank[idx] = 1.0 - t_rank / t_div
    elif len(timed) == 1:
        time_rank[timed[0][0]] = 1.0   # eneste med tid → best i feltet

    X = []
    for i, r in enumerate(raw):
        X.append([
            r["win_rate_smooth"],
            r["top3_rate_smooth"],
            r["form_norm"],
            r["experience_norm"],
            r["jockey_wr"],
            wr_rank[i],
            t3_rank[i],
            min(r["win_rate_smooth"] / field_wr, 3.0),   # cap ved 3x
            r["has_data"],
            n / 14.0,                                     # felt-stoerrelse norm
            r["won_last"],
            time_rank[i],                                 # PR-rang i feltet
        ])

    return np.nan_to_num(np.array(X, dtype=np.float32), nan=0.0)


# ── Temporal treningsdata (lekkasje-fri) ──────────────────────────────────────

def build_training_data() -> tuple:
    """
    Lekkasje-fri treningsdata med temporal ordering.

    For hvert loep beregnes hestestatistikk KUN fra loep som fant sted
    FOER gjeldende loep (kronologisk rekkefoelge). Dette simulerer ekte
    prediksjonssituasjon og eliminerer den sirkulaere data-lekkasjen.
    """
    with get_conn() as conn:
        jockey_stats = _precompute_jockey_stats(conn)

        all_rows = conn.execute("""
            SELECT r.race_id, r.horse_name, r.position, r.jockey,
                   r.time_sec, rc.date, rc.distance
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE r.position IS NOT NULL
            ORDER BY rc.date, r.race_id
        """).fetchall()

    by_race = defaultdict(list)
    race_date = {}
    for row in all_rows:
        by_race[row["race_id"]].append(row)
        race_date[row["race_id"]] = row["date"]

    # Kumulativ statistikk per hest (oppdateres etter hvert loep)
    h_races    = defaultdict(int)
    h_wins     = defaultdict(int)
    h_top3     = defaultdict(int)
    h_recent   = defaultdict(list)   # siste 5 posisjoner
    h_best_kmt: dict = {}            # beste km-tid (sek/km) – lavere = raskere

    sorted_races = sorted(by_race.items(), key=lambda x: race_date[x[0]])

    X_all, y_all, dates_all = [], [], []

    def _update_kmt(nm, time_sec, distance):
        """Oppdaterer beste km-tid for en hest hvis den er bedre enn lagret."""
        if time_sec and time_sec > 0 and distance and distance >= 500:
            kmt = time_sec / (distance / 1000.0)
            if nm not in h_best_kmt or kmt < h_best_kmt[nm]:
                h_best_kmt[nm] = kmt

    for race_id, starters in sorted_races:
        if len(starters) < 3:
            # Oppdater loepende totaler selv for utelatte loep
            for s in starters:
                nm = s["horse_name"].lower()
                h_races[nm] += 1
                if s["position"] == 1:
                    h_wins[nm] += 1
                if s["position"] <= 3:
                    h_top3[nm] += 1
                h_recent[nm] = (h_recent[nm] + [s["position"]])[-5:]
                _update_kmt(nm, s["time_sec"], s["distance"])
            continue

        y_race = [1 if s["position"] == 1 else 0 for s in starters]
        if sum(y_race) != 1:
            for s in starters:
                nm = s["horse_name"].lower()
                h_races[nm] += 1
                if s["position"] == 1:
                    h_wins[nm] += 1
                if s["position"] <= 3:
                    h_top3[nm] += 1
                h_recent[nm] = (h_recent[nm] + [s["position"]])[-5:]
                _update_kmt(nm, s["time_sec"], s["distance"])
            continue

        # Bygg snapshot av statistikk FOER dette loepet
        snap = {}
        for s in starters:
            nm     = s["horse_name"].lower()
            races  = h_races[nm]
            wins   = h_wins[nm]
            top3   = h_top3[nm]
            recent = h_recent[nm]

            snap[s["horse_name"]] = {
                "has_data":        1 if races > 0 else 0,
                "win_rate_smooth":  (wins  + ALPHA_WIN)  / (races + ALPHA_WIN  + BETA_WIN),
                "top3_rate_smooth": (top3  + ALPHA_TOP3) / (races + ALPHA_TOP3 + BETA_TOP3),
                "experience_norm":  min(races / 30.0, 1.0),
                "avg_pos_5":        sum(recent) / len(recent) if recent else 5.0,
                "won_last":         1.0 if (recent and recent[-1] == 1) else 0.0,
                "last_jockey":      "",
                "best_km_time":     h_best_kmt.get(nm),   # None = ingen data ennaa
            }

        names = [s["horse_name"] for s in starters]
        jocks = {s["horse_name"]: (s["jockey"] or "") for s in starters}
        X_race = _build_feature_matrix(names, snap, jockey_stats, jocks)
        X_all.extend(X_race.tolist())
        y_all.extend(y_race)
        dates_all.extend([race_date[race_id]] * len(starters))

        # Oppdater loepende totaler ETTER at vi har brukt dem
        for s in starters:
            nm = s["horse_name"].lower()
            h_races[nm] += 1
            if s["position"] == 1:
                h_wins[nm] += 1
            if s["position"] <= 3:
                h_top3[nm] += 1
            h_recent[nm] = (h_recent[nm] + [s["position"]])[-5:]
            _update_kmt(nm, s["time_sec"], s["distance"])

    return (np.array(X_all, dtype=np.float32),
            np.array(y_all, dtype=np.int32),
            dates_all)


# ── Trening ───────────────────────────────────────────────────────────────────

def train(log_fn=None) -> dict:
    """
    Trener og lagrer modellen.
    Bruker temporal train/test-split: test = siste 20% av datoer.
    """
    import sklearn
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import joblib

    def _log(msg):
        if log_fn:
            log_fn(msg)

    _log("Bygger temporal treningsdata (lekkasje-fri)...")
    X, y, dates = build_training_data()

    n_samples = len(X)
    n_races   = int(y.sum())
    _log(f"  {n_samples} starter-rader  |  {n_races} vinnere")

    if n_samples < 50:
        raise ValueError(
            f"For lite data: {n_samples} rader. "
            "Last inn minst noen loep i databasen foerst."
        )

    # Temporal split: test paa siste 20% av datoer
    sorted_dates = sorted(set(dates))
    split_idx    = int(len(sorted_dates) * 0.80)
    split_date   = sorted_dates[split_idx] if split_idx < len(sorted_dates) else sorted_dates[-1]

    train_mask = np.array([d < split_date for d in dates])
    test_mask  = ~train_mask

    _log(f"  Trening: dato < {split_date}  ({train_mask.sum()} rader)")
    _log(f"  Test:    dato >= {split_date}  ({test_mask.sum()} rader)")

    if train_mask.sum() < 30 or test_mask.sum() < 10:
        _log("  For lite data for temporal split - bruker 80/20 tilfeldig")
        np.random.seed(42)
        perm = np.random.permutation(len(y))
        split = int(0.8 * len(y))
        train_mask = np.zeros(len(y), dtype=bool)
        train_mask[perm[:split]] = True
        test_mask = ~train_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    # ── Modell 1: GradientBoosting ────────────────────────────────────────────
    _log("Trener GradientBoostingClassifier...")
    gbm = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=2,
        subsample=0.7,
        min_samples_leaf=25,
        random_state=42,
    )
    gbm.fit(X_tr, y_tr)
    gbm_train_auc = roc_auc_score(y_tr, gbm.predict_proba(X_tr)[:, 1])
    gbm_test_auc  = roc_auc_score(y_te, gbm.predict_proba(X_te)[:, 1])
    _log(f"  GBM   train={gbm_train_auc:.3f}  test={gbm_test_auc:.3f}  "
         f"gap={gbm_train_auc - gbm_test_auc:.3f}")

    # ── Modell 2: Logistisk regresjon (referanse, ingen overfitting) ──────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    lr = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    lr.fit(X_tr_s, y_tr)
    lr_train_auc = roc_auc_score(y_tr, lr.predict_proba(X_tr_s)[:, 1])
    lr_test_auc  = roc_auc_score(y_te, lr.predict_proba(X_te_s)[:, 1])
    _log(f"  LR    train={lr_train_auc:.3f}  test={lr_test_auc:.3f}  "
         f"gap={lr_train_auc - lr_test_auc:.3f}")

    # Velg modell med best test-AUC
    if gbm_test_auc >= lr_test_auc:
        best_model  = gbm
        best_name   = "GradientBoosting"
        train_auc   = gbm_train_auc
        test_auc    = gbm_test_auc
        need_scaler = False
        _log(f"Velger GradientBoosting (test-AUC {test_auc:.3f})")
    else:
        best_model  = lr
        best_name   = "LogisticRegression"
        train_auc   = lr_train_auc
        test_auc    = lr_test_auc
        need_scaler = True
        _log(f"Velger LogisticRegression (test-AUC {test_auc:.3f})")

    # Feature importance / koeffisienter
    if hasattr(best_model, "feature_importances_"):
        imp_vals = best_model.feature_importances_
    else:
        imp_vals = np.abs(best_model.coef_[0])
        imp_vals = imp_vals / imp_vals.sum()

    importances = sorted(zip(FEATURE_COLS, imp_vals), key=lambda x: x[1], reverse=True)
    _log("Viktigste features: " +
         "  ".join(f"{n}={v:.3f}" for n, v in importances[:5]))

    # Lagre modell (+ scaler hvis nodvendig)
    payload = {"model": best_model, "scaler": scaler if need_scaler else None,
               "needs_scale": need_scaler}
    joblib.dump(payload, MODEL_PATH)

    meta = {
        "trained_at":      datetime.datetime.now().isoformat(),
        "model_type":      best_name,
        "split_date":      split_date,
        "n_samples":       n_samples,
        "n_races":         n_races,
        "train_auc":       round(float(train_auc), 3),
        "test_auc":        round(float(test_auc), 3),
        "gbm_test_auc":    round(float(gbm_test_auc), 3),
        "lr_test_auc":     round(float(lr_test_auc), 3),
        "feature_names":   FEATURE_COLS,
        "importances":     {n: round(float(v), 4) for n, v in importances},
        "sklearn_version": sklearn.__version__,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    gap = train_auc - test_auc
    _log(f"Ferdig  |  test-AUC={test_auc:.3f}  gap={gap:.3f}"
         + ("  (god regularisering)" if gap < 0.05 else
            "  (noe overfitting)" if gap < 0.10 else
            "  (ADVARSEL: overfitting)"))
    return meta


# ── Status ────────────────────────────────────────────────────────────────────

def is_trained() -> bool:
    return os.path.exists(MODEL_PATH)


def get_meta() -> dict:
    if not os.path.exists(META_PATH):
        return {}
    try:
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Prediksjon ────────────────────────────────────────────────────────────────

def predict_field(
    horses:          list,
    jockeys:         dict | None = None,
    odds:            dict | None = None,
    start_positions: dict | None = None,
    extra_distances: dict | None = None,
    trainers:        dict | None = None,
) -> list:
    """
    Ranger et felt med 9 vektede komponenter:

      | # | Komponent              | Vekt |
      |---|------------------------|------|
      | 1 | PR-tid (kapasitet)     | 25%  |
      | 2 | Markedsodds            | 22%  |
      | 3 | Klasse (inntekt/start) | 18%  |
      | 4 | Form (best 2-av-3)     | 12%  |
      | 5 | Kusk siste 40 løp      | 8%   |
      | 6 | Startspor (inside)     | 6%   |
      | 7 | Konsistens (top3%)     | 4%   |
      | 8 | Handicap-meter         | 3%   |
      | 9 | Trener (vinn%)         | 2%   |
      |   | Sum                    | 100% |

    Markedsodds gjennomgår favoritt-langskudd-korrigering (Benter-shrinkage):
    odds-implisitte sannsynligheter løftes svakt opp på favoritter,
    ned på langskudd (empirisk konsistent med 30 års forskning).

    Hester uten data for en komponent får nøytral rang 0.5.
    """
    if not horses:
        return []
    jockeys         = jockeys         or {}
    odds            = odds            or {}
    start_positions = start_positions or {}
    extra_distances = extra_distances or {}
    trainers        = trainers        or {}

    with get_conn() as conn:
        horse_stats    = _precompute_horse_stats(conn)
        jockey_stats   = _precompute_jockey_stats(conn)
        jockey_recent  = _precompute_jockey_recent_stats(conn)
        trainer_stats  = _precompute_trainer_stats(conn)

    X       = _build_feature_matrix(horses, horse_stats, jockey_stats, jockeys)
    used_ml = False

    n = len(horses)

    # ── Hjelper: rang innen felt [min_rank = dårligst, 1.0 = best] ──────────
    def _ranks(values, higher_is_better=True, min_rank=0.15) -> list:
        idx_sorted = sorted(range(n), key=lambda i: values[i], reverse=higher_is_better)
        out  = [0.0] * n
        div  = max(n - 1, 1)
        span = 1.0 - min_rank
        for rank, idx in enumerate(idx_sorted):
            out[idx] = 1.0 - rank / div * span
        return out

    # Variant som ignorerer manglende verdier (None) og gir dem 0.5
    def _ranks_partial(values, higher_is_better=True, min_rank=0.15) -> list:
        out = [0.5] * n
        with_vals = [(i, v) for i, v in enumerate(values) if v is not None]
        if len(with_vals) < 2:
            if len(with_vals) == 1:
                out[with_vals[0][0]] = 1.0
            return out
        with_vals.sort(key=lambda x: x[1], reverse=higher_is_better)
        div  = max(len(with_vals) - 1, 1)
        span = 1.0 - min_rank
        for rank, (idx, _) in enumerate(with_vals):
            out[idx] = 1.0 - rank / div * span
        return out

    # ── 1 (25%) Kapasitet: personlig rekord (km-tid) ────────────────────────
    time_idx = FEATURE_COLS.index("time_rank_norm")
    wr_idx   = FEATURE_COLS.index("win_rank_norm")
    t3r_idx  = FEATURE_COLS.index("top3_rank_norm")
    capacity = []
    for i, name in enumerate(horses):
        s = horse_stats.get(name.lower(), {})
        if s.get("best_km_time") is not None:
            capacity.append(float(X[i, time_idx]))
        else:
            capacity.append((float(X[i, wr_idx]) + float(X[i, t3r_idx])) / 2.0)
    cap_ranks = _ranks(capacity)

    # ── 2 (22%) Marked: pre-race vinnerodds (med Benter-shrinkage) ──────────
    # Konverterer odds til implisitt sannsynlighet, anvender shrinkage,
    # og rangerer deretter. Shrinkage 0.94 = svak favoritt-bonus.
    odds_vals = [odds.get(name) for name in horses]
    market_ranks = [0.5] * n
    odds_indexed = [(i, o) for i, o in enumerate(odds_vals)
                    if o is not None and 0 < o < 9999]
    if len(odds_indexed) >= 2:
        # Implisitt p = 1/odds, så shrinkage: p' = p^0.94 (forsterker favoritter)
        # Sortering på shrinket sannsynlighet = sortering på odds (monotont),
        # men shrinkage får betydning hvis vi senere bruker som probability direkte.
        # For ranking: lavest odds = beste rang.
        odds_sorted = sorted(odds_indexed, key=lambda x: x[1])
        n_with_odds = len(odds_sorted)
        div  = max(n_with_odds - 1, 1)
        span = 1.0 - 0.15
        for rank, (idx, _) in enumerate(odds_sorted):
            base_rank = 1.0 - rank / div * span
            # Shrinkage: løft favoritter (høy base_rank) litt opp via potensiering
            market_ranks[idx] = base_rank ** ODDS_SHRINKAGE_EXP
    elif len(odds_indexed) == 1:
        market_ranks[odds_indexed[0][0]] = 1.0

    # ── 3 (18%) Klasse: livstidsinntekt per start (ATG-berikelse) ───────────
    # Forskningens enkelt-sterkeste fundamental.
    class_vals = [horse_stats.get(name.lower(), {}).get("earnings_per_start")
                  for name in horses]
    class_ranks = _ranks_partial(class_vals, higher_is_better=True)

    # ── 4 (12%) Form: best 2-av-3 siste løp ─────────────────────────────────
    # Lavere snittposisjon (av de 2 beste av siste 3) = bedre form.
    form_vals = [horse_stats.get(name.lower(), {}).get("best2of3_pos")
                 for name in horses]
    # Inverter: lavere posisjon = bedre = høyere "form"-verdi
    form_inv = [(-v if v is not None else None) for v in form_vals]
    form_ranks = _ranks_partial(form_inv, higher_is_better=True)

    # ── 5 (8%) Kusk: vinnerprosent siste 40 løp ─────────────────────────────
    jwr_recent_vals = []
    for name in horses:
        s   = horse_stats.get(name.lower(), {})
        jk  = (jockeys.get(name) or s.get("last_jockey") or "").lower()
        val = jockey_recent.get(jk) or jockey_stats.get(jk, PRIOR_WIN_RATE)
        jwr_recent_vals.append(val)
    jockey_ranks = _ranks(jwr_recent_vals)

    # ── 6 (6%) Startspor: inside-fordel (trav) ─────────────────────────────
    # Lavt startnummer = bedre. Logaritmisk avtakende effekt: spor 1 mye bedre
    # enn 4, spor 4 og 7 er forholdsvis like.
    sp_vals = []
    for name in horses:
        sp = start_positions.get(name)
        if sp is None or sp <= 0:
            sp_vals.append(None)
        else:
            # Score: 1/log(sp + 1) gir 1.44 for sp=1, 0.91 for sp=2, ..., 0.40 for sp=8
            sp_vals.append(1.0 / np.log(sp + 1))
    post_ranks = _ranks_partial(sp_vals, higher_is_better=True)

    # ── 7 (4%) Konsistens: topp-3% karriere ─────────────────────────────────
    top3_career_vals = [horse_stats.get(name.lower(), {}).get("top3_rate_smooth", PRIOR_TOP3_RATE)
                        for name in horses]
    top3_career_ranks = _ranks(top3_career_vals)

    # ── 8 (3%) Handicap-meter: extra_distance som klasse-indikator ──────────
    # Større handicap = sterkere hest (gitt av offisiell handikapper).
    handicap_vals = []
    for name in horses:
        xd = extra_distances.get(name)
        # 0 meter er nøytralt (varmblod eller laveste klasse); kun ulikt 0 betyr noe
        handicap_vals.append(xd if (xd is not None and xd > 0) else None)
    handicap_ranks = _ranks_partial(handicap_vals, higher_is_better=True)

    # ── 9 (2%) Trener: vinnerprosent (Bayesiansk smoothed) ──────────────────
    tr_vals = []
    for name in horses:
        s   = horse_stats.get(name.lower(), {})
        tr  = (trainers.get(name) or s.get("last_trainer") or "").lower()
        tr_vals.append(trainer_stats.get(tr, PRIOR_WIN_RATE) if tr else PRIOR_WIN_RATE)
    trainer_ranks = _ranks(tr_vals)

    # ── Endelig score med eksplisitte vekter ─────────────────────────────────
    raw_scores = np.array([
        W_CAPACITY      * cap_ranks[i]         +
        W_MARKET        * market_ranks[i]      +
        W_CLASS         * class_ranks[i]       +
        W_FORM_BEST2OF3 * form_ranks[i]        +
        W_JOCKEY        * jockey_ranks[i]      +
        W_POST_POS      * post_ranks[i]        +
        W_TOP3_CAREER   * top3_career_ranks[i] +
        W_HANDICAP      * handicap_ranks[i]    +
        W_TRAINER       * trainer_ranks[i]
        for i in range(n)
    ], dtype=np.float64)

    total = float(raw_scores.sum())
    probs = raw_scores / total if total > 0 else np.ones(n) / n
    raw_probs = raw_scores

    out = []
    for i, name in enumerate(horses):
        key = name.lower()
        s   = horse_stats.get(key, {})

        jockey_provided = jockeys.get(name, "")
        jk_key          = (jockey_provided or s.get("last_jockey") or "").lower()
        jockey_disp     = (jockey_provided
                           or s.get("last_jockey_disp")
                           or jk_key.title())

        trainer_provided = trainers.get(name, "")
        trainer_disp = (trainer_provided
                        or s.get("last_trainer_disp")
                        or "")

        eps = s.get("earnings_per_start")
        out.append({
            "horse":           name,
            "win_prob":        round(float(probs[i]) * 100, 1),
            "score":           round(float(raw_probs[i]), 4),
            # Komponentskårer (alle som % 0–100)
            "cap_rank":        round(cap_ranks[i]         * 100, 1),   # 25%
            "market_rank":     round(market_ranks[i]      * 100, 1),   # 22%
            "class_rank":      round(class_ranks[i]       * 100, 1),   # 18%
            "form_rank":       round(form_ranks[i]        * 100, 1),   # 12%
            "jockey_wr":       round(jwr_recent_vals[i]   * 100, 1),   # 8%
            "post_rank":       round(post_ranks[i]        * 100, 1),   # 6%
            "top3_career":     round(top3_career_vals[i]  * 100, 1),   # 4%
            "handicap_rank":   round(handicap_ranks[i]    * 100, 1),   # 3%
            "trainer_rank":    round(trainer_ranks[i]     * 100, 1),   # 2%
            # Ekstra rådata
            "odds":              odds.get(name),
            "start_pos":         start_positions.get(name),
            "extra_distance":    extra_distances.get(name),
            "earnings_per_start": round(eps) if eps else None,
            "top3_rate":         round(s.get("top3_rate_smooth", PRIOR_TOP3_RATE) * 100, 1),
            "win_rate":          round(s.get("win_rate_smooth",  PRIOR_WIN_RATE) * 100, 1),
            "form_pos":          round(s.get("best2of3_pos"), 2) if s.get("best2of3_pos") else None,
            "jockey":            jockey_disp,
            "trainer":           trainer_disp,
            "data_points":       s.get("races", 0),
            "has_data":          bool(s.get("has_data", 0)),
            "jockey_from_db":    not bool(jockey_provided),
            "best_km_time":      round(s.get("best_km_time"), 2) if s.get("best_km_time") else None,
            "atg_age":           s.get("atg_age"),
            "atg_money":         s.get("atg_money"),
            "used_ml":           used_ml,
        })

    out.sort(key=lambda x: x["win_prob"], reverse=True)
    return out
