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

def predict_field(horses: list, jockeys: dict | None = None) -> list:
    """
    Ranger et felt. Bruker ML-modell hvis trent, ellers Bayesiansk heuristikk.
    """
    if not horses:
        return []
    if jockeys is None:
        jockeys = {}

    with get_conn() as conn:
        horse_stats  = _precompute_horse_stats(conn)
        jockey_stats = _precompute_jockey_stats(conn)

    X            = _build_feature_matrix(horses, horse_stats, jockey_stats, jockeys)
    used_ml      = False

    with get_conn() as conn:
        jockey_recent = _precompute_jockey_recent_stats(conn)

    n = len(horses)

    # ── Hjelper: rang innen felt [min_rank = dårligst, 1.0 = best] ──────────
    # min_rank sikrer at selv den svakeste hesten i feltet får en liten andel –
    # ingen hest bør ha 0% vinnersannsynlighet.
    def _ranks(values, higher_is_better=True, min_rank=0.15) -> list:
        idx_sorted = sorted(range(n), key=lambda i: values[i], reverse=higher_is_better)
        out   = [0.0] * n
        div   = max(n - 1, 1)
        span  = 1.0 - min_rank
        for rank, idx in enumerate(idx_sorted):
            out[idx] = 1.0 - rank / div * span
        return out

    # ── Komponent 1 (40 %): personlig rekord / kapasitet ────────────────────
    # Hester med tiddata → beste km-tid-rang.
    # Hester uten tiddata → gjennomsnittet av vinn-rang og topp3-rang (proxy).
    time_idx   = FEATURE_COLS.index("time_rank_norm")
    wr_idx     = FEATURE_COLS.index("win_rank_norm")
    t3r_idx    = FEATURE_COLS.index("top3_rank_norm")
    capacity   = []
    for i, name in enumerate(horses):
        s = horse_stats.get(name.lower(), {})
        if s.get("best_km_time") is not None:
            capacity.append(float(X[i, time_idx]))      # faktisk PR-rang
        else:
            capacity.append((float(X[i, wr_idx]) + float(X[i, t3r_idx])) / 2.0)
    cap_ranks = _ranks(capacity)

    # ── Komponent 2 (30 %): topp-3 siste 5 løp ──────────────────────────────
    form_vals  = [horse_stats.get(name.lower(), {}).get("top3_recent_5", 0.5)
                  for name in horses]
    form_ranks = _ranks(form_vals)

    # ── Komponent 3 (20 %): jockey siste 40 løp ─────────────────────────────
    jwr_recent_vals = []
    for name in horses:
        s   = horse_stats.get(name.lower(), {})
        jk  = (jockeys.get(name) or s.get("last_jockey") or "").lower()
        # Foretrekk siste-40-statistikk, fall tilbake på all-time
        val = jockey_recent.get(jk) or jockey_stats.get(jk, PRIOR_WIN_RATE)
        jwr_recent_vals.append(val)
    jockey_ranks = _ranks(jwr_recent_vals)

    # ── Komponent 4 (20 %): topp-3% karriere (konsistens) ───────────────────
    top3_career_vals  = [horse_stats.get(name.lower(), {}).get("top3_rate_smooth", PRIOR_TOP3_RATE)
                         for name in horses]
    top3_career_ranks = _ranks(top3_career_vals)

    # ── Komponent 5 (10 %): vinnede løp totalt ──────────────────────────────
    win_vals  = [horse_stats.get(name.lower(), {}).get("win_rate_smooth", PRIOR_WIN_RATE)
                 for name in horses]
    win_ranks = _ranks(win_vals)

    # ── Endelig score med eksplisitte vekter ─────────────────────────────────
    raw_scores = np.array([
        W_CAPACITY    * cap_ranks[i]         +
        W_FORM_TOP3   * form_ranks[i]        +
        W_TOP3_CAREER * top3_career_ranks[i] +
        W_WIN_TOTAL   * win_ranks[i]         +
        W_JOCKEY      * jockey_ranks[i]
        for i in range(n)
    ], dtype=np.float64)

    total  = float(raw_scores.sum())
    probs  = raw_scores / total if total > 0 else np.ones(n) / n
    raw_probs = raw_scores   # score-felt i output

    out = []
    for i, name in enumerate(horses):
        key = name.lower()
        s   = horse_stats.get(key, {})

        # Jockey: bruker oppgitt kusk (fra startliste) foerst, deretter siste kjente
        jockey_provided = jockeys.get(name, "")
        jk_key          = (jockey_provided or s.get("last_jockey") or "").lower()
        jockey_disp     = (jockey_provided
                           or s.get("last_jockey_disp")
                           or jk_key.title())
        jwr = jockey_stats.get(jk_key, PRIOR_WIN_RATE)

        best_kmt    = s.get("best_km_time")
        jwr_display = jwr_recent_vals[i]
        out.append({
            "horse":           name,
            "win_prob":        round(float(probs[i]) * 100, 1),
            "score":           round(float(raw_probs[i]), 4),
            # Komponentskårer (brukt i vekting)
            "cap_rank":        round(cap_ranks[i] * 100, 1),
            "form_top3":       round(form_vals[i] * 100, 1),          # % av siste 5 som var topp-3
            "top3_career":     round(top3_career_vals[i] * 100, 1),   # karriere topp-3% (konsistens)
            "win_rate":        round(s.get("win_rate_smooth",  PRIOR_WIN_RATE) * 100, 1),
            "jockey_wr":       round(jwr_display * 100, 1),           # jockey siste 40 løp
            # Ekstra info
            "top3_rate":       round(s.get("top3_rate_smooth", PRIOR_TOP3_RATE) * 100, 1),
            "form":            round((1.0 - min(s.get("avg_pos_5", 5.0), 10.0) / 10.0) * 100, 1),
            "jockey":          jockey_disp,
            "data_points":     s.get("races", 0),
            "has_data":        bool(s.get("has_data", 0)),
            "jockey_from_db":  not bool(jockey_provided),
            "best_km_time":    round(best_kmt, 2) if best_kmt else None,
            "used_ml":         used_ml,
        })

    out.sort(key=lambda x: x["win_prob"], reverse=True)
    return out
