"""
Prediksjonsmodell basert på form-score og historisk statistikk.

Score-komponenter (vektet sum, skala 0–100):
  - Seiersprosent (siste 10 løp)          30 %
  - Topp-3-prosent (siste 10)             20 %
  - Formkurve  (1. plass=5..5. plass=1)   20 %
  - Beste tid vs. feltet                  15 %
  - Jockey win-rate                       15 %
"""
import math
from database import get_conn


def _last_n_results(horse: str, n: int = 10) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT r.position, r.time_sec, r.jockey, rc.date
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE LOWER(r.horse_name) = LOWER(?)
              AND r.position IS NOT NULL
            ORDER BY rc.date DESC
            LIMIT ?
        """, (horse, n)).fetchall()
    return [dict(r) for r in rows]


def _jockey_win_rate(jockey: str) -> float:
    if not jockey:
        return 0.25
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) AS wins
            FROM results
            WHERE LOWER(jockey) = LOWER(?)
        """, (jockey,)).fetchone()
    total = row["total"] or 0
    wins  = row["wins"] or 0
    return wins / total if total >= 5 else 0.25


def _form_score(results: list[dict]) -> float:
    """Vekter nylige plasseringer – nyeste teller mest."""
    if not results:
        return 0.0
    score = 0.0
    weights = [5, 4, 3, 2, 1, 1, 1, 1, 1, 1]
    pos_points = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    for i, r in enumerate(results[:10]):
        w = weights[i] if i < len(weights) else 1
        p = pos_points.get(r["position"], 0)
        score += p * w
    max_score = sum(5 * w for w in weights[: len(results)])
    return score / max_score if max_score > 0 else 0.0


def score_horse(horse: str, field_times: list[float] | None = None) -> dict:
    results = _last_n_results(horse)
    if not results:
        return {"horse": horse, "score": 0.0, "win_prob": 0.0, "data_points": 0}

    total = len(results)
    wins  = sum(1 for r in results if r["position"] == 1)
    top3  = sum(1 for r in results if r["position"] <= 3)

    win_rate  = wins / total
    top3_rate = top3 / total
    form      = _form_score(results)

    jockey    = results[0]["jockey"] if results else ""
    jkey_rate = _jockey_win_rate(jockey)

    times = [r["time_sec"] for r in results if r["time_sec"]]
    time_score = 0.5
    if times and field_times:
        best = min(times)
        field_best = min(field_times)
        diff = best - field_best        # negative = hesten er raskere
        time_score = max(0.0, 1.0 - diff / 10.0)

    raw = (
        win_rate  * 30 +
        top3_rate * 20 +
        form      * 20 +
        time_score * 15 +
        jkey_rate  * 15
    )

    return {
        "horse":       horse,
        "score":       round(raw, 2),
        "win_rate":    round(win_rate * 100, 1),
        "top3_rate":   round(top3_rate * 100, 1),
        "form":        round(form * 100, 1),
        "jockey":      jockey,
        "jockey_wr":   round(jkey_rate * 100, 1),
        "data_points": total,
        "win_prob":    0.0,  # fylles inn av rank_field
    }


def rank_field(horses: list[str]) -> list[dict]:
    """Ranger en liste hester og tildel sannsynligheter via softmax."""
    field_times = []
    for h in horses:
        with get_conn() as conn:
            row = conn.execute("""
                SELECT MIN(time_sec) AS best
                FROM results r
                JOIN races rc ON r.race_id = rc.race_id
                WHERE LOWER(r.horse_name) = LOWER(?)
                  AND r.time_sec IS NOT NULL
            """, (h,)).fetchone()
        if row and row["best"]:
            field_times.append(row["best"])

    scores = [score_horse(h, field_times or None) for h in horses]

    raw_scores = [s["score"] for s in scores]
    if max(raw_scores, default=0) == 0:
        for s in scores:
            s["win_prob"] = round(100 / len(scores), 1) if scores else 0
        return sorted(scores, key=lambda x: x["score"], reverse=True)

    # Softmax for å konvertere scores til sannsynligheter
    exp_scores = [math.exp(s / 10) for s in raw_scores]
    total_exp  = sum(exp_scores)
    for i, s in enumerate(scores):
        s["win_prob"] = round(exp_scores[i] / total_exp * 100, 1)

    return sorted(scores, key=lambda x: x["score"], reverse=True)


def save_predictions(race_id: str, ranked: list[dict]):
    with get_conn() as conn:
        conn.execute("DELETE FROM predictions WHERE race_id = ?", (race_id,))
        for r in ranked:
            conn.execute("""
                INSERT INTO predictions (race_id, horse_name, score, win_prob)
                VALUES (?, ?, ?, ?)
            """, (race_id, r["horse"], r["score"], r["win_prob"]))
