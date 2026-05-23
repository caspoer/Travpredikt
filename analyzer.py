"""
Statistikk- og analyselag mot SQLite-databasen.

Sentral forbedring: alle hestespørringer bruker en dedupliserings-CTE
som kun teller ETT resultat per (hest, dato, posisjon, startnr) selv om
dataen finnes i flere kilder (rikstoto + travsport). Travsport-rader
foretrekkes siden de har bloddtype, tid og trener.

Aliassystemet gjør det mulig å slå sammen hestenavn som egentlig er
samme hest (f.eks. norsk vs. svensk stavemåte, eller demo-data).
"""
from database import get_conn


# ─── Dedup-CTE ────────────────────────────────────────────────────────────────
# Returnerer én rad per (hest, dato, posisjon/startnr) – travsport-raden
# foretrekkes foran rikstoto når begge kildene har data for samme løp.

_DEDUP_CTE = """
WITH deduped AS (
    SELECT
        r.horse_name,
        LOWER(r.horse_name)                       AS name_key,
        r.position,
        r.start_pos,
        r.jockey,
        r.trainer,
        r.odds,
        r.time_sec,
        rc.race_id,
        rc.date,
        rc.track,
        rc.distance,
        rc.blood_type,
        rc.race_type,
        rc.source,
        ROW_NUMBER() OVER (
            PARTITION BY
                LOWER(r.horse_name),
                rc.date,
                COALESCE(CAST(r.position  AS TEXT), 'x'),
                COALESCE(CAST(r.start_pos AS TEXT), 'x')
            ORDER BY
                CASE rc.source WHEN 'travsport' THEN 0 ELSE 1 END,
                r.id
        ) AS _rank
    FROM results r
    JOIN races rc ON r.race_id = rc.race_id
    WHERE r.position IS NOT NULL
)
"""


# ─── Alias-hjelper ────────────────────────────────────────────────────────────

def _resolve(name: str, conn) -> str:
    """Returnerer kanonisk navn (lowercase) for et hestenavn, eller lowercase(name)."""
    row = conn.execute(
        "SELECT canonical FROM horse_aliases WHERE alias = LOWER(?)", (name,)
    ).fetchone()
    return row["canonical"] if row else name.lower()


# ─── Hest-statistikk ──────────────────────────────────────────────────────────

def horse_stats(horse_name: str) -> dict:
    with get_conn() as conn:
        canon = _resolve(horse_name, conn)

        # Alle aliaser som peker hit (for å vise evt. alternative navn)
        alias_rows = conn.execute(
            "SELECT alias FROM horse_aliases WHERE canonical = ?", (canon,)
        ).fetchall()
        aliases = [r["alias"] for r in alias_rows]

        # Dedupliserte resultater for denne hesten (og evt. aliaser)
        rows = conn.execute(f"""
            {_DEDUP_CTE}
            SELECT * FROM deduped
            WHERE _rank = 1
              AND (name_key = ?
                   OR name_key IN (
                       SELECT alias FROM horse_aliases WHERE canonical = ?
                   ))
            ORDER BY date DESC
            LIMIT 100
        """, (canon, canon)).fetchall()

    if not rows:
        return {}

    total = len(rows)
    wins  = sum(1 for r in rows if r["position"] == 1)
    top3  = sum(1 for r in rows if r["position"] and r["position"] <= 3)
    times = [r["time_sec"] for r in rows if r["time_sec"]]

    # Bloddtyper: travsport-rader er prioritert i dedup, så blood_type er korrekt
    blood_types = list({r["blood_type"] for r in rows if r["blood_type"]})

    return {
        "horse":        horse_name,
        "canonical":    canon,
        "aliases":      aliases,
        "blood_types":  blood_types,
        "races":        total,
        "wins":         wins,
        "win_pct":      round(wins / total * 100, 1),
        "top3":         top3,
        "top3_pct":     round(top3 / total * 100, 1),
        "best_time":    round(min(times), 2) if times else None,
        "last_5":       [{"date": r["date"], "pos": r["position"],
                          "track": r["track"], "dist": r["distance"],
                          "blood_type": r["blood_type"],
                          "race_type": r["race_type"]}
                         for r in rows[:5]],
    }


# ─── Jockey-statistikk ───────────────────────────────────────────────────────

def jockey_stats(jockey: str) -> dict:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT r.position, rc.date, rc.track
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE LOWER(r.jockey) = LOWER(?)
            ORDER BY rc.date DESC
            LIMIT 100
        """, (jockey,)).fetchall()

    if not rows:
        return {}

    total = len(rows)
    wins  = sum(1 for r in rows if r["position"] == 1)
    top3  = sum(1 for r in rows if r["position"] and r["position"] <= 3)

    return {
        "jockey":   jockey,
        "races":    total,
        "wins":     wins,
        "win_pct":  round(wins / total * 100, 1),
        "top3_pct": round(top3 / total * 100, 1),
    }


# ─── Bane-statistikk ─────────────────────────────────────────────────────────

def track_stats(track: str) -> dict:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT r.horse_name, r.position, r.odds
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE LOWER(rc.track) LIKE LOWER(?)
            ORDER BY rc.date DESC
            LIMIT 200
        """, (f"%{track}%",)).fetchall()

    if not rows:
        return {}

    from collections import Counter
    winners = [r["horse_name"] for r in rows if r["position"] == 1]
    top_horses = Counter(winners).most_common(5)

    return {
        "track":       track,
        "races_found": len(rows),
        "top_winners": [{"horse": h, "wins": c} for h, c in top_horses],
    }


# ─── Hesteliste ───────────────────────────────────────────────────────────────

def all_horses(
    search: str = "",
    blood_type: str | None = None,
    sort: str = "wins",
    page: int = 1,
    per_page: int = 50,
) -> dict:
    dedup_where = ["_rank = 1"]
    params: list = []

    if blood_type and blood_type != "alle":
        dedup_where.append("blood_type = ?")
        params.append(blood_type)

    if search:
        dedup_where.append("name_key LIKE LOWER(?)")
        params.append(f"%{search}%")

    where = "WHERE " + " AND ".join(dedup_where)

    valid_sorts = {
        "wins":     "wins DESC",
        "races":    "races DESC",
        "win_pct":  "win_pct DESC",
        "top3_pct": "top3_pct DESC",
        "name":     "horse_name ASC",
    }
    order = valid_sorts.get(sort, "wins DESC")

    with get_conn() as conn:
        total_row = conn.execute(f"""
            {_DEDUP_CTE}
            SELECT COUNT(DISTINCT name_key) AS c FROM deduped {where}
        """, params).fetchone()
        total = total_row["c"] if total_row else 0

        offset = (page - 1) * per_page
        rows = conn.execute(f"""
            {_DEDUP_CTE}
            SELECT
                horse_name,
                -- Velg bloddtype fra travsport-rad (allerede prioritert i dedup)
                MAX(CASE WHEN source = 'travsport' THEN blood_type ELSE NULL END)
                    AS blood_type_ts,
                blood_type,
                COUNT(*)  AS races,
                SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN position <= 3 THEN 1 ELSE 0 END) AS top3,
                ROUND(SUM(CASE WHEN position = 1 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1)
                    AS win_pct,
                ROUND(SUM(CASE WHEN position <= 3 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1)
                    AS top3_pct,
                ROUND(MIN(time_sec), 2) AS best_time,
                MAX(date)               AS last_race
            FROM deduped
            {where}
            GROUP BY name_key
            ORDER BY {order}
            LIMIT ? OFFSET ?
        """, (*params, per_page, offset)).fetchall()

    # Normaliser blood_type: foretrekk travsport-verdien
    def _row(r):
        d = dict(r)
        d["blood_type"] = d.pop("blood_type_ts") or d.pop("blood_type", "varmblod")
        return d

    return {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    max(1, -(-total // per_page)),
        "horses":   [_row(r) for r in rows],
    }


# ─── Toppliste ────────────────────────────────────────────────────────────────

def leaderboard(limit: int = 20, blood_type: str | None = None) -> list[dict]:
    extra = ""
    params: list = []
    if blood_type and blood_type != "alle":
        extra = "AND blood_type = ?"
        params.append(blood_type)

    with get_conn() as conn:
        rows = conn.execute(f"""
            {_DEDUP_CTE}
            SELECT
                horse_name,
                MAX(CASE WHEN source = 'travsport' THEN blood_type ELSE NULL END)
                    AS blood_type_ts,
                blood_type,
                COUNT(*)  AS races,
                SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN position <= 3 THEN 1 ELSE 0 END) AS top3,
                ROUND(MIN(time_sec), 2) AS best_time
            FROM deduped
            WHERE _rank = 1 {extra}
            GROUP BY name_key
            HAVING races >= 3
            ORDER BY wins DESC, top3 DESC
            LIMIT ?
        """, (*params, limit)).fetchall()

    def _row(r):
        d = dict(r)
        d["blood_type"] = d.pop("blood_type_ts") or d.pop("blood_type", "varmblod")
        return d

    return [_row(r) for r in rows]


# ─── Nylige løp ──────────────────────────────────────────────────────────────

def recent_races(limit: int = 20, blood_type: str | None = None) -> list[dict]:
    where = ""
    params: list = []
    if blood_type and blood_type != "alle":
        where = "WHERE rc.blood_type = ?"
        params.append(blood_type)

    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT rc.race_id, rc.track, rc.date, rc.distance,
                   rc.source, rc.blood_type, rc.race_type,
                   COUNT(r.id) AS starters
            FROM races rc
            LEFT JOIN results r ON rc.race_id = r.race_id
            {where}
            GROUP BY rc.race_id
            ORDER BY rc.date DESC, rc.id DESC
            LIMIT ?
        """, (*params, limit)).fetchall()

    return [dict(r) for r in rows]


# ─── Løpsdetaljer ────────────────────────────────────────────────────────────

def race_detail(race_id: str) -> dict:
    with get_conn() as conn:
        race = conn.execute(
            "SELECT * FROM races WHERE race_id = ?", (race_id,)
        ).fetchone()

        if not race:
            return {}

        # NULL-plasseringer (disq/scratched) skal sist, ikke først.
        starters = conn.execute("""
            SELECT r.*,
                   s.age   AS atg_age,
                   s.sex   AS atg_sex,
                   s.money AS atg_money,
                   s.life_starts AS atg_life_starts,
                   s.life_win_pct AS atg_life_win_pct,
                   s.best_time_sec AS atg_best_time
            FROM results r
            LEFT JOIN horse_stats_atg s ON s.horse_reg_no = r.horse_reg_no
            WHERE r.race_id = ?
            ORDER BY CASE WHEN r.position IS NULL THEN 999 ELSE r.position END,
                     r.start_pos
        """, (race_id,)).fetchall()

    return {
        "race":     dict(race),
        "starters": [dict(s) for s in starters],
    }


# ─── Alias/sammenslåing ───────────────────────────────────────────────────────

def merge_horses(alias_name: str, canonical_name: str) -> dict:
    """
    Slår sammen to hestenavn til ett profil.
    alias_name vil heretter behandles som canonical_name i alle spørringer.
    """
    alias = alias_name.lower().strip()
    canon = canonical_name.lower().strip()

    if alias == canon:
        return {"error": "Alias og kanonisk navn er identiske"}

    with get_conn() as conn:
        # Sjekk at begge finnes i databasen
        a_exists = conn.execute(
            "SELECT 1 FROM results WHERE LOWER(horse_name) = ? LIMIT 1", (alias,)
        ).fetchone()
        c_exists = conn.execute(
            "SELECT 1 FROM results WHERE LOWER(horse_name) = ? LIMIT 1", (canon,)
        ).fetchone()

        if not a_exists:
            return {"error": f"Fant ingen resultater for '{alias_name}'"}
        if not c_exists:
            return {"error": f"Fant ingen resultater for '{canonical_name}'"}

        conn.execute("""
            INSERT OR REPLACE INTO horse_aliases(alias, canonical)
            VALUES (?, ?)
        """, (alias, canon))

    return {
        "status":    "ok",
        "alias":     alias,
        "canonical": canon,
        "message":   f"'{alias_name}' slått sammen med '{canonical_name}'",
    }


def list_aliases() -> list[dict]:
    """Returnerer alle registrerte aliaser."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT alias, canonical, created_at FROM horse_aliases ORDER BY canonical, alias"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_alias(alias_name: str) -> dict:
    """Fjerner et alias."""
    alias = alias_name.lower().strip()
    with get_conn() as conn:
        conn.execute("DELETE FROM horse_aliases WHERE alias = ?", (alias,))
    return {"status": "ok", "deleted": alias}


def auto_merge_by_reg_no() -> dict:
    """
    Finner alle hester der samme horse_reg_no har flere ulike navn (innenfor
    samme kilde-konvensjon) og slår dem sammen automatisk.

    Velger som kanonisk navn:
      - Det navnet som forekommer flest ganger
      - Ved likhet: alfabetisk første

    horse_reg_no har forskjellig format per kilde:
      - rikstoto: 15-sifret ITU-format ('578001020120391')
      - ATG:      6-sifret intern ID ('766018')
    Vi grupperer derfor pr. reg_no, ikke pr. (reg_no, source) – siden samme
    nummer betyr samme hest INNENFOR det formatet.
    """
    from collections import Counter, defaultdict

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT horse_reg_no,
                   LOWER(horse_name) AS name_key,
                   COUNT(*)          AS occurrences
            FROM results
            WHERE horse_reg_no IS NOT NULL
              AND horse_reg_no != ''
              AND horse_name IS NOT NULL
              AND horse_name != ''
            GROUP BY horse_reg_no, LOWER(horse_name)
        """).fetchall()

        # Grupper: reg_no → [(navn, antall), ...]
        by_reg: dict = defaultdict(list)
        for r in rows:
            by_reg[r["horse_reg_no"]].append((r["name_key"], r["occurrences"]))

        merges_created = 0
        merges_skipped = 0
        details: list[dict] = []

        for reg_no, name_counts in by_reg.items():
            if len(name_counts) < 2:
                continue   # ingen flere navn-varianter for samme ID

            # Velg kanonisk: høyest antall, deretter alfabetisk
            name_counts_sorted = sorted(name_counts, key=lambda x: (-x[1], x[0]))
            canonical = name_counts_sorted[0][0]
            aliases   = [n for n, _ in name_counts_sorted[1:]]

            for alias in aliases:
                if alias == canonical:
                    continue
                # Sjekk om aliaset allerede finnes som canonical (kjede)
                existing = conn.execute(
                    "SELECT canonical FROM horse_aliases WHERE alias = ?",
                    (alias,)
                ).fetchone()
                if existing and existing["canonical"] == canonical:
                    merges_skipped += 1
                    continue

                conn.execute("""
                    INSERT OR REPLACE INTO horse_aliases(alias, canonical)
                    VALUES (?, ?)
                """, (alias, canonical))
                merges_created += 1
                details.append({
                    "reg_no":    reg_no,
                    "alias":     alias,
                    "canonical": canonical,
                })

    return {
        "status":         "ok",
        "merges_created": merges_created,
        "merges_skipped": merges_skipped,
        "details":        details[:50],   # begrens responsstørrelse
    }


def horse_enrichment(name: str) -> dict:
    """
    Returnerer ATG-berikelse for en hest (alder, kjønn, livstidsinntekter, PR).
    Slår opp via horse_aliases → kanonisk navn → horse_stats_atg.
    """
    with get_conn() as conn:
        canon = _resolve(name, conn)

        # Slå opp via navn (case-insensitive)
        row = conn.execute("""
            SELECT * FROM horse_stats_atg
            WHERE LOWER(name) = ?
            ORDER BY updated_at DESC
            LIMIT 1
        """, (canon,)).fetchone()

        if not row:
            # Fallback: prøv også alle aliaser
            row = conn.execute("""
                SELECT s.* FROM horse_stats_atg s
                JOIN horse_aliases a ON LOWER(s.name) = a.alias
                WHERE a.canonical = ?
                ORDER BY s.updated_at DESC
                LIMIT 1
            """, (canon,)).fetchone()

    return dict(row) if row else {}


def suggest_duplicates(limit: int = 30) -> list[dict]:
    """
    Foreslår mulige duplikater basert på:
    - Samme dato og posisjon (sterk indikasjon på dobbeltregistrering)
    - Nesten identisk navn (samme fire første bokstaver, maks 4 tegns lengdeforskjell)
    - Samme navn i ulike kilder (rikstoto vs travsport vs atg) → "kilde-overlapp"

    Bruker Python-side gruppering for å unngå treg SQL self-join.
    """
    from collections import defaultdict

    with get_conn() as conn:
        # Hent alle (navn, dato, posisjon, source)-kombinasjoner
        rows = conn.execute("""
            SELECT LOWER(r.horse_name) AS name, rc.date, r.position, rc.source
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE r.position IS NOT NULL
        """).fetchall()

        # Hent eksisterende aliaser for å filtrere dem ut
        alias_rows = conn.execute(
            "SELECT alias, canonical FROM horse_aliases"
        ).fetchall()

    known_pairs = {(r["alias"], r["canonical"]) for r in alias_rows}

    # Bygg indeks: (dato, posisjon) → set av hestenavn
    dp_index: dict = defaultdict(set)
    # Bygg også: navn → set av kilder (for cross-source-deteksjon)
    name_sources: dict = defaultdict(set)
    for row in rows:
        dp_index[(row["date"], row["position"])].add(row["name"])
        if row["source"]:
            name_sources[row["name"]].add(row["source"])

    # Tell delte (dato, posisjon)-par for hvert navnepar.
    # Kriterier:
    #   - Samme fire første bokstaver
    #   - Maks 4 tegns lengdeforskjell
    pair_counts: dict = defaultdict(int)
    for names in dp_index.values():
        lst = sorted(names)
        for i, n1 in enumerate(lst):
            for n2 in lst[i + 1:]:
                if (n1[:4] == n2[:4]
                        and abs(len(n1) - len(n2)) <= 4
                        and n1 != n2):
                    pair_counts[(n1, n2)] += 1

    # Filtrer bort allerede registrerte aliaser og sorter.
    # Krever minst 2 felles løp for å redusere tilfeldige treff.
    result = []
    for (n1, n2), cnt in sorted(pair_counts.items(), key=lambda x: -x[1]):
        if cnt < 2:
            break  # sortert fallende – resten er også < 2
        if (n1, n2) not in known_pairs and (n2, n1) not in known_pairs:
            # Annoter med hvilke kilder navnene finnes i
            src1 = sorted(name_sources.get(n1, set()))
            src2 = sorted(name_sources.get(n2, set()))
            cross = "ja" if (set(src1) - set(src2)) or (set(src2) - set(src1)) else "nei"
            result.append({
                "name1":         n1,
                "name2":         n2,
                "shared_races":  cnt,
                "source1":       ",".join(src1) or "?",
                "source2":       ",".join(src2) or "?",
                "cross_source":  cross,
            })
        if len(result) >= limit:
            break

    return result
