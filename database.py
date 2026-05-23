import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "hestedata.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS races (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id     TEXT UNIQUE,
            track       TEXT,
            date        TEXT,
            distance    INTEGER,
            surface     TEXT,
            race_type   TEXT,
            blood_type  TEXT DEFAULT 'varmblod',
            source      TEXT,
            fetched_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS horses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT,
            country     TEXT,
            birth_year  INTEGER,
            blood_type  TEXT DEFAULT 'varmblod',
            UNIQUE(name, country)
        );

        -- Aliaser: gjor det mulig aa sla sammen to hestenavn til ett profil.
        -- alias = LOWER(alternativt navn), canonical = LOWER(kanonisk navn).
        CREATE TABLE IF NOT EXISTS horse_aliases (
            alias       TEXT PRIMARY KEY,
            canonical   TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id         TEXT REFERENCES races(race_id),
            horse_name      TEXT,
            position        INTEGER,
            start_pos       INTEGER,
            jockey          TEXT,
            trainer         TEXT,
            odds            REAL,
            time_sec        REAL,
            scratched       INTEGER DEFAULT 0,
            extra_distance  INTEGER DEFAULT 0,
            win_odds        REAL,
            horse_reg_no    TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id     TEXT,
            horse_name  TEXT,
            score       REAL,
            win_prob    REAL,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        -- Heste-berikelse fra ATG (per hest, ikke per loep).
        -- Oppdateres hver gang vi henter et ATG-loep der hesten deltar.
        -- horse_reg_no = ATG sin interne heste-ID (ikke samme system som rikstoto!)
        CREATE TABLE IF NOT EXISTS horse_stats_atg (
            horse_reg_no    TEXT PRIMARY KEY,
            name            TEXT,
            age             INTEGER,
            sex             TEXT,
            color           TEXT,
            money           INTEGER,        -- livstidsinntekter i oere
            life_starts     INTEGER,
            life_wins       INTEGER,
            life_2nd        INTEGER,
            life_3rd        INTEGER,
            life_win_pct    REAL,           -- vinnerprosent (livstid)
            life_place_pct  REAL,           -- topp-3-prosent (livstid)
            best_time_sec   REAL,           -- beste personlige rekord (sek)
            father_name     TEXT,
            mother_name     TEXT,
            updated_at      TEXT DEFAULT (datetime('now'))
        );

        -- Indeks for navnoppslag (case-insensitive)
        CREATE INDEX IF NOT EXISTS idx_horse_stats_atg_name
            ON horse_stats_atg(LOWER(name));
        """)
        # Migrer eksisterende DB: legg til kolonner om de mangler
        for col, tbl, default in [
            ("blood_type",     "races",   "'varmblod'"),
            ("blood_type",     "horses",  "'varmblod'"),
            ("extra_distance", "results", "0"),
            ("win_odds",       "results", "NULL"),
            ("horse_reg_no",   "results", "NULL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT DEFAULT {default}")
            except Exception:
                pass

        # Sikre unik indeks på results (race_id + horse_name).
        # Rydd opp duplikater fra tidligere kjøringer før indeksen opprettes.
        try:
            conn.execute("""
                DELETE FROM results
                WHERE id NOT IN (
                    SELECT MIN(id) FROM results GROUP BY race_id, horse_name
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_results_unique
                ON results(race_id, horse_name)
            """)
        except Exception:
            pass


if __name__ == "__main__":
    init_db()
    print("Database initialisert:", DB_PATH)
