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
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id     TEXT REFERENCES races(race_id),
            horse_name  TEXT,
            position    INTEGER,
            start_pos   INTEGER,
            jockey      TEXT,
            trainer     TEXT,
            odds        REAL,
            time_sec    REAL,
            scratched   INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id     TEXT,
            horse_name  TEXT,
            score       REAL,
            win_prob    REAL,
            created_at  TEXT DEFAULT (datetime('now'))
        );
        """)
        # Migrer eksisterende DB: legg til kolonner om de mangler
        for col, tbl, default in [
            ("blood_type", "races",  "'varmblod'"),
            ("blood_type", "horses", "'varmblod'"),
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
