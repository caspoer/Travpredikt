"""
Genererer realistisk demo-data slik at appen fungerer uten nett-tilgang.
Kjør: python demo_data.py
"""
import random
import datetime
from database import get_conn, init_db

# Baner som typisk kjører kaldblod og/eller varmblod i Norge
TRACKS_KALD = ["Bjerke", "Leangen", "Momarken", "Jarlsberg", "Klosterskogen", "Forus"]
TRACKS_VARM = ["Bjerke", "Øvrevoll", "Forus", "Leangen", "Åby", "Solvalla"]
TRACKS_GALOPP = ["Øvrevoll", "Bjerke"]

# Kaldblodshester – typiske norske kaldblodstravere-navn
HORSES_KALD = [
    "Gausdal Brage", "Hallaren Odin", "Vålåsjøen Prins", "Dovre Kongen",
    "Jotun Svarten", "Fjord Hauk", "Heidal Blesen", "Mjøsa Viking",
    "Gudbrand Svarten", "Rondane Guten", "Otta Blomsten", "Peer Gynt",
    "Håkon Jarl", "Askeladden Jr", "Kjønnå Blesen", "Valdres Prinsesse",
    "Numedal Svarten", "Telemark Kongen", "Hardanger Guten", "Voss Brage",
]

# Varmblodshester – mer internasjonale navn
HORSES_VARM = [
    "Global Rocket", "Ready Cash Jr", "Zacon Gio", "Propulsion",
    "Readly Express", "Don Fanucci Zet", "Melby Pål", "Nuncio",
    "Maharajah", "Quack", "Trixton", "Tycoon Cr",
    "Timoko", "Bold Eagle", "Ringostarr Treb", "Milliondollar",
    "Quite Easy", "Muscle Hill Jr", "Muscle Mass", "Chapter Seven",
]

# Galoppere
HORSES_GALOPP = [
    "Enable Jr", "Frankel Son", "Sea The Stars Jr", "Galileo Jr",
    "Dubawi Jr", "Golden Horn Jr", "Roaring Lion Jr", "Too Darn Hot",
    "Saxon Warrior Jr", "Anthony Van Dyck",
]

JOCKEYS_KALD = ["Jan-Roar Mjølnerød", "Vidar Hop", "Tom Erik Solberg",
                 "Øystein Tjomsland", "Frode Hamre Jr"]
JOCKEYS_VARM = ["Bjørn Goop", "Örjan Kihlström", "Eirik Høitomt",
                 "Lars Anvar Kolle", "Kenneth Haugstad"]
JOCKEYS_GALOPP = ["Pål Bjoerk", "Raul Da Silva", "Frode Schindler",
                   "Christophe Soumillon Jr"]

TRAINERS_KALD = ["Frode Hamre", "Geir Vegard Gundersen", "Björn Aurstad"]
TRAINERS_VARM = ["Timo Nurmos", "Stefan Melander", "Per Oleg Midtfjeld"]
TRAINERS_GALOPP = ["Niels Petersen", "Svend Pedersen", "Tony Carroll Jr"]

BLOOD_CONFIGS = {
    "kaldblod": {
        "horses":   HORSES_KALD,
        "jockeys":  JOCKEYS_KALD,
        "trainers": TRAINERS_KALD,
        "tracks":   TRACKS_KALD,
        "distances": [1000, 1200, 1500, 1600, 2000, 2100],
        # Kaldblod er tregere: ca 1 km/min → 1000m ≈ 90s
        "time_base_per_m": 1 / 13.5,
        "race_type": "kaldblodstrav",
    },
    "varmblod": {
        "horses":   HORSES_VARM,
        "jockeys":  JOCKEYS_VARM,
        "trainers": TRAINERS_VARM,
        "tracks":   TRACKS_VARM,
        "distances": [1600, 2100, 2600, 3100],
        # Varmblod raskere: ca 1 km/1:12 → 1600m ≈ 115s
        "time_base_per_m": 1 / 14.5,
        "race_type": "varmblodstrav",
    },
    "galopp": {
        "horses":   HORSES_GALOPP,
        "jockeys":  JOCKEYS_GALOPP,
        "trainers": TRAINERS_GALOPP,
        "tracks":   TRACKS_GALOPP,
        "distances": [1000, 1200, 1400, 1600, 2000, 2400],
        # Galopp enda raskere
        "time_base_per_m": 1 / 16.5,
        "race_type": "galopp",
    },
}


def _random_time(dist: int, base_per_m: float) -> float:
    base = dist * base_per_m
    return round(base + random.uniform(-4, 4), 1)


def generate_demo(days: int = 365):
    init_db()
    today = datetime.date.today()
    conn  = get_conn()

    # Slett gammel demo-data
    conn.execute("DELETE FROM results WHERE race_id LIKE 'demo_%'")
    conn.execute("DELETE FROM races   WHERE race_id LIKE 'demo_%'")
    conn.commit()

    race_count = 0
    for days_back in range(days, 0, -1):
        if random.random() < 0.35:
            continue
        race_date = (today - datetime.timedelta(days=days_back)).isoformat()

        # Blande kaldblod, varmblod og galopp på samme raceday
        blood_types_today = random.choices(
            ["kaldblod", "varmblod", "galopp"],
            weights=[45, 40, 15],
            k=random.randint(3, 7)
        )

        for r_idx, blood_type in enumerate(blood_types_today):
            cfg     = BLOOD_CONFIGS[blood_type]
            track   = random.choice(cfg["tracks"])
            distance = random.choice(cfg["distances"])
            race_id  = f"demo_{race_date}_{r_idx}"

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO races
                        (race_id, track, date, distance, surface, race_type, blood_type, source)
                    VALUES (?, ?, ?, ?, 'grus', ?, ?, 'demo')
                """, (race_id, track, race_date, distance, cfg["race_type"], blood_type))
            except Exception:
                continue

            n_starters = random.randint(6, 10)
            starters   = random.sample(cfg["horses"], min(n_starters, len(cfg["horses"])))
            positions  = list(range(1, len(starters) + 1))
            random.shuffle(positions)
            base_time  = _random_time(distance, cfg["time_base_per_m"])

            for i, horse in enumerate(starters):
                pos     = positions[i]
                jockey  = random.choice(cfg["jockeys"])
                trainer = random.choice(cfg["trainers"])
                t       = base_time + (pos - 1) * random.uniform(0.3, 1.0)
                odds    = round(random.uniform(1.5, 25.0), 1)

                conn.execute("""
                    INSERT INTO results
                        (race_id, horse_name, position, start_pos, jockey, trainer, odds, time_sec)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (race_id, horse, pos, i + 1, jockey, trainer, odds, round(t, 1)))

            race_count += 1

    conn.commit()
    conn.close()

    kald = sum(1 for bt in ["kaldblod"] if bt)
    print(f"Demo-data generert: {race_count} løp (kaldblod + varmblod + galopp)")


if __name__ == "__main__":
    generate_demo()
