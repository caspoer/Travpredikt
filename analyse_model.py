"""
Analyseskript: datakvalitet og modellytelse.
"""
import sqlite3

conn = sqlite3.connect("hestedata.db")
conn.row_factory = sqlite3.Row

print("=== DATAKVALITET ===")
print(f"Totalt loep:                   {conn.execute('SELECT COUNT(*) FROM races').fetchone()[0]}")
print(f"Resultater med posisjon:       {conn.execute('SELECT COUNT(*) FROM results WHERE position IS NOT NULL').fetchone()[0]}")
print(f"Vinnere (pos=1):               {conn.execute('SELECT COUNT(*) FROM results WHERE position = 1').fetchone()[0]}")
print(f"Resultater med jockey:         {conn.execute('SELECT COUNT(*) FROM results WHERE jockey IS NOT NULL AND jockey != \"\"').fetchone()[0]}")
print(f"Resultater med tid (time_sec): {conn.execute('SELECT COUNT(*) FROM results WHERE time_sec IS NOT NULL').fetchone()[0]}")

print("\n=== LOEPSFORDELING (blood_type) ===")
for row in conn.execute("SELECT blood_type, COUNT(*) as c FROM races GROUP BY blood_type ORDER BY c DESC").fetchall():
    print(f"  {row['blood_type']:<20} {row['c']} loep")

print("\n=== STARTERE PER LOEP ===")
r = conn.execute("""
    SELECT AVG(cnt) as avg, MIN(cnt) as mn, MAX(cnt) as mx
    FROM (SELECT COUNT(*) as cnt FROM results WHERE position IS NOT NULL GROUP BY race_id)
""").fetchone()
print(f"  Snitt: {r['avg']:.1f}   Min: {r['mn']}   Max: {r['mx']}")

print("\n=== HESTERDYBDE ===")
for threshold, label in [(1,"1+ loep"),(3,"3+ loep"),(5,"5+ loep"),(10,"10+ loep"),(20,"20+ loep")]:
    n = conn.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT horse_name FROM results WHERE position IS NOT NULL
            GROUP BY LOWER(horse_name) HAVING COUNT(*) >= {threshold}
        )
    """).fetchone()[0]
    print(f"  {n} hester med {label}")

print("\n=== DATA-LEKKASJE RISIKO ===")
r = conn.execute("""
    SELECT
        SUM(CASE WHEN cnt < 3  THEN 1 ELSE 0 END) as few,
        SUM(CASE WHEN cnt >= 3 AND cnt < 10 THEN 1 ELSE 0 END) as medium,
        SUM(CASE WHEN cnt >= 10 THEN 1 ELSE 0 END) as many
    FROM (
        SELECT LOWER(horse_name) as name, COUNT(*) as cnt
        FROM results WHERE position IS NOT NULL
        GROUP BY LOWER(horse_name)
    )
""").fetchone()
print(f"  < 3 loep: {r['few']} hester   (usikre stats, stoey i treningen)")
print(f"  3-9 loep: {r['medium']} hester  (akseptabelt)")
print(f"  10+ loep: {r['many']} hester    (paalitelige stats)")

print("\n=== MODUL: OVERFITTING-ANALYSE ===")
print("  Train AUC: 0.769")
print("  Test AUC:  0.592")
print("  Gap:       0.177   STORT - tyder paa betydelig overfitting")
print()
print("  Viktigste feature: win_vs_field = 84%   (for dominerende)")
print("  Problemet: win_rate er beregnet inkl. FREMTIDIGE loep (data-lekkasje)")

print("\n=== JOCKEY-DEKNING ===")
total      = conn.execute("SELECT COUNT(*) FROM results WHERE position IS NOT NULL").fetchone()[0]
with_j     = conn.execute("SELECT COUNT(*) FROM results WHERE jockey IS NOT NULL AND jockey != ''").fetchone()[0]
print(f"  {with_j}/{total} resultater har jockey ({with_j/total*100:.0f}%)")
unique_j   = conn.execute("SELECT COUNT(DISTINCT LOWER(jockey)) FROM results WHERE jockey IS NOT NULL AND jockey != ''").fetchone()[0]
j5plus     = conn.execute("""
    SELECT COUNT(*) FROM (
        SELECT jockey FROM results WHERE jockey IS NOT NULL AND jockey != '' AND position IS NOT NULL
        GROUP BY LOWER(jockey) HAVING COUNT(*) >= 5
    )
""").fetchone()[0]
print(f"  {unique_j} unike jockeyer  /  {j5plus} med 5+ loep (paalitelig win-rate)")

print("\n=== DATOFORDELING (siste 12 mnd) ===")
rows = conn.execute("""
    SELECT substr(date,1,7) as month, COUNT(*) as c
    FROM races GROUP BY month ORDER BY month DESC LIMIT 12
""").fetchall()
for r in rows:
    print(f"  {r['month']}: {r['c']} loep")

conn.close()
