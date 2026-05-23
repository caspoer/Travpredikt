"""
Utforsk Rikstoto-APIet for aa finne mer data.
"""
import json, requests, datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept":   "application/json, */*",
    "Referer":  "https://www.rikstoto.no/Resultater",
    "Origin":   "https://www.rikstoto.no",
}
BASE = "https://www.rikstoto.no/api"

def get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# --- 1. Test: hent en hel uke med ett kall (range) ---
print("=== 1. LISTE OVER EN UKE ===")
date_from = "2026-05-12"
date_to   = "2026-05-18"
data = get(f"{BASE}/results/racedays/{date_from}/{date_to}/list")
if "error" in data:
    print("Feil:", data)
else:
    result = data.get("result", [])
    print(f"Dager returnert: {len(result)}")
    for day in result:
        for rd in day.get("raceDays", []):
            print(f"  {rd.get('startTime','?')[:10]}  {rd.get('raceDayName','?'):<25}  "
                  f"sport={rd.get('sportType','?')}  country={rd.get('countryIsoCode','?')}  "
                  f"status={rd.get('progressStatus','?')}  key={rd.get('raceDay','?')}")

# --- 2. Finn en raceday-key og se hva starts inneholder ---
print("\n=== 2. DETALJERT STARTS (foerste finished raceday) ===")
rdk = None
for day in data.get("result", []):
    for rd in day.get("raceDays", []):
        if rd.get("progressStatus") == "Finished" and rd.get("countryIsoCode") == "NO":
            rdk = rd["raceDay"]
            rdk_name = rd.get("raceDayName", "?")
            break
    if rdk:
        break

if rdk:
    print(f"Bruker raceday: {rdk} ({rdk_name})")
    starts = get(f"{BASE}/racedays/{rdk}/starts")
    if "error" not in starts:
        result = starts.get("result", {})
        # Se paa foerste loep
        for rnum, runners in list(result.items())[:1]:
            print(f"\n  Loep {rnum}: {len(runners)} startere")
            if runners:
                print(f"  Foerste runner-keys: {list(runners[0].keys())}")
                print(f"  Eksempel: {json.dumps(runners[0], ensure_ascii=False, indent=4)}")
else:
    print("Ingen norsk finished raceday funnet")

# --- 3. Se hva raceresults inneholder (tider?) ---
print("\n=== 3. DETALJERT RACERESULTS ===")
if rdk:
    res = get(f"{BASE}/results/racedays/{rdk}/raceresults")
    if "error" not in res:
        rr = res.get("result", {}).get("raceResults", {})
        print(f"  Loep med resultater: {list(rr.keys())}")
        for rnum, entries in list(rr.items())[:1]:
            print(f"\n  Loep {rnum}: {len(entries)} entries")
            if entries:
                print(f"  Keys: {list(entries[0].keys())}")
                print(f"  Eksempel: {json.dumps(entries[0], ensure_ascii=False, indent=4)}")

# --- 4. Prøv race-spesifikk API ---
print("\n=== 4. RACE-SPESIFIKK ENDEPUNKT ===")
if rdk:
    # Prøv å hente mer info om ett spesifikt loep
    r1 = get(f"{BASE}/racedays/{rdk}/races/1")
    if "error" not in r1:
        print(f"  /races/1 keys: {list(r1.keys())}")
        print(f"  Data: {json.dumps(r1, ensure_ascii=False, indent=2)[:500]}")
    else:
        print(f"  /races/1: {r1}")

    r2 = get(f"{BASE}/results/racedays/{rdk}/races/1")
    if "error" not in r2:
        print(f"\n  /results/.../races/1 keys: {list(r2.keys())}")
        print(f"  Data: {json.dumps(r2, ensure_ascii=False, indent=2)[:500]}")
    else:
        print(f"  /results/.../races/1: {r2}")

# --- 5. Utenlandske loep tilgjengelig? ---
print("\n=== 5. UTENLANDSKE LOEP (SE, FI, DK) ===")
for day in data.get("result", []):
    for rd in day.get("raceDays", []):
        country = rd.get("countryIsoCode", "")
        if country not in ("NO", ""):
            print(f"  {rd.get('startTime','?')[:10]}  {rd.get('raceDayName','?'):<25}  "
                  f"country={country}  sport={rd.get('sportType','?')}  "
                  f"status={rd.get('progressStatus','?')}")

# --- 6. Kan vi hente lengre datoperioder? ---
print("\n=== 6. LANG PERIODE (30 dager med ett kall) ===")
d_from = "2026-04-01"
d_to   = "2026-04-30"
data2 = get(f"{BASE}/results/racedays/{d_from}/{d_to}/list")
if "error" not in data2:
    total = sum(len(day.get("raceDays", [])) for day in data2.get("result", []))
    print(f"  Kall for april 2026: {len(data2.get('result',[]))} dager, {total} racedays totalt")
else:
    print(f"  Feil: {data2}")
