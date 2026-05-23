"""
Scraper mot Rikstoto JSON-API og travsport.no HTML.

Rikstoto-endepunkter:
  - /api/results/racedays/{from}/{to}/list   → racedays i en periode
  - /api/racedays/{raceday_key}/starts       → startliste (hester + kusk)
  - /api/results/racedays/{raceday_key}/raceresults → plassering + startnr

Travsport.no:
  - /sportsbasen/lopskalender/?year=Y&month=M → liste over stevner med resultater
  - /travbaner/{track-slug}/results/{date}    → resultatsider (HTML)
  Gir: bloddtype (kald/varm), distanse, tid, trener, odds – data rikstoto mangler.
"""
import datetime
import re
import time
import requests
from bs4 import BeautifulSoup

# ── Rikstoto ──────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":   "application/json, text/plain, */*",
    "Referer":  "https://www.rikstoto.no/Resultater",
    "Origin":   "https://www.rikstoto.no",
}
BASE    = "https://www.rikstoto.no/api"
TIMEOUT = 15

COUNTRIES = {
    "NO": "Norge",
    "SE": "Sverige",
    "DK": "Danmark",
    "FI": "Finland",
    "FR": "Frankrike",
}

SOURCES = {"rikstoto": None, "travsport": None}


def _get(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Rikstoto: periode-basert raceday-liste ────────────────────────────────────

def get_racedays_for_period(
    date_from: str,
    date_to: str,
    countries: list[str] | None = None,
    only_finished: bool = True,
) -> list[dict]:
    """
    Henter alle raceday-metadata for en periode med ETT API-kall.

    Filterer bort:
    - Kombinasjonsloep (X1, country='')
    - Land utenfor oensket liste
    - Paagaaende/fremtidige loep (om only_finished=True)
    """
    if countries is None:
        countries = ["NO"]
    country_set = {c.upper() for c in countries}

    data = _get(f"{BASE}/results/racedays/{date_from}/{date_to}/list")
    if not data or "error" in data:
        return []

    out = []
    for day in data.get("result", []):
        for rd in day.get("raceDays", []):
            country = rd.get("countryIsoCode", "")
            if not country:
                continue
            if country not in country_set:
                continue
            status = rd.get("progressStatus", "")
            if only_finished and status != "Finished":
                continue
            out.append({
                "raceDay":        rd["raceDay"],
                "raceDayName":    rd.get("raceDayName", ""),
                "sportType":      rd.get("sportType", "T"),
                "countryIsoCode": country,
                "startTime":      rd.get("startTime", ""),
                "progressStatus": status,
                "date":           rd.get("startTime", "")[:10],
            })
    return out


# ── Rikstoto: enkelt raceday ──────────────────────────────────────────────────

def _starts(raceday_key: str) -> dict[str, list]:
    """Dict: raceNumber → list of runner-dicts."""
    data = _get(f"{BASE}/racedays/{raceday_key}/starts")
    if not data or "error" in data:
        return {}
    result = data.get("result", {})
    return {str(k): v for k, v in result.items()}


def _results(raceday_key: str) -> tuple[dict[str, list], dict[str, dict]]:
    """
    Returnerer:
      - results_by_race: raceNumber → [{place, startNumber}]
      - win_odds_by_race: raceNumber → {startNumber: odds}  (kun vinneren)
    """
    data = _get(f"{BASE}/results/racedays/{raceday_key}/raceresults")
    if not data or "error" in data:
        return {}, {}

    result = data.get("result", {})
    rr = result.get("raceResults", {})
    results_by_race = {str(k): v for k, v in rr.items()}

    # winOdds: {raceNum: {startNum: {odds, payoutStatus}}}
    win_odds_raw = result.get("finalOdds", {}).get("winOdds", {})
    win_odds_by_race: dict[str, dict] = {}
    for race_num, starters in win_odds_raw.items():
        # Kun én startNumber per løp (vinneren)
        for start_num_str, odds_data in starters.items():
            odds_val = odds_data.get("odds")
            if odds_val is not None:
                win_odds_by_race[str(race_num)] = {
                    "start_num": int(start_num_str),
                    "odds":      float(odds_val),
                }

    return results_by_race, win_odds_by_race


def _sport_to_blood(sport_type: str, track: str = "") -> str:
    if sport_type == "G":
        return "galopp"
    return "varmblod"


def fetch_raceday(meta: dict) -> list[dict]:
    """Henter starter + resultater for ett rikstoto-raceday."""
    rdk  = meta["raceDay"]
    date = meta["date"]

    starts_by_race              = _starts(rdk)
    results_by_race, win_odds_by_race = _results(rdk)

    races = []
    for race_num_str, runners in starts_by_race.items():
        result_list = results_by_race.get(race_num_str, [])
        place_map   = {str(r["startNumber"]): r["place"] for r in result_list}

        # Vinnerodd for dette løpet (kun for vinneren)
        wo_info = win_odds_by_race.get(race_num_str)  # {"start_num": N, "odds": X}

        race_id = f"rikstoto_{rdk}_{race_num_str}"
        race = {
            "source":     "rikstoto",
            "race_id":    race_id,
            "date":       date,
            "track":      meta["raceDayName"],
            "distance":   None,
            "surface":    "grus",
            "race_type":  "galopp" if meta["sportType"] == "G" else "trav",
            "blood_type": _sport_to_blood(meta["sportType"], meta["raceDayName"]),
            "results":    [],
        }

        for runner in runners:
            sn       = str(runner.get("startNumber", ""))
            sn_int   = runner.get("startNumber")
            pos      = place_map.get(sn)

            # Vinnerodd: kun sett på hesten som faktisk vant
            win_odds = None
            if wo_info and sn_int == wo_info["start_num"]:
                win_odds = wo_info["odds"]

            race["results"].append({
                "horse_name":    runner.get("horseName", ""),
                "position":      pos,
                "start_pos":     sn_int,
                "jockey":        runner.get("driverName", ""),
                "trainer":       "",
                "odds":          None,
                "time_sec":      None,
                "scratched":     1 if runner.get("isScratched") else 0,
                "extra_distance": runner.get("extraDistance", 0) or 0,
                "win_odds":      win_odds,
                "horse_reg_no":  runner.get("horseRegistrationNumber"),
            })

        if race["results"]:
            races.append(race)

    return races


# ── Rikstoto: enkeltdato-henting ──────────────────────────────────────────────

def fetch_rikstoto_results(
    date: str | None = None,
    countries: list[str] | None = None,
) -> list[dict]:
    """Henter alle avsluttede loep for en dato via Rikstoto API."""
    if date is None:
        date = datetime.date.today().isoformat()
    if countries is None:
        countries = ["NO"]

    raceday_metas = get_racedays_for_period(date, date, countries)
    if not raceday_metas:
        return [{"error": f"Ingen avsluttede loep funnet for {date}", "source": "rikstoto"}]

    races = []
    for meta in raceday_metas:
        races.extend(fetch_raceday(meta))
        time.sleep(0.3)

    return races if races else [{"error": f"Ingen loep hentet for {date}", "source": "rikstoto"}]


# ── Travsport.no ──────────────────────────────────────────────────────────────

TRAVSPORT_BASE = "https://www.travsport.no"
TRAVSPORT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":  "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.travsport.no/",
}


def _ts_get(url: str) -> str | None:
    """Henter HTML fra travsport.no."""
    try:
        r = requests.get(url, headers=TRAVSPORT_HEADERS, timeout=20)
        r.raise_for_status()
        r.encoding = "utf-8"
        return r.text
    except Exception:
        return None


def _parse_time_sec(time_str: str) -> float | None:
    """Konverterer '2.56,2' eller '1.57,2' til sekunder (176.2 / 117.2)."""
    if not time_str:
        return None
    s = time_str.strip().replace("\xa0", "")
    m = re.match(r"^(\d+)\.(\d{2}),(\d+)$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10.0
    return None


def _parse_position(pos_str: str) -> int | None:
    """'1' → 1, '0' → None, 'Disk.' → None, 'Strøket' → None."""
    s = pos_str.strip()
    if not s:
        return None
    if s.lower().startswith("disk") or s.lower().startswith("str"):
        return None
    try:
        n = int(s)
        return n if n > 0 else None
    except ValueError:
        return None


def _strip_license(name: str) -> str:
    """Fjerner kuskelisens-suffiks: 'Vidar Hop (T)' → 'Vidar Hop'."""
    return re.sub(r"\s*\([A-Z]\)\s*$", "", name).strip()


def get_travsport_result_urls(date_from: str, date_to: str) -> list[dict]:
    """
    Henter alle travsport.no-resultat-URLer for perioden.
    Henter kalender maaned for maaned, returnerer metadata-dicts.
    """
    d_from = datetime.date.fromisoformat(date_from)
    d_to   = datetime.date.fromisoformat(date_to)

    # Samle alle aar/maaned-kombinasjoner
    months: set[tuple[int, int]] = set()
    cur = datetime.date(d_from.year, d_from.month, 1)
    end = datetime.date(d_to.year,   d_to.month,   1)
    while cur <= end:
        months.add((cur.year, cur.month))
        if cur.month == 12:
            cur = datetime.date(cur.year + 1, 1, 1)
        else:
            cur = datetime.date(cur.year, cur.month + 1, 1)

    result_metas: list[dict] = []
    seen: set[str] = set()

    for year, month in sorted(months):
        html = _ts_get(
            f"{TRAVSPORT_BASE}/sportsbasen/lopskalender/?year={year}&month={month}"
        )
        if not html:
            time.sleep(0.5)
            continue

        # Finn alle result-lenker i kalendersiden
        pattern = re.compile(
            r'href="(https://www\.travsport\.no/travbaner/([^/]+)/results/(\d{4}-\d{2}-\d{2}))"'
        )
        for hit in pattern.finditer(html):
            url        = hit.group(1)
            track_slug = hit.group(2)
            date       = hit.group(3)
            if date < date_from or date > date_to:
                continue
            if url in seen:
                continue
            seen.add(url)
            result_metas.append({
                "url":        url,
                "date":       date,
                "track_slug": track_slug,
                "track_name": track_slug.replace("-", " ").title(),
            })

        time.sleep(0.4)

    return result_metas


def fetch_travsport_raceday(meta: dict) -> list[dict]:
    """
    Henter og parser ett travsport.no-resultatside.
    Returnerer liste med loep-dicts (kompatibelt med _save_races).

    Gir bloddtype (kaldblod/varmblod), distanse, tid og trener –
    data som rikstoto-APIet ikke tilbyr.
    """
    url   = meta["url"]
    date  = meta["date"]
    track = meta["track_name"]
    slug  = meta.get("track_slug", "")

    html = _ts_get(url)
    if not html:
        return []

    soup   = BeautifulSoup(html, "html.parser")
    panels = soup.select("div.js-tabbedContent-panel")

    races = []
    for panel in panels:
        panel_id = panel.get("id", "")
        if not panel_id.startswith("race-"):
            continue
        race_num = panel_id.split("-", 1)[1]   # "1", "2" …

        # ── Bloddtype ──────────────────────────────────────────────────────
        # "Kaldblods"/"Varmblods" er tekst-noder like etter <h2> i HTML-en,
        # ikke INNE i h2 – saa vi sjekker naerliggende tekst i panelet.
        blood_type = "varmblod"
        h2 = panel.find("h2")
        nearby_text = ""
        if h2:
            # Sjekk tekst i h2 + dens naeste soskennoder (tekst/tagger)
            nearby_text = h2.get_text()
            for sib in h2.next_siblings:
                chunk = sib.get_text() if hasattr(sib, "get_text") else str(sib)
                nearby_text += " " + chunk
                if len(nearby_text) > 200:
                    break
        # Fallback: hele panelteksten de foerste 300 tegn
        if not nearby_text:
            nearby_text = panel.get_text()[:300]
        if re.search(r"kaldblod", nearby_text, re.I):
            blood_type = "kaldblod"
        elif re.search(r"varmblod", nearby_text, re.I):
            blood_type = "varmblod"

        # ── Resultatstabell ─────────────────────────────────────────────────
        table = panel.find("table", class_="results")
        if not table:
            continue

        distance = None
        results  = []

        for tbody in table.find_all("tbody"):
            rows = tbody.find_all("tr", recursive=False)
            if not rows:
                continue
            main_row = rows[0]
            cells    = main_row.find_all(["td", "th"], recursive=False)
            if len(cells) < 8:
                continue

            # Plassering
            pos_span = cells[0].find("span")
            pos_str  = pos_span.get_text(strip=True) if pos_span else cells[0].get_text(strip=True)
            position = _parse_position(pos_str)

            # Startnummer
            try:
                start_pos = int(cells[1].get_text(strip=True))
            except (ValueError, IndexError):
                start_pos = None

            # Hestenavn
            horse_a    = cells[2].find("a")
            horse_name = (
                horse_a.get("data-name") or horse_a.get_text(strip=True)
                if horse_a else cells[2].get_text(strip=True)
            )
            if not horse_name or len(horse_name) < 2:
                continue

            # Distanse (hent en gang fra foerste loeper)
            try:
                dist_val = int(cells[3].get_text(strip=True))
                if distance is None:
                    distance = dist_val
            except (ValueError, IndexError):
                pass

            # Tid
            time_sec = _parse_time_sec(cells[4].get_text(strip=True))

            # Kusk
            jockey_cell = cells[7] if len(cells) > 7 else None
            if jockey_cell:
                ja = jockey_cell.find("a")
                jockey = _strip_license(
                    ja.get("data-name") or ja.get_text(strip=True)
                    if ja else jockey_cell.get_text(strip=True)
                )
            else:
                jockey = ""

            # Odds
            odds = None
            if len(cells) > 10:
                try:
                    odds_str = cells[10].get_text(strip=True).replace("\xa0", "").replace(" ", "")
                    odds = float(odds_str) if odds_str else None
                except ValueError:
                    pass

            # Trener (fra infoslekte-rad)
            trainer = ""
            if len(rows) > 1:
                for strong in rows[1].find_all("strong"):
                    if "Trener" in strong.get_text():
                        ta = strong.find_next("a")
                        if ta:
                            trainer = _strip_license(
                                ta.get("data-name") or ta.get_text(strip=True)
                            )
                        break

            results.append({
                "horse_name": horse_name,
                "position":   position,
                "start_pos":  start_pos,
                "jockey":     jockey,
                "trainer":    trainer,
                "odds":       odds,
                "time_sec":   time_sec,
                "scratched":  0,
            })

        if results:
            races.append({
                "source":     "travsport",
                "race_id":    f"travsport_{slug}_{date}_{race_num}",
                "date":       date,
                "track":      track,
                "distance":   distance,
                "surface":    "grus",
                "race_type":  "trav",
                "blood_type": blood_type,
                "results":    results,
            })

    return races


def fetch_travsport_results(
    date_from: str,
    date_to: str | None = None,
) -> list[dict]:
    """Henter alle travsport.no-resultater for en periode."""
    if date_to is None:
        date_to = date_from

    metas = get_travsport_result_urls(date_from, date_to)
    if not metas:
        return [{"error": f"Ingen travsport-resultater for {date_from}–{date_to}", "source": "travsport"}]

    all_races = []
    for meta in metas:
        races = fetch_travsport_raceday(meta)
        all_races.extend(races)
        time.sleep(0.5)

    return all_races if all_races else [{"error": "Ingen loep hentet fra travsport.no", "source": "travsport"}]


# ── Felles fetch_all ──────────────────────────────────────────────────────────

def fetch_all(
    date: str | None = None,
    sources: list[str] | None = None,
    countries: list[str] | None = None,
) -> dict:
    """Henter data fra valgte kilder for en enkeltdato."""
    if date is None:
        date = datetime.date.today().isoformat()
    if sources is None:
        sources = ["rikstoto"]      # default: kun rikstoto for enkel henting

    result = {}
    if "rikstoto" in sources:
        result["rikstoto"] = fetch_rikstoto_results(date, countries)
    if "travsport" in sources:
        result["travsport"] = fetch_travsport_results(date, date)
    return result
