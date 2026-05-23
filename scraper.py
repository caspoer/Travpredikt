"""
Scraper mot Rikstoto JSON-API, travsport.no HTML og ATG (svensk) API.

Rikstoto-endepunkter:
  - /api/results/racedays/{from}/{to}/list   → racedays i en periode
  - /api/racedays/{raceday_key}/starts       → startliste (hester + kusk)
  - /api/results/racedays/{raceday_key}/raceresults → plassering + startnr

Travsport.no:
  - /sportsbasen/lopskalender/?year=Y&month=M → liste over stevner med resultater
  - /travbaner/{track-slug}/results/{date}    → resultatsider (HTML)
  Gir: bloddtype (kald/varm), distanse, tid, trener, odds – data rikstoto mangler.

ATG (Aktiebolaget Trav och Galopp, Sverige):
  - horse-betting-info.prod.c1.atg.cloud/api-public/v0/calendar/day/{date}
      → alle loep med race-IDer for en dag
  - horse-betting-info.prod.c1.atg.cloud/api-public/v0/races/{race_id}
      → komplett loep med startere, resultater og heste-statistikk
  Gir: heste-ID, alder, kjoenn, trekkmerker, livstidsinntekter, trener-statistikk,
       personlige rekorder per underlag – data som ingen av de andre kildene har.
"""
import datetime
import re
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _build_session(headers: dict, pool_size: int = 20) -> requests.Session:
    """
    Felles HTTP-session med:
      - Keep-alive (gjenbruker TCP+TLS-koblinger → ~1.5-2x raskere per kall)
      - Connection pool på 20 (nok for vår parallellitet)
      - Auto-retry på midlertidige feil (502/503/504)
    """
    s = requests.Session()
    s.headers.update(headers)
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=retry,
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

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

SOURCES = {"rikstoto": None, "travsport": None, "atg": None}

# Felles session for alle Rikstoto-kall (HTTP keep-alive)
_RIKSTOTO_SESSION = _build_session(HEADERS)


def _get(url: str) -> dict | None:
    try:
        r = _RIKSTOTO_SESSION.get(url, timeout=TIMEOUT)
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


_TRAVSPORT_SESSION = _build_session(TRAVSPORT_HEADERS)


def _ts_get(url: str) -> str | None:
    """Henter HTML fra travsport.no."""
    try:
        r = _TRAVSPORT_SESSION.get(url, timeout=20)
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


# ── ATG (Sverige) ────────────────────────────────────────────────────────────

ATG_BASE = "https://horse-betting-info.prod.c1.atg.cloud/api-public/v0"
ATG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":  "application/json, */*",
    "Referer": "https://www.atg.se/",
    "Origin":  "https://www.atg.se",
}

# Hvilke land vi henter fra ATG (SE er kjerne, DK kan inkluderes)
ATG_COUNTRIES = {"SE"}

_ATG_SESSION = _build_session(ATG_HEADERS)


def _atg_get(url: str) -> dict | None:
    try:
        r = _ATG_SESSION.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _atg_blood_type(race: dict) -> str:
    """
    Utleder bloddtype fra løpsnavn, termer og sport-kode.
    ATG bruker 'kallblod' (kaldblod) i løpsnavn/termer for kaldblodshester.
    """
    sport = race.get("sport", "").lower()
    if sport == "gallop":
        return "galopp"

    # Sjekk løpsnavn og termer for 'kallblod'
    text = race.get("name", "") + " ".join(race.get("terms", []))
    if re.search(r"kallblod", text, re.I):
        return "kaldblod"

    return "varmblod"


def _atg_time_sec(time_dict: dict | None) -> float | None:
    """Konverterer ATG time-dict {minutes, seconds, tenths} til sekunder."""
    if not time_dict:
        return None
    try:
        return (int(time_dict.get("minutes", 0)) * 60
                + int(time_dict.get("seconds", 0))
                + int(time_dict.get("tenths", 0)) / 10.0)
    except (TypeError, ValueError):
        return None


def _atg_sport_to_race_type(sport: str) -> str:
    mapping = {"trot": "trav", "gallop": "galopp", "monte": "monté"}
    return mapping.get(sport.lower(), "trav")


def _atg_position(finish_order: int | None, n_starters: int) -> int | None:
    """
    ATG bruker høye tall (38, 56) for diskvalifiserte/strøkne.
    Behandler kun reelle plasseringer (1..n_starters) som gyldige.
    """
    if finish_order is None:
        return None
    if 1 <= finish_order <= n_starters:
        return finish_order
    return None   # disq / DNF / scratch


def _atg_extract_horse_stats(horse: dict) -> dict | None:
    """
    Trekker ut beriket horse-statistikk fra ATG sin `horse`-struktur.
    Returnerer dict med felt til horse_stats_atg-tabellen, eller None hvis ID mangler.
    """
    reg_no = horse.get("id")
    if not reg_no:
        return None

    stats     = horse.get("statistics", {})
    life      = stats.get("life", {})
    placement = life.get("placement", {})
    pedigree  = horse.get("pedigree", {})

    # Vinst/plass i prosent: ATG bruker promille (8200 = 82.00%)
    # → bygg om til prosent (0–100)
    def _pct(v):
        if v is None:
            return None
        try:
            return round(float(v) / 100.0, 2)
        except (TypeError, ValueError):
            return None

    # Personlig rekord: finn raskeste sekunder/km på tvers av alle records
    best_kmt = None
    for rec in life.get("records", []):
        t = _atg_time_sec(rec.get("time"))
        if t is None or t <= 0:
            continue
        # ATG-tid er sek/km direkte ("1.13,0" = 1 min 13.0 sek per km)
        if best_kmt is None or t < best_kmt:
            best_kmt = t

    return {
        "horse_reg_no":   str(reg_no),
        "name":           horse.get("name", ""),
        "age":            horse.get("age"),
        "sex":            horse.get("sex"),
        "color":          horse.get("color"),
        "money":          life.get("earnings") or horse.get("money"),
        "life_starts":    life.get("starts"),
        "life_wins":      placement.get("1"),
        "life_2nd":       placement.get("2"),
        "life_3rd":       placement.get("3"),
        "life_win_pct":   _pct(life.get("winPercentage")),
        "life_place_pct": _pct(life.get("placePercentage")),
        "best_time_sec":  best_kmt,
        "father_name":    (pedigree.get("father") or {}).get("name"),
        "mother_name":    (pedigree.get("mother") or {}).get("name"),
    }


def fetch_atg_race(race_id: str) -> dict | None:
    """
    Henter ett ATG-løp og returnerer et race-dict kompatibelt med _save_races.

    Bruker /services/racinginfo/v1/api/games/vinnare_{id} fordi det gir oss
    ALT i én forespørsel:
      - Race-metadata (distanse, sport, bane, terms)
      - Komplette startere med horse-statistikk og pedigree
      - Pre-race vinnerodds per starter (NOK! markedsindikator)
      - Resultat-plassering for ferdige løp

    race_id har format '{dato}_{bane_id}_{loepsnr}', f.eks. '2026-05-23_16_1'.
    """
    url = f"https://www.atg.se/services/racinginfo/v1/api/games/vinnare_{race_id}"
    try:
        r = _ATG_SESSION.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    races = data.get("races", [])
    if not races:
        return None
    race = races[0]

    date      = race.get("date", race_id[:10])
    track     = race.get("track", {})
    starts    = race.get("starts", [])
    sport     = race.get("sport", "trot")
    n         = len(starts)

    # Berikelses-data (horse_stats_atg) for hver hest
    horse_stats: list[dict] = []
    results = []

    for s in starts:
        horse  = s.get("horse", {})
        driver = s.get("driver", {})
        res    = s.get("result", {})
        pools  = s.get("pools", {})

        # Navn
        horse_name = horse.get("name", "")
        if not horse_name:
            continue

        # Kusk
        jockey = " ".join(filter(None, [
            driver.get("firstName", ""),
            driver.get("lastName", ""),
        ])).strip()

        # Trener
        trainer_d = horse.get("trainer", {})
        trainer = " ".join(filter(None, [
            trainer_d.get("firstName", ""),
            trainer_d.get("lastName", ""),
        ])).strip()

        # Pre-race vinnerodds (ATG bruker hundredeler: 4192 → 41.92x)
        # Strøkne har odds=0, "ikke spillbar" har odds=9999
        odds_raw = (pools.get("vinnare") or {}).get("odds")
        odds = None
        if odds_raw is not None and 0 < odds_raw < 9999:
            odds = round(odds_raw / 100.0, 2)

        # Plassering (gyldig kun innenfor antall startere)
        position = _atg_position(res.get("finishOrder"), n)

        # Disq/scratched: hvis odds=0 og posisjon mangler eller > n
        scratched = 0
        finish = res.get("finishOrder")
        if odds_raw == 0 and (finish is None or finish > n):
            scratched = 1

        # Individuelle handikap-meter (forskjell mellom løpets distanse og hestens)
        race_dist  = race.get("distance")
        start_dist = s.get("distance")
        extra_dist = 0
        if race_dist and start_dist and start_dist > race_dist:
            extra_dist = start_dist - race_dist

        results.append({
            "horse_name":    horse_name,
            "position":      position,
            "start_pos":     s.get("number"),
            "jockey":        jockey,
            "trainer":       trainer,
            "odds":          odds,             # PRE-race vinnerodds
            "time_sec":      None,             # ATG gir ikke faktisk løpstid
            "scratched":     scratched,
            "extra_distance": extra_dist,
            "win_odds":      None,             # post-race utbetaling sett separat
            "horse_reg_no":  str(horse.get("id")) if horse.get("id") else None,
        })

        # Berik hesteprofilen
        hs = _atg_extract_horse_stats(horse)
        if hs:
            horse_stats.append(hs)

    if not results:
        return None

    # Post-race utbetalt vinnerodds: pools.vinnare.result.winners[0]
    race_pools = race.get("pools", {})
    winner_info = (race_pools.get("vinnare") or {}).get("result", {}).get("winners", [])
    if winner_info:
        w_num  = winner_info[0].get("number")
        w_odds = winner_info[0].get("odds")
        if w_num is not None and w_odds is not None:
            w_odds_dec = round(float(w_odds) / 100.0, 2)
            for r in results:
                if r["start_pos"] == w_num:
                    r["win_odds"] = w_odds_dec
                    break

    return {
        "source":      "atg",
        "race_id":     f"atg_{race_id}",
        "date":        date,
        "track":       track.get("name", ""),
        "distance":    race.get("distance"),
        "surface":     "grus",
        "race_type":   _atg_sport_to_race_type(sport),
        "blood_type":  _atg_blood_type(race),
        "results":     results,
        "horse_stats": horse_stats,    # eget felt - lagres separat
    }


def get_atg_race_ids(
    date: str,
    countries: set | None = None,
    only_finished: bool = True,
) -> list[str]:
    """
    Henter alle race-IDer for en dato fra ATG.
    Filtrerer på land (default: kun SE) og status.
    only_finished=True henter bare ferdige løp (status='results').
    """
    if countries is None:
        countries = ATG_COUNTRIES

    data = _atg_get(f"{ATG_BASE}/calendar/day/{date}?headToHeadEnabled=true")
    if not data:
        return []

    ids = []
    for track in data.get("tracks", []):
        country = track.get("countryCode", "")
        if country not in countries:
            continue
        for race in track.get("races", []):
            status = race.get("status", "")
            if only_finished and status != "results":
                continue
            ids.append(race["id"])
    return ids


def fetch_atg_races_parallel(
    race_ids:    list[str],
    max_workers: int = 5,
    progress_cb=None,
) -> list[dict]:
    """
    Henter mange ATG-løp parallelt med ThreadPoolExecutor.
    max_workers=5 er trygt for ATG-APIet (10+ kan trigger throttling).
    progress_cb(done, total, race) kalles etter hvert ferdig løp.
    """
    if not race_ids:
        return []

    results = []
    done = 0
    total = len(race_ids)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(fetch_atg_race, rid): rid for rid in race_ids}
        for fut in as_completed(future_map):
            race = fut.result()
            done += 1
            if race:
                results.append(race)
            if progress_cb:
                try:
                    progress_cb(done, total, race)
                except Exception:
                    pass
    return results


def fetch_rikstoto_racedays_parallel(
    raceday_metas: list[dict],
    max_workers:   int = 4,
    progress_cb=None,
) -> list[dict]:
    """Henter mange Rikstoto-racedays parallelt (hver gjør 2 API-kall internt)."""
    if not raceday_metas:
        return []

    all_races = []
    done = 0
    total = len(raceday_metas)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(fetch_raceday, m): m for m in raceday_metas}
        for fut in as_completed(future_map):
            races = fut.result() or []
            done += 1
            all_races.extend(races)
            if progress_cb:
                try:
                    progress_cb(done, total, races)
                except Exception:
                    pass
    return all_races


def fetch_atg_results(
    date_from: str,
    date_to: str | None = None,
    countries: set | None = None,
) -> list[dict]:
    """
    Henter alle ATG-resultater for en periode.
    Standard: kun svenske løp (SE).
    """
    if date_to is None:
        date_to = date_from
    if countries is None:
        countries = ATG_COUNTRIES

    d_from = datetime.date.fromisoformat(date_from)
    d_to   = datetime.date.fromisoformat(date_to)

    all_races = []
    cur = d_from
    while cur <= d_to:
        date_str = cur.isoformat()
        race_ids = get_atg_race_ids(date_str, countries)

        for rid in race_ids:
            race = fetch_atg_race(rid)
            if race:
                all_races.append(race)
            time.sleep(0.2)

        if race_ids:
            time.sleep(0.4)

        cur += datetime.timedelta(days=1)

    if not all_races:
        return [{"error": f"Ingen ATG-resultater for {date_from}–{date_to}", "source": "atg"}]
    return all_races


def get_atg_upcoming_ids(
    date_from: str,
    date_to: str,
    countries: set | None = None,
) -> list[dict]:
    """
    Henter race-IDer for KOMMENDE løp (status != results) for en periode.
    Returnerer [{race_id, status, start_time}] sortert kronologisk.
    """
    if countries is None:
        countries = ATG_COUNTRIES

    d_from = datetime.date.fromisoformat(date_from)
    d_to   = datetime.date.fromisoformat(date_to)

    out = []
    cur = d_from
    while cur <= d_to:
        date_str = cur.isoformat()
        data = _atg_get(f"{ATG_BASE}/calendar/day/{date_str}?headToHeadEnabled=true")
        if data:
            for track in data.get("tracks", []):
                if track.get("countryCode", "") not in countries:
                    continue
                for race in track.get("races", []):
                    status = race.get("status", "")
                    if status != "results":
                        out.append({
                            "race_id":    race["id"],
                            "status":     status,
                            "track":      track.get("name", ""),
                            "race_num":   race.get("number"),
                            "start_time": race.get("startTime", ""),
                        })
        cur += datetime.timedelta(days=1)
        time.sleep(0.2)

    out.sort(key=lambda x: x["start_time"])
    return out


def fetch_atg_upcoming(
    date_from:   str,
    date_to:     str | None = None,
    countries:   set | None = None,
    max_workers: int = 10,
) -> list[dict]:
    """
    Henter kommende ATG-løp med startere, pre-race odds og horse-stats.
    Bruker parallell henting (5 workers default) for rask respons.
    """
    if date_to is None:
        # Standard: dagens dato + 1 dag (vanlig spillevindu)
        d_from = datetime.date.fromisoformat(date_from)
        date_to = (d_from + datetime.timedelta(days=1)).isoformat()

    metas = get_atg_upcoming_ids(date_from, date_to, countries)
    if not metas:
        return [{"error": f"Ingen kommende ATG-løp for {date_from}–{date_to}", "source": "atg"}]

    # Behold meta-info per race_id for å sette status + start_time etterpå
    meta_by_id = {m["race_id"]: m for m in metas}
    race_ids   = [m["race_id"] for m in metas]

    races = fetch_atg_races_parallel(race_ids, max_workers=max_workers)

    # Berik med meta-info
    for race in races:
        rid = race["race_id"].replace("atg_", "")
        m = meta_by_id.get(rid)
        if m:
            race["status"]     = m["status"]
            race["start_time"] = m["start_time"]
    return races


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
    if "atg" in sources:
        # ATG bruker eget land-sett (SE som standard)
        atg_countries = set(countries) & ATG_COUNTRIES if countries else ATG_COUNTRIES
        result["atg"] = fetch_atg_results(date, date, atg_countries or ATG_COUNTRIES)
    return result
