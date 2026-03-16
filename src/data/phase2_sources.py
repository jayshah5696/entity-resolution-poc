"""
Phase 2 data source download and parsing.

Downloads and parses real-world data sources for cross-encoder training:
  - GLEIF Golden Copy (company names + aliases)
  - SEC EDGAR (company names + former names)
  - O*NET 29.0 (job titles + alternates)
  - US Census 2010 (frequency-weighted surnames)
  - SSA Baby Names (frequency-weighted first names)
  - nicknames PyPI (name → nickname mappings)

Usage:
    python src/data/phase2_sources.py --output-dir data/phase2/raw/
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
app = typer.Typer(help="Download and parse Phase 2 data sources.")


# ---------------------------------------------------------------------------
# GLEIF Golden Copy
# ---------------------------------------------------------------------------

GLEIF_CSV_URL = (
    "https://goldencopy.gleif.org/api/v2/golden-copies/publishes/lei2/latest.csv"
)


def parse_gleif(path: Path | str) -> pl.DataFrame:
    """Parse a GLEIF golden copy CSV into a clean DataFrame.

    Parameters
    ----------
    path : Path
        Path to the GLEIF CSV file (local).

    Returns
    -------
    pl.DataFrame with columns: legal_name, other_names, country, entity_status
    """
    path = Path(path)
    df = pl.read_csv(
        path,
        ignore_errors=True,
        truncate_ragged_lines=True,
    )

    # Map GLEIF columns to our schema
    col_map = {
        "Entity.LegalName": "legal_name",
        "Entity.LegalAddress.Country": "country",
        "Entity.EntityStatus": "entity_status",
    }

    # Rename columns that exist
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(rename)

    # Parse OtherEntityNames JSON array
    other_names_col = "Entity.OtherEntityNames"
    if other_names_col in df.columns:
        df = df.rename({other_names_col: "other_names_raw"})
    elif "other_names_raw" not in df.columns:
        df = df.with_columns(pl.lit("[]").alias("other_names_raw"))

    # Parse JSON arrays from the other_names_raw column
    def _parse_other_names(raw: str | None) -> list[str]:
        if not raw or raw == "[]" or raw == "":
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
            return []
        except (json.JSONDecodeError, TypeError):
            return []

    other_names_parsed = [
        _parse_other_names(v) for v in df["other_names_raw"].to_list()
    ]
    df = df.with_columns(pl.Series("other_names", other_names_parsed))

    # Select final columns
    cols = ["legal_name", "other_names", "country", "entity_status"]
    available = [c for c in cols if c in df.columns]
    return df.select(available)


def download_gleif(output_dir: Path) -> Path:
    """Download GLEIF golden copy CSV."""
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gleif_golden_copy.csv"

    if output_path.exists():
        console.print(f"[yellow]GLEIF already downloaded: {output_path}")
        return output_path

    console.print("[cyan]Downloading GLEIF golden copy CSV...")
    resp = requests.get(GLEIF_CSV_URL, allow_redirects=True, timeout=300)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    console.print(f"[green]GLEIF saved: {output_path} ({len(resp.content) / 1e6:.1f} MB)")
    return output_path


# ---------------------------------------------------------------------------
# O*NET
# ---------------------------------------------------------------------------

ONET_DB_URL = "https://www.onetcenter.org/dl_files/database/db_29_0_text.zip"


def parse_onet_alternates(path: Path | str) -> dict[str, list[str]]:
    """Parse O*NET alternate titles file.

    Returns
    -------
    dict mapping canonical title → list of alternate titles
    """
    path = Path(path)
    df = pl.read_csv(path, separator="\t", ignore_errors=True)

    # Column names: O*NET-SOC Code, Title, Alternate Title, Short Title, Sources
    title_col = "Title"
    alt_col = "Alternate Title"

    if title_col not in df.columns or alt_col not in df.columns:
        # Try alternate column names
        for col in df.columns:
            if "title" in col.lower() and "alternate" not in col.lower():
                title_col = col
            elif "alternate" in col.lower():
                alt_col = col

    result: dict[str, list[str]] = {}
    for row in df.iter_rows(named=True):
        canonical = row.get(title_col, "")
        alt = row.get(alt_col, "")
        if canonical and alt:
            result.setdefault(canonical, []).append(alt)

    return result


def parse_onet_reported(path: Path | str) -> list[str]:
    """Parse O*NET reported titles file.

    Returns
    -------
    list of real-world reported job titles
    """
    path = Path(path)
    df = pl.read_csv(path, separator="\t", ignore_errors=True)

    # Find the reported title column
    reported_col = None
    for col in df.columns:
        if "reported" in col.lower():
            reported_col = col
            break

    if reported_col is None:
        raise ValueError(f"No 'Reported' column found. Columns: {df.columns}")

    return df[reported_col].drop_nulls().unique().to_list()


def download_onet(output_dir: Path) -> tuple[Path, Path]:
    """Download O*NET database and extract title files."""
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alt_path = output_dir / "onet_alternate_titles.txt"
    reported_path = output_dir / "onet_reported_titles.txt"

    if alt_path.exists() and reported_path.exists():
        console.print(f"[yellow]O*NET already downloaded: {output_dir}")
        return alt_path, reported_path

    console.print("[cyan]Downloading O*NET 29.0 database...")
    resp = requests.get(ONET_DB_URL, timeout=300)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        # Find the alternate titles and reported titles files
        for name in z.namelist():
            if "Alternate Title" in name and name.endswith(".txt"):
                with z.open(name) as f:
                    alt_path.write_bytes(f.read())
            elif "Sample of Reported" in name and name.endswith(".txt"):
                with z.open(name) as f:
                    reported_path.write_bytes(f.read())

    console.print(f"[green]O*NET titles extracted to {output_dir}")
    return alt_path, reported_path


# ---------------------------------------------------------------------------
# US Census 2010 Surnames
# ---------------------------------------------------------------------------

CENSUS_URL = "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"


def load_census_surnames(path: Path | str) -> pl.DataFrame:
    """Load Census 2010 surnames with frequency counts.

    Returns
    -------
    pl.DataFrame with columns: name, count
    """
    path = Path(path)
    df = pl.read_csv(path, ignore_errors=True, truncate_ragged_lines=True)

    # Standardize column names
    col_lower = {c: c.lower() for c in df.columns}
    df = df.rename(col_lower)

    # Ensure name column is title case
    if "name" in df.columns:
        df = df.with_columns(pl.col("name").str.to_titlecase())

    # Ensure count column exists
    if "count" not in df.columns:
        # Census file may have different count column name
        for col in df.columns:
            if "count" in col.lower() or "number" in col.lower():
                df = df.rename({col: "count"})
                break

    if "count" in df.columns:
        df = df.with_columns(pl.col("count").cast(pl.Int64))

    return df.select([c for c in ["name", "count", "rank"] if c in df.columns])


def download_census_surnames(output_dir: Path) -> Path:
    """Download Census 2010 surnames."""
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "census_surnames.csv"

    if output_path.exists():
        console.print(f"[yellow]Census surnames already downloaded: {output_path}")
        return output_path

    console.print("[cyan]Downloading Census 2010 surnames...")
    resp = requests.get(CENSUS_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                with z.open(name) as f:
                    output_path.write_bytes(f.read())
                break

    console.print(f"[green]Census surnames saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# SSA Baby Names
# ---------------------------------------------------------------------------

SSA_URL = "https://www.ssa.gov/oact/babynames/names.zip"


def load_ssa_names(path: Path | str) -> pl.DataFrame:
    """Load SSA baby names with frequency counts.

    For a pre-aggregated CSV (name, sex, count), reads directly.
    For raw SSA zip (yobYYYY.txt files), aggregates across years.

    Returns
    -------
    pl.DataFrame with columns: name, count
    """
    path = Path(path)

    if path.suffix == ".zip":
        return _load_ssa_from_zip(path)

    # Pre-aggregated CSV
    df = pl.read_csv(path, ignore_errors=True)
    col_lower = {c: c.lower() for c in df.columns}
    df = df.rename(col_lower)

    if "count" in df.columns:
        df = df.with_columns(pl.col("count").cast(pl.Int64))

    # Aggregate by name (combine M/F)
    if "sex" in df.columns:
        df = df.group_by("name").agg(pl.col("count").sum())

    return df.select([c for c in ["name", "count"] if c in df.columns])


def _load_ssa_from_zip(path: Path) -> pl.DataFrame:
    """Load SSA names from raw zip file with yobYYYY.txt files."""
    all_dfs = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.startswith("yob") and name.endswith(".txt"):
                year = int(name[3:7])
                if year < 1970:
                    continue
                with z.open(name) as f:
                    df = pl.read_csv(
                        f.read(),
                        has_header=False,
                        new_columns=["name", "sex", "count"],
                    )
                    all_dfs.append(df)

    if not all_dfs:
        return pl.DataFrame({"name": [], "count": []})

    combined = pl.concat(all_dfs)
    return combined.group_by("name").agg(pl.col("count").sum())


def download_ssa_names(output_dir: Path) -> Path:
    """Download SSA baby names zip."""
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ssa_names.zip"

    if output_path.exists():
        console.print(f"[yellow]SSA names already downloaded: {output_path}")
        return output_path

    console.print("[cyan]Downloading SSA baby names...")
    resp = requests.get(SSA_URL, timeout=120)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    console.print(f"[green]SSA names saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------

EDGAR_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
EDGAR_HEADERS = {"User-Agent": "EntityResolutionPOC jayshah5696@gmail.com"}


def parse_edgar_submission(data: dict) -> dict:
    """Parse a single EDGAR submission JSON object.

    Returns
    -------
    dict with keys: name, former_names, sic, tickers
    """
    name = data.get("name", "")
    former_names = []
    for fn in data.get("formerNames", []):
        fn_name = fn.get("name", "")
        if fn_name:
            former_names.append(fn_name)

    return {
        "name": name,
        "former_names": former_names,
        "sic": data.get("sic", ""),
        "tickers": data.get("tickers", []),
    }


def download_and_parse_edgar(output_dir: Path, max_companies: int = 10_000) -> pl.DataFrame:
    """Download EDGAR submissions.zip and extract company names.

    Parameters
    ----------
    output_dir : Path
        Where to save the downloaded zip.
    max_companies : int
        Maximum number of companies to parse (zip has ~800K files).

    Returns
    -------
    pl.DataFrame with columns: name, former_names
    """
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "edgar_submissions.zip"

    if not zip_path.exists():
        console.print("[cyan]Downloading SEC EDGAR submissions.zip (~1.8GB)...")
        resp = requests.get(
            EDGAR_URL, headers=EDGAR_HEADERS, timeout=600, stream=True
        )
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        console.print(f"[green]EDGAR saved: {zip_path}")

    # Parse company names from the zip
    companies = []
    with zipfile.ZipFile(zip_path) as z:
        count = 0
        for name in z.namelist():
            if not name.endswith(".json") or name.startswith("__"):
                continue
            try:
                with z.open(name) as f:
                    data = json.load(f)
                parsed = parse_edgar_submission(data)
                if parsed["name"]:
                    companies.append(parsed)
                    count += 1
                    if count >= max_companies:
                        break
            except (json.JSONDecodeError, KeyError):
                continue

    return pl.DataFrame(companies)


# ---------------------------------------------------------------------------
# Nicknames
# ---------------------------------------------------------------------------


def load_nicknames() -> dict[str, set[str]]:
    """Load nickname mappings from the nicknames PyPI package.

    Returns
    -------
    dict mapping formal name (lowercase) → set of nicknames
    """
    try:
        from nicknames import NickNamer

        nn = NickNamer()
        # Build a dict from the internal data
        result: dict[str, set[str]] = {}
        # NickNamer exposes nicknames_of() for formal→nicknames
        # and canonicals_of() for nickname→formal names
        # We want formal → set(nicknames)
        # Access internal data if available
        if hasattr(nn, "_nickname_to_canonical"):
            for nick, canonicals in nn._nickname_to_canonical.items():
                for canonical in canonicals:
                    canonical_lower = canonical.lower()
                    result.setdefault(canonical_lower, set()).add(nick.lower())
        elif hasattr(nn, "nicknames_of"):
            # Fallback: test common names
            common = [
                "william", "robert", "richard", "james", "john",
                "michael", "david", "thomas", "charles", "daniel",
                "elizabeth", "margaret", "catherine", "jennifer", "patricia",
            ]
            for name in common:
                nicks = nn.nicknames_of(name)
                if nicks:
                    result[name] = {n.lower() for n in nicks}
        return result
    except ImportError:
        console.print("[yellow]nicknames package not installed. Using fallback.")
        return _fallback_nicknames()


def _fallback_nicknames() -> dict[str, set[str]]:
    """Multi-ethnic nickname dict when the nicknames package isn't available.

    Covers English, Hispanic/Latino, South Asian, East Asian, Middle Eastern,
    Eastern European, and African name variants.
    """
    return {
        # English / Western European
        "william": {"bill", "will", "billy", "liam", "willy"},
        "robert": {"bob", "rob", "bobby", "robbie", "bert"},
        "richard": {"dick", "rick", "rich", "ricky"},
        "james": {"jim", "jimmy", "jamie"},
        "john": {"jack", "johnny", "jon"},
        "michael": {"mike", "mikey", "mick"},
        "elizabeth": {"liz", "beth", "betty", "eliza", "lizzy"},
        "margaret": {"maggie", "meg", "peggy", "marge"},
        "catherine": {"kate", "cathy", "cat", "katie"},
        "jennifer": {"jen", "jenny"},
        "patricia": {"pat", "patty", "trish"},
        "thomas": {"tom", "tommy"},
        "charles": {"charlie", "chuck", "chas"},
        "daniel": {"dan", "danny"},
        "david": {"dave", "davy"},
        "joseph": {"joe", "joey"},
        "edward": {"ed", "eddie", "ted", "teddy"},
        "anthony": {"tony"},
        "christopher": {"chris"},
        "benjamin": {"ben"},
        "alexander": {"alex", "sasha"},
        "andrew": {"andy", "drew"},
        "matthew": {"matt"},
        "nicholas": {"nick", "nico"},
        "timothy": {"tim"},
        "jonathan": {"jon"},
        "stephen": {"steve"},
        "gregory": {"greg"},
        "samuel": {"sam"},
        "raymond": {"ray"},
        "lawrence": {"larry"},
        # Hispanic / Latino
        "alejandro": {"alex", "ale"},
        "francisco": {"paco", "pancho", "frank"},
        "guadalupe": {"lupe"},
        "josé": {"pepe", "jose", "che"},
        "jose": {"pepe", "che"},
        "jesús": {"chucho", "chuy", "jesus"},
        "jesus": {"chucho", "chuy"},
        "concepción": {"concha", "conchita"},
        "fernando": {"nando", "fer"},
        "guillermo": {"memo"},
        "enrique": {"quique", "kike"},
        "roberto": {"beto"},
        "eduardo": {"lalo", "edu"},
        "alberto": {"beto", "al"},
        "rafael": {"rafa"},
        "ignacio": {"nacho"},
        "dolores": {"lola", "lolita"},
        "rosario": {"charo"},
        "mercedes": {"meche"},
        # South Asian
        "rajesh": {"raj"},
        "suresh": {"suri"},
        "mahesh": {"mahi"},
        "ramesh": {"ram"},
        "prakash": {"pk"},
        "krishna": {"krish"},
        "lakshmi": {"lucky"},
        "priyanka": {"priya"},
        "abhishek": {"abhi"},
        "siddharth": {"sid"},
        "harshvardhan": {"harsh"},
        "devendra": {"dev"},
        "mohammed": {"mo", "mohammad", "muhammed"},
        "muhammad": {"mo", "mohammad"},
        "abdulrahman": {"abdul", "rahman"},
        "subramaniam": {"subbu"},
        "venkatesh": {"venky"},
        # East Asian
        "takeshi": {"take"},
        "yoshiko": {"yoshi"},
        "hiroshi": {"hiro"},
        "kazuki": {"kazu"},
        "yusuke": {"yu"},
        "xiaoming": {"xiao"},
        "jianwei": {"jian"},
        "guangming": {"guang"},
        # Middle Eastern / Arabic
        "fatima": {"fati"},
        "ahmed": {"ahmad"},
        "ibrahim": {"abe", "brahim"},
        "mustafa": {"musti"},
        "hussein": {"hussain"},
        "abdallah": {"abd", "abdullah"},
        # Eastern European / Russian
        "aleksander": {"sasha", "alex"},
        "dmitri": {"dima", "mitya"},
        "vladimir": {"vlad", "vova"},
        "ekaterina": {"katya", "kate"},
        "anastasia": {"nastya"},
        "nikolai": {"kolya", "nico"},
        "mikhail": {"misha"},
        "svetlana": {"sveta"},
        # African
        "oluwaseun": {"seun"},
        "oluwafemi": {"femi"},
        "oluwadamilola": {"dami"},
        "chukwuemeka": {"emeka"},
        "nnamdi": {"nam"},
        "babatunde": {"tunde"},
        "olayinka": {"yinka"},
        "oluwatobi": {"tobi"},
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def download_all(
    output_dir: Path = typer.Option("data/phase2/raw", help="Output directory for raw data"),
):
    """Download all Phase 2 data sources."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}")) as progress:
        task = progress.add_task("Downloading data sources...", total=5)

        # 1. GLEIF
        progress.update(task, description="Downloading GLEIF golden copy...")
        download_gleif(output_dir)
        progress.advance(task)

        # 2. O*NET
        progress.update(task, description="Downloading O*NET 29.0...")
        download_onet(output_dir)
        progress.advance(task)

        # 3. Census surnames
        progress.update(task, description="Downloading Census 2010 surnames...")
        download_census_surnames(output_dir)
        progress.advance(task)

        # 4. SSA baby names
        progress.update(task, description="Downloading SSA baby names...")
        download_ssa_names(output_dir)
        progress.advance(task)

        # 5. EDGAR
        progress.update(task, description="Downloading SEC EDGAR submissions...")
        download_and_parse_edgar(output_dir)
        progress.advance(task)

    console.print("[bold green]All data sources downloaded successfully!")


if __name__ == "__main__":
    app()
