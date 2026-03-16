"""
Phase 2 base entity pool builder.

Assembles 50K entity records from real data sources:
  - Company names from GLEIF / SEC EDGAR
  - Surnames from US Census 2010 (frequency-weighted)
  - First names from SSA Baby Names (frequency-weighted)
  - Job titles from O*NET reported titles

Usage:
    python src/data/phase2_pool.py --source-dir data/phase2/raw/ --output-dir data/phase2/ --n 50000
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import polars as pl
import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Build Phase 2 base entity pool.")

PERSONAL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]

EMAIL_PATTERNS = [
    "firstname.lastname",  # 60%
    "f.lastname",  # 20%
    "flastname",  # 10%
    "firstname",  # 10% (personal domain)
]


# ---------------------------------------------------------------------------
# Company pool
# ---------------------------------------------------------------------------


def build_company_pool(
    gleif_df: pl.DataFrame,
    edgar_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Combine GLEIF + EDGAR companies into a pool.

    Returns DataFrame with columns: company, country, company_canonical
    """
    companies = []

    # GLEIF companies
    for row in gleif_df.iter_rows(named=True):
        name = row.get("legal_name", "")
        if not name:
            continue
        country = row.get("country", "US")
        canonical = _canonicalize_company(name)
        companies.append({"company": name, "country": country, "company_canonical": canonical})

    # EDGAR companies
    if edgar_df is not None:
        for row in edgar_df.iter_rows(named=True):
            name = row.get("name", "")
            if not name:
                continue
            canonical = _canonicalize_company(name)
            companies.append({"company": name, "country": "US", "company_canonical": canonical})

    if not companies:
        return pl.DataFrame(
            {"company": [], "country": [], "company_canonical": []},
            schema={"company": pl.Utf8, "country": pl.Utf8, "company_canonical": pl.Utf8},
        )

    df = pl.DataFrame(companies)
    # Deduplicate on canonical name
    df = df.unique(subset=["company_canonical"], keep="first")
    return df


def _canonicalize_company(name: str) -> str:
    """Normalize company name for dedup: lowercase, strip suffixes/punctuation."""
    name = name.lower().strip()
    # Strip common suffixes
    for suffix in [
        "inc.", "inc", "llc", "ltd.", "ltd", "corp.", "corp", "corporation",
        "co.", "co", "company", "group", "holdings", "international",
        "plc", "limited", "gmbh", "ag", "sa", "s.a.",
    ]:
        if name.endswith(f" {suffix}"):
            name = name[: -(len(suffix) + 1)].strip()
    # Remove punctuation
    name = re.sub(r"[^\w\s]", "", name)
    return name.strip()


def _company_to_domain(company: str) -> str:
    """Generate a plausible domain from company name."""
    canonical = re.sub(r"[^\w]", "", company.lower())
    # Truncate very long names
    if len(canonical) > 20:
        words = company.split()
        if len(words) >= 2:
            canonical = "".join(w[0].lower() for w in words if w[0].isalpha())
            if len(canonical) < 3:
                canonical = re.sub(r"[^\w]", "", company.lower())[:15]
        else:
            canonical = canonical[:15]
    return f"{canonical}.com"


# ---------------------------------------------------------------------------
# Pool assembly
# ---------------------------------------------------------------------------


def build_base_pool(
    companies: pl.DataFrame,
    surnames: pl.DataFrame,
    first_names: pl.DataFrame,
    titles: list[str],
    n: int = 50_000,
    seed: int = 42,
) -> pl.DataFrame:
    """Assemble n entity records from real data sources.

    Parameters
    ----------
    companies : pl.DataFrame
        Must have columns: company, country, company_canonical
    surnames : pl.DataFrame
        Must have columns: name, count
    first_names : pl.DataFrame
        Must have columns: name, count
    titles : list[str]
        List of real job titles from O*NET
    n : int
        Number of records to generate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pl.DataFrame with columns: entity_id, first_name, last_name, middle_name,
        company, title, email, country, company_canonical
    """
    import random as stdlib_random

    rng = stdlib_random.Random(seed)

    # Prepare weighted sampling arrays
    surname_names = surnames["name"].to_list()
    surname_weights = surnames["count"].cast(pl.Float64).to_list()
    surname_total = sum(surname_weights)
    surname_probs = [w / surname_total for w in surname_weights]

    fn_names = first_names["name"].to_list()
    fn_weights = first_names["count"].cast(pl.Float64).to_list()
    fn_total = sum(fn_weights)
    fn_probs = [w / fn_total for w in fn_weights]

    company_list = companies.to_dicts()

    records = []
    used_emails: set[str] = set()

    for _ in range(n):
        # Sample surname (frequency-weighted)
        last_name = _weighted_choice(surname_names, surname_probs, rng)

        # Sample first name (frequency-weighted)
        first_name = _weighted_choice(fn_names, fn_probs, rng)

        # Middle name (~20% of records)
        middle_name = ""
        if rng.random() < 0.20:
            middle_name = _weighted_choice(fn_names, fn_probs, rng)

        # Sample company
        co = rng.choice(company_list)
        company = co["company"]
        country = co["country"]
        company_canonical = co["company_canonical"]

        # Sample title
        title = rng.choice(titles)

        # Generate email
        domain = _company_to_domain(company)
        email = _generate_email(first_name, last_name, domain, rng, used_emails)
        used_emails.add(email)

        records.append({
            "entity_id": str(uuid.uuid4()),
            "first_name": first_name,
            "last_name": last_name,
            "middle_name": middle_name,
            "company": company,
            "title": title,
            "email": email,
            "country": country,
            "company_canonical": company_canonical,
        })

    return pl.DataFrame(records)


def _weighted_choice(items: list, weights: list[float], rng) -> str:
    """Weighted random choice using cumulative distribution."""
    r = rng.random()
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item
    return items[-1]


def _generate_email(
    first_name: str, last_name: str, domain: str, rng, used: set[str],
) -> str:
    """Generate a unique email address."""
    fn = first_name.lower().replace(" ", "")
    ln = last_name.lower().replace(" ", "")

    roll = rng.random()
    if roll < 0.60:
        local = f"{fn}.{ln}"
    elif roll < 0.80:
        local = f"{fn[0]}.{ln}" if fn else ln
    elif roll < 0.90:
        local = f"{fn[0]}{ln}" if fn else ln
    else:
        # Personal domain
        local = fn
        domain = rng.choice(PERSONAL_DOMAINS)

    email = f"{local}@{domain}"

    # Ensure uniqueness
    attempt = 0
    base_email = email
    while email in used:
        attempt += 1
        email = f"{local}{attempt}@{domain}"
        if attempt > 100:
            email = f"{fn}.{ln}.{uuid.uuid4().hex[:4]}@{domain}"
            break

    return email


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def build(
    source_dir: Path = typer.Option("data/phase2/raw", help="Directory with raw source data"),
    output_dir: Path = typer.Option("data/phase2", help="Output directory"),
    n: int = typer.Option(50_000, help="Number of entity records to generate"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Build the base entity pool from real data sources."""
    from src.data.phase2_sources import (
        load_census_surnames,
        load_ssa_names,
        parse_gleif,
        parse_onet_reported,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sources
    console.print("[cyan]Loading GLEIF companies...")
    gleif_df = parse_gleif(source_dir / "gleif_golden_copy.csv")
    company_pool = build_company_pool(gleif_df)
    console.print(f"[green]Company pool: {len(company_pool):,} unique companies")

    console.print("[cyan]Loading Census surnames...")
    surnames = load_census_surnames(source_dir / "census_surnames.csv")
    console.print(f"[green]Surnames: {len(surnames):,}")

    console.print("[cyan]Loading SSA first names...")
    first_names = load_ssa_names(source_dir / "ssa_names.zip")
    console.print(f"[green]First names: {len(first_names):,}")

    console.print("[cyan]Loading O*NET titles...")
    titles = parse_onet_reported(source_dir / "onet_reported_titles.txt")
    console.print(f"[green]Titles: {len(titles):,}")

    # Build pool
    console.print(f"[bold cyan]Building {n:,} entity records...")
    pool = build_base_pool(company_pool, surnames, first_names, titles, n=n, seed=seed)

    # Save
    output_path = output_dir / "base_pool.parquet"
    pool.write_parquet(output_path)
    console.print(f"[bold green]Pool saved: {output_path} ({len(pool):,} records)")

    # Save stats
    stats = {
        "n_records": len(pool),
        "n_companies": len(company_pool),
        "n_surnames": len(surnames),
        "n_first_names": len(first_names),
        "n_titles": len(titles),
        "seed": seed,
    }
    stats_path = output_dir / "pool_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    console.print(f"[green]Stats saved: {stats_path}")


if __name__ == "__main__":
    app()
