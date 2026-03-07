"""
Profile generation script.

Generates synthetic entity profiles and runs a 7-step quality pipeline,
then splits into index / triplet_source / eval sets.

Usage
-----
    uv run python src/data/generate.py --config configs/dataset.yaml --output-dir data/
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import typer
import yaml
from faker import Faker
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

PERSONAL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]
EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _company_slug(company: str) -> str:
    """Lowercase, spaces→hyphens, strip special chars, max 20 chars."""
    slug = company.lower()
    slug = re.sub(r"[^a-z0-9\s\-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug[:20].rstrip("-")


def _make_email(first: str, last: str, company: str, is_work: bool, fake: Faker,
                personal_domains: list[str]) -> str:
    """Generate a plausible email address."""
    fn = re.sub(r"[^a-z]", "", first.lower())
    ln = re.sub(r"[^a-z]", "", last.lower())
    # Guard against empty names
    fn = fn or "user"
    ln = ln or "contact"

    if is_work:
        domain = _company_slug(company) + ".com"
        roll = fake.random_int(0, 2)
        if roll == 0:
            local = f"{fn}.{ln}"
        elif roll == 1:
            local = f"{fn[0]}{ln}"
        else:
            local = f"{fn}_{ln}"
    else:
        domain = fake.random_element(personal_domains)
        local = f"{fn}.{ln}"

    local = local[:30]  # keep emails reasonable
    return f"{local}@{domain}"


def _hash_entity(entity_id: str) -> int:
    """Deterministic integer hash of an entity_id."""
    return int(hashlib.sha256(entity_id.encode()).hexdigest(), 16)


def _entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


# ── Profile generation ────────────────────────────────────────────────────────

def generate_profiles(cfg: dict, n: int) -> list[dict]:
    """Generate `n` raw synthetic profiles."""
    fake = Faker("en_US")
    Faker.seed(cfg.get("random_seed", 42))

    country_dist: dict[str, float] = cfg["country_distribution"]
    countries = list(country_dist.keys())
    weights = [country_dist[c] for c in countries]
    email_work_ratio: float = cfg.get("email_work_ratio", 0.70)
    personal_domains: list[str] = cfg.get("personal_email_domains", PERSONAL_DOMAINS)

    profiles: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Generating profiles…"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("gen", total=n)

        batch = 0
        while len(profiles) < n:
            batch_size = min(10_000, n - len(profiles))
            for _ in range(batch_size):
                first = fake.first_name()
                last = fake.last_name()
                company = fake.company()
                country = fake.random_element(
                    elements=countries,
                    # weights not directly supported — use choices
                )
                # weighted country
                country = fake.random.choices(countries, weights=weights, k=1)[0]
                is_work = fake.random.random() < email_work_ratio
                email = _make_email(first, last, company, is_work, fake, personal_domains)
                entity_id = str(uuid.uuid4())

                profiles.append(
                    {
                        "entity_id": entity_id,
                        "first_name": first,
                        "last_name": last,
                        "company": company,
                        "email": email,
                        "country": country,
                    }
                )
            batch += 1
            progress.update(task, completed=len(profiles))

    return profiles


# ── Quality pipeline ──────────────────────────────────────────────────────────

def quality_pipeline(profiles: list[dict], n_profiles: int) -> tuple[list[dict], dict]:
    """Run 7-step quality pipeline. Returns (cleaned_profiles, stats)."""
    stats: dict[str, Any] = {}

    console.rule("[bold yellow]Quality Pipeline")

    # ── Step 1: Exact dedup ───────────────────────────────────────────────────
    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    for p in profiles:
        key = hashlib.md5(
            (p["first_name"] + p["last_name"] + p["company"] + p["email"] + p["country"]).encode()
        ).hexdigest()
        if key not in seen_hashes:
            seen_hashes.add(key)
            deduped.append(p)
    console.print(
        f"[green]Step 1 — Exact dedup:[/] {len(profiles):,} → {len(deduped):,} "
        f"(dropped {len(profiles) - len(deduped):,})"
    )
    profiles = deduped

    # ── Step 2: Email uniqueness ──────────────────────────────────────────────
    seen_emails: set[str] = set()
    email_unique: list[dict] = []
    for p in profiles:
        e = p["email"].lower()
        if e not in seen_emails:
            seen_emails.add(e)
            email_unique.append(p)
    console.print(
        f"[green]Step 2 — Email uniqueness:[/] {len(profiles):,} → {len(email_unique):,} "
        f"(dropped {len(profiles) - len(email_unique):,})"
    )
    profiles = email_unique

    # ── Step 3: Name collision audit ──────────────────────────────────────────
    name_company_counts: Counter = Counter(
        (p["first_name"], p["last_name"], p["company"]) for p in profiles
    )
    collisions = sum(1 for c in name_company_counts.values() if c > 1)
    collision_rows = sum(c for c in name_company_counts.values() if c > 1)
    console.print(
        f"[green]Step 3 — Name collision audit:[/] "
        f"{collisions:,} duplicate name+company combos "
        f"({collision_rows:,} rows involved — intentional hard cases, not dropped)"
    )
    stats["name_company_collisions"] = collisions
    stats["name_company_collision_rows"] = collision_rows

    # ── Step 4: Distribution check ────────────────────────────────────────────
    country_counts: Counter = Counter(p["country"] for p in profiles)
    domain_counts: Counter = Counter(p["email"].split("@")[-1] for p in profiles)
    name_counts: Counter = Counter((p["first_name"] + " " + p["last_name"]) for p in profiles)
    top20_names = dict(name_counts.most_common(20))
    country_entropy = _entropy(dict(country_counts))
    domain_entropy = _entropy(dict(domain_counts))

    console.print(
        f"[green]Step 4 — Distribution check:[/] "
        f"country entropy={country_entropy:.3f} bits, "
        f"domain entropy={domain_entropy:.3f} bits"
    )
    stats["country_entropy"] = round(country_entropy, 4)
    stats["domain_entropy"] = round(domain_entropy, 4)
    stats["country_distribution"] = dict(country_counts)
    stats["top_20_names"] = top20_names
    stats["email_domain_distribution"] = dict(domain_counts.most_common(30))

    # ── Step 5: Minimum field length ──────────────────────────────────────────
    min_len = [
        p for p in profiles
        if len(p.get("first_name", "") or "") >= 2 and len(p.get("last_name", "") or "") >= 2
    ]
    console.print(
        f"[green]Step 5 — Min field length:[/] {len(profiles):,} → {len(min_len):,} "
        f"(dropped {len(profiles) - len(min_len):,})"
    )
    profiles = min_len

    # ── Step 6: Email format validation ──────────────────────────────────────
    valid_email = [p for p in profiles if EMAIL_RE.match(p.get("email", ""))]
    console.print(
        f"[green]Step 6 — Email format:[/] {len(profiles):,} → {len(valid_email):,} "
        f"(dropped {len(profiles) - len(valid_email):,})"
    )
    profiles = valid_email

    # ── Step 7: Final count assertion ─────────────────────────────────────────
    min_required = int(n_profiles * 0.95)
    console.print(
        f"[green]Step 7 — Final count:[/] {len(profiles):,} profiles "
        f"(minimum required: {min_required:,})"
    )
    assert len(profiles) >= min_required, (
        f"Quality pipeline dropped too many records: {len(profiles):,} < {min_required:,}"
    )

    stats["final_count"] = len(profiles)
    return profiles, stats


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_profiles(profiles: list[dict], cfg: dict) -> tuple[
    list[dict], list[dict], list[dict]
]:
    """
    Deterministic hash split.
      eval:           hash % 120 == 0
      triplet_source: hash % 120 in [1 … 20]
      index:          remainder
    """
    eval_set: list[dict] = []
    triplet_source: list[dict] = []
    index_set: list[dict] = []

    for p in profiles:
        h = _hash_entity(p["entity_id"]) % 120
        if h == 0:
            eval_set.append(p)
        elif 1 <= h <= 20:
            triplet_source.append(p)
        else:
            index_set.append(p)

    return index_set, triplet_source, eval_set


# ── Main ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    config: str = typer.Option("configs/dataset.yaml", "--config", help="Path to dataset.yaml"),
    output_dir: str = typer.Option("data/", "--output-dir", help="Root output directory"),
) -> None:
    cfg = _load_config(config)
    n_profiles: int = cfg["n_profiles"]
    n_index: int = cfg["n_index"]
    n_triplet: int = cfg["n_triplet_source"]
    n_eval: int = cfg["n_eval"]
    out = Path(output_dir)

    # Create output dirs
    for sub in ["raw", "processed", "eval", "triplets"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Entity Resolution Dataset Generator")
    console.print(
        f"Target: [bold]{n_profiles:,}[/] profiles "
        f"→ index:[bold]{n_index:,}[/]  "
        f"triplet:[bold]{n_triplet:,}[/]  "
        f"eval:[bold]{n_eval:,}[/]"
    )

    # ── 1. Generate ───────────────────────────────────────────────────────────
    # Over-generate slightly to absorb quality-pipeline drops
    target = int(n_profiles * 1.08)
    console.print(f"Generating [cyan]{target:,}[/] raw profiles (8% buffer)…")
    raw_profiles = generate_profiles(cfg, target)

    console.print(f"[dim]Raw generated: {len(raw_profiles):,}[/]")

    # ── 2. Quality pipeline ───────────────────────────────────────────────────
    profiles, stats = quality_pipeline(raw_profiles, n_profiles)

    # ── 3. Save raw parquet ───────────────────────────────────────────────────
    raw_path = out / "raw" / "profiles_all.parquet"
    df_all = pl.DataFrame(profiles)
    df_all.write_parquet(raw_path)
    console.print(f"Saved raw parquet → [bold]{raw_path}[/]  ({len(profiles):,} rows)")

    # ── 4. Split ──────────────────────────────────────────────────────────────
    console.rule("[bold yellow]Splitting")
    index_set, triplet_set, eval_set = split_profiles(profiles, cfg)
    console.print(
        f"  index:          {len(index_set):,}\n"
        f"  triplet_source: {len(triplet_set):,}\n"
        f"  eval:           {len(eval_set):,}"
    )

    # Save splits
    pl.DataFrame(index_set).write_parquet(out / "processed" / "index.parquet")
    pl.DataFrame(triplet_set).write_parquet(out / "processed" / "triplet_source.parquet")
    pl.DataFrame(eval_set).write_parquet(out / "eval" / "eval_profiles.parquet")

    # ── 5. Save stats ─────────────────────────────────────────────────────────
    stats_path = out / "processed" / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    console.print(f"Saved stats → [bold]{stats_path}[/]")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hash_method": "sha256(entity_id) % 120",
        "split_rules": {
            "eval": "hash % 120 == 0",
            "triplet_source": "hash % 120 in [1..20]",
            "index": "remainder",
        },
        "counts": {
            "total": len(profiles),
            "index": len(index_set),
            "triplet_source": len(triplet_set),
            "eval": len(eval_set),
        },
        "quality_steps": {
            "name_company_collisions": stats.get("name_company_collisions"),
            "country_entropy_bits": stats.get("country_entropy"),
            "domain_entropy_bits": stats.get("domain_entropy"),
        },
    }
    manifest_path = out / "processed" / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    console.print(f"Saved manifest → [bold]{manifest_path}[/]")

    # ── 6. Summary table ──────────────────────────────────────────────────────
    table = Table(title="Dataset Generation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Path")
    table.add_row("All (raw)", f"{len(profiles):,}", str(raw_path))
    table.add_row("Index", f"{len(index_set):,}", "data/processed/index.parquet")
    table.add_row("Triplet Source", f"{len(triplet_set):,}", "data/processed/triplet_source.parquet")
    table.add_row("Eval", f"{len(eval_set):,}", "data/eval/eval_profiles.parquet")
    console.print(table)
    console.print("[bold green]Done![/]")


if __name__ == "__main__":
    app()
