#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "pyyaml",
# ]
# ///
"""
run_ablation_benchmarks.py

Reads configs/models.yaml to dynamically discover models and their supported MRL dimensions.
Scans results/indexes/ to find the full-dimension base and fine-tuned source indices.
Iterates through all combinations of dimensions and quantizations, deriving the index
and running the evaluation.

Can be run in manual mode (waits for confirmation before each step) or --auto mode.
"""

import json
import subprocess
from pathlib import Path

import click
import yaml


def find_source_indices() -> dict[tuple[str, str], dict]:
    """
    Scan the results/indexes directory to find the original full-dimension
    source indexes for each model and mode (base vs ft).
    """
    sources: dict[tuple[str, str], dict] = {}
    indexes_dir = Path("results/indexes")
    if not indexes_dir.exists():
        return sources

    for p in indexes_dir.iterdir():
        if not p.is_dir():
            continue

        meta_path = p / "metadata.json"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            
            model = meta.get("model_key")
            if not model:
                continue

            # If it is a derived index, we don't want to use it as the source
            if meta.get("derivation_source"):
                continue

            # Determine mode (base or ft)
            model_path = meta.get("model_path")
            if model_path and "-ft" in str(model_path).lower():
                mode = "ft"
            elif "ft" in p.name.lower():
                mode = "ft"
            else:
                mode = "base"

            dim = meta.get("dim", 0)

            # Keep the index with the maximum dimension for this model+mode
            key = (model, mode)
            if key not in sources or dim > sources[key]["dim"]:
                sources[key] = {
                    "path": str(p),
                    "dim": dim,
                    "model_path": model_path,
                }
        except Exception as e:
            click.echo(f"Warning: Could not parse {meta_path}: {e}", err=True)

    return sources


@click.command()
@click.option(
    "--auto",
    is_flag=True,
    help="Run automatically without prompting for confirmation before each step.",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False),
    default="configs/models.yaml",
    help="Path to models config file.",
)
def main(auto: bool, config_file: str) -> None:
    """Run Dimensionality & Quantization Ablation."""
    
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Discover available source indices dynamically
    sources = find_source_indices()
    if not sources:
        click.echo(
            "Error: No source indexes found in results/indexes/. Run a full encode first.",
            err=True,
        )
        raise SystemExit(1)

    quantizations = ["fp32", "int8", "binary"]
    modes = ["base", "ft"]
    auto_mode = auto

    click.echo("=================================================================")
    click.echo(f"Starting Dynamic Dimensionality & Quantization Ablation (Auto: {auto_mode})")
    click.echo("=================================================================")

    # Iterate through models defined in the config
    for model_key, model_cfg in config.items():
        if not isinstance(model_cfg, dict) or model_cfg.get("type") == "bm25":
            continue

        target_dims = model_cfg.get("dims", [])
        if not target_dims:
            continue

        for mode in modes:
            source_info = sources.get((model_key, mode))
            if not source_info:
                # We don't have a source index for this combination yet
                continue

            source_index = source_info["path"]
            source_dim = source_info["dim"]

            for dim in target_dims:
                if dim > source_dim:
                    click.echo(
                        f"Skipping {model_key} {mode} at {dim}d (source is only {source_dim}d)"
                    )
                    continue

                for quant in quantizations:
                    output_idx = f"results/indexes/{model_key}_{mode}_{dim}_{quant}"
                    output_json = f"results/{model_key}_{mode}_{dim}_{quant}.json"

                    click.echo(f"\n-----------------------------------------------------------------")
                    click.echo(f"Next Run: {model_key} ({mode}) | Dim: {dim} | Quant: {quant}")
                    click.echo(f"Source Index: {source_index}")
                    click.echo(f"Output Index: {output_idx}")
                    click.echo(f"Output JSON:  {output_json}")
                    click.echo(f"-----------------------------------------------------------------")

                    if not auto_mode:
                        try:
                            # click.prompt isn't perfectly suited for single keystrokes without enter,
                            # so we use a standard input with click.echo for the prompt.
                            click.echo("Press Enter to execute (or 's' to skip, 'a' for auto-mode, Ctrl+C to abort)...", nl=False)
                            user_input = input().strip().lower()
                            if user_input == "s":
                                click.echo("Skipping...")
                                continue
                            elif user_input == "a":
                                click.echo("Switching to auto-mode for the remaining runs...")
                                auto_mode = True
                        except (KeyboardInterrupt, EOFError):
                            click.echo("\nAborted by user.")
                            raise SystemExit(0)

                    try:
                        # Step 1: Derive Index
                        # Check for metadata.json, not just directory existence —
                        # a dir without metadata.json is a corrupt leftover from a failed build.
                        index_meta = Path(output_idx) / "metadata.json"
                        if not index_meta.exists():
                            # Clean up corrupt leftovers if the dir exists but has no metadata
                            if Path(output_idx).exists():
                                import shutil
                                click.echo(f"[1/2] Removing corrupt index dir (no metadata.json): {output_idx}")
                                shutil.rmtree(output_idx)
                            click.echo("[1/2] Deriving index...")
                            cmd_derive = [
                                "uv", "run", "python", "src/eval/build_index.py",
                                "--source-index", source_index,
                                "--output-dir", output_idx,
                                "--truncate-dim", str(dim),
                                "--quantization", quant,
                            ]
                            subprocess.run(cmd_derive, check=True)
                        else:
                            click.echo(f"[1/2] Index {output_idx} already exists. Skipping derivation.")

                        # Step 2: Evaluate Index
                        if not Path(output_json).exists():
                            click.echo("[2/2] Running evaluation...")
                            cmd_eval = [
                                "uv", "run", "python", "src/eval/run_eval.py",
                                "--model", model_key,
                                "--index-dir", output_idx,
                                "--eval-queries", "data/eval/eval_queries.parquet",
                                "--output", output_json,
                                "--serialization", "pipe",
                                "--experiment-id", f"{model_key}_{mode}_{dim}_{quant}",
                            ]
                            # If it's a fine-tuned model, we must pass the model-path for the encoder
                            if source_info["model_path"]:
                                cmd_eval.extend(["--model-path", source_info["model_path"]])

                            subprocess.run(cmd_eval, check=True)
                        else:
                            click.echo(f"[2/2] Evaluation results {output_json} already exists. Skipping.")

                        click.echo(f"✓ Finished {model_key} {mode} ({dim}-dim {quant})")

                    except subprocess.CalledProcessError as e:
                        click.echo(f"✗ FAILED {model_key} {mode} ({dim}-dim {quant}): {e}", err=True)
                        click.echo("  Continuing to next run...", err=True)
                        continue

    click.echo("\n=================================================================")
    click.echo("All ablation runs complete!")
    click.echo("Run aggregate.py to compile the final report:")
    click.echo("uv run python src/eval/aggregate.py --results-dir results/ --output-csv results/master_results.csv --output-report results/report.md")
    click.echo("=================================================================")


if __name__ == "__main__":
    main()
