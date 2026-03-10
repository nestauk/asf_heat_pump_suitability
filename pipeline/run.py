"""CLI: Orchestrate the full pipeline — runs uprns.py then add_features.py in order.

Example usage:

    # Full UK run (production, writes to S3)
    LOCAL_DEV=false uv run python pipeline/run.py

    # Greater Manchester only (local, writes to ./outputs/)
    uv run python pipeline/run.py --local-authorities greater_manchester_las

    # Plymouth, custom output dir
    uv run python pipeline/run.py --local-authorities plymouth --output-dir /data/outputs/
"""

import subprocess
import sys
from typing import Optional

import typer

app = typer.Typer(help=__doc__)


@app.command()
def main(
    local_authorities: Optional[str] = typer.Option(
        None,
        "--local-authorities",
        help=(
            "Local authority preset or space-separated LA codes. "
            "Available presets: plymouth, plymouth_similar_cities, sampling_areas, greater_manchester_las."
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Override OUTPUT_DIR env var for both steps.",
    ),
) -> None:
    """Run the full pipeline: uprns.py followed by add_features.py."""
    base_cmd = [sys.executable]

    def build_args(script: str, extra: list[str] | None = None) -> list[str]:
        cmd = base_cmd + [script]
        if local_authorities:
            cmd += ["--local-authorities", local_authorities]
        if output_dir:
            cmd += ["--output-dir", output_dir]
        if extra:
            cmd += extra
        return cmd

    # Step 1: filter UPRNs to domestic
    print("=== Step 1/2: uprns.py ===")
    result = subprocess.run(build_args("pipeline/uprns.py"), check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # Step 2: add features
    print("=== Step 2/2: add_features.py ===")
    result = subprocess.run(build_args("pipeline/add_features.py"), check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print("=== Pipeline complete ===")


if __name__ == "__main__":
    app()
