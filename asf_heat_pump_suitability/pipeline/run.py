"""CLI: Orchestrate the full pipeline — runs uprns.py then add_features.py in order.

Resolves the supplied geography arguments to a list of LAD codes and runs
``uprns.py`` then ``add_features.py`` sequentially for each LAD.

Example usage:

    # Single LAD
    python run.py --lad manchester

    # All LADs in a combined authority
    python run.py --combined greater-manchester

    # All LADs in a UTLA
    python run.py --utla devon

    # Multiple mixed arguments
    python run.py --lad manchester --lad salford --utla bolton

    # Full GB run (no filter)
    python run.py

    # Production (writes to S3)
    LOCAL_DEV=false python run.py --lad manchester
"""

import subprocess
import sys
from typing import Optional

import typer

from asf_heat_pump_suitability.getters.geography_lookup import get_geography_lookup, resolve_lads

app = typer.Typer(help=__doc__)


@app.command()
def main(
    lad: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--lad",
        help="LAD name, slug, or ONS code. May be repeated.",
    ),
    utla: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--utla",
        help="UTLA name, slug, or ONS code. May be repeated.",
    ),
    combined: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--combined",
        help="Combined authority name, slug, or ONS code. May be repeated.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Override OUTPUT_DIR env var for both steps.",
    ),
) -> None:
    """Run the full pipeline: uprns.py followed by add_features.py, per LAD."""
    lookup = get_geography_lookup()
    lad_codes = resolve_lads(lads=lad, utlas=utla, combineds=combined, lookup=lookup)
    print(f"Resolved {len(lad_codes)} LAD(s).")

    base_cmd = [sys.executable]

    def _run_step(script: str, lad_code: str) -> None:
        cmd = base_cmd + [script, "--lad", lad_code]
        if output_dir:
            cmd += ["--output-dir", output_dir]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)

    for lad_code in lad_codes:
        print(f"\n=== LAD: {lad_code} ===")

        print("--- Step 1/2: uprns.py ---")
        _run_step("asf_heat_pump_suitability/pipeline/uprns.py", lad_code)

        print("--- Step 2/2: add_features.py ---")
        _run_step("asf_heat_pump_suitability/pipeline/add_features.py", lad_code)

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    app()
