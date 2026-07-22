"""
Functions to build and save run manifests recording pipeline output lineage.

Each pipeline entrypoint writes a companion `{output_basename}.manifest.json`
next to every output it saves to S3, recording which input versions, git
commit and parameters produced that output. The `.manifest.json` suffix keeps
the output basename, so it never collides with the front-end `manifest.json`
written by pipeline/run/create_manifest.py.
"""

import json
import logging
import subprocess
from datetime import datetime, timezone

import fsspec

from asf_heat_pump_suitability import PROJECT_DIR, config

MANIFEST_SUFFIX = ".manifest.json"
UNKNOWN_GIT_COMMIT = "unknown"

# Curated lists of the config["data"] datasets each pipeline entrypoint reads
# (directly or through the getters and transform modules it calls), as
# dot-separated key paths. Recorded as `input_versions` in that stage's run
# manifests. Update a stage's list when it starts or stops reading a dataset.
# Legacy config["data_source"] (v1) inputs are out of scope for lineage.
# All stages resolve local authorities via
# pipeline.transform.local_authority.get_dict_la_data, which reads
# processed.valid_la_names and (through load_boundaries) the LA boundaries.
STAGE_INPUT_KEYS = {
    "uprns": [
        "geodata.uk_osopen_uprn",  # load_geodata.load_df_osopen_uprn
        "geodata.boundaries.UK_ons_lad_bounds",  # load_boundaries.load_gdf_local_authority_boundaries
        "geodata.UK_poi_locations",  # load_geodata.load_gdf_poi
        "geodata.grid_square_os_openmap_local",  # load_gdf_os_openmap_layer: important_building, railway_station, building
        "processed.non_domestic_poi_categories",  # poi.load_set_non_domestic_poi_categories
        "processed.valid_la_names",  # local_authority.resolve_list_la_names
        "epc.domestic",  # uprns.load_set_valid_epc_uprns (also via non_residential_entities)
        "epc.commercial.EW",  # uprns.load_set_valid_epc_uprns
        "epc.commercial.S",  # uprns.load_set_valid_epc_uprns
        "EW_household_census_data",  # uprns.get_dict_census_uprn_range
        "S_household_census_data",  # uprns.get_dict_census_uprn_range
    ],
    "add_features": [
        "geodata.boundaries.UK_ons_lad_bounds",  # load_boundaries.load_gdf_local_authority_boundaries
        "geodata.grid_square_os_openmap_local",  # load_gdf_os_openmap_layer: building
        "processed.valid_la_names",  # local_authority.resolve_list_la_names
        "processed.manually_labelled_block_of_flats",  # read directly in add_features
        "geodata.heat_network_zones.desnz_files",  # geo_utils.concat_gdfs reads this directory
        "geodata.heat_network_zones.desnz_polygons",  # load_geodata.load_gdf_heat_network_zones
        "geodata.gb_spatial_signatures.full",  # load_gdf_spatial_signatures_gb; --detail picks
        "geodata.gb_spatial_signatures.simplified",  # one of the two, params.detail records which
        "processed.inspire_file_names",  # read directly; lists the INSPIRE parcel files to load
        "epc.domestic",  # read directly in add_features
        "geodata.gb_code_points",  # load_geodata.load_gdf_code_points
        "geodata.gb_coast_boundaries",  # load_geodata.load_gdf_gb_coast_boundaries
        "geodata.gb_uprn_country_mapping",  # load_geodata.load_transform_dict_uprn_to_country_mapping
    ],
    "decision_tree": [
        "geodata.boundaries.UK_ons_lad_bounds",  # via local_authority.get_dict_la_data
        "geodata.grid_square_os_openmap_local",  # load_gdf_os_openmap_layer: building
        "processed.valid_la_names",  # local_authority.resolve_list_la_names
    ],
    "cluster": [
        "geodata.boundaries.UK_ons_lad_bounds",  # load_boundaries.load_gdf_local_authority_boundaries
        "geodata.grid_square_os_openmap_local",  # load_gdf_os_openmap_layer: building, railway_track, woodland, surface_water_area, tidal_water, important_building
        "geodata.grid_square_os_openmap_greenspace",  # load_gdf_os_openmap_layer: greenspace_site
        "geodata.grid_square_os_openroad",  # load_geodata.load_gdf_os_openroad
        "processed.poi_anchor_properties",  # cluster.load_transform_anchor_property_gdfs
        "geodata.heat_network_zones.desnz_polygons",  # load_geodata.load_gdf_heat_network_zones
        "processed.valid_la_names",  # local_authority.resolve_list_la_names
    ],
    "compute_contextual_features": [
        "geodata.boundaries.UK_ons_lad_bounds",  # via local_authority.get_dict_la_data
        "geodata.grid_square_os_openmap_local",  # load_gdf_os_openmap_layer: building
        "processed.valid_la_names",  # local_authority.resolve_list_la_names
    ],
}


def get_str_git_commit() -> str:
    """
    Get the git commit hash of the currently checked-out code.

    Runs `git rev-parse HEAD` in the project directory so the hash reflects
    the imported package, not the caller's working directory.

    Returns:
        str: 40-character commit hash, or "unknown" if git is unavailable
            (e.g. a deployed environment without a .git directory)
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        logging.warning(
            "Could not read git commit hash; recording '%s' in run manifest.",
            UNKNOWN_GIT_COMMIT,
        )
        return UNKNOWN_GIT_COMMIT


def generate_dict_input_versions(input_keys: list[str]) -> dict:
    """
    Resolve curated config["data"] key paths to their input dataset path strings.

    Records the raw resolved path strings (whose dated prefixes carry the input
    versions) under their dot-separated keys, e.g. `{"epc.domestic": "s3://..."}`.

    Args:
        input_keys (list[str]): dot-separated key paths into `config["data"]`,
            e.g. "geodata.uk_osopen_uprn" — typically one of the curated
            per-stage lists in `STAGE_INPUT_KEYS`

    Returns:
        dict: mapping of each dot-separated dataset key to its path string

    Raises:
        KeyError: if a key path does not exist in `config["data"]` or resolves
            to a config subtree rather than a dataset path string — a typo in a
            curated list must fail loudly at run time, not silently omit lineage
    """
    input_versions = {}
    for dot_key in input_keys:
        value = config["data"]
        for part in dot_key.split("."):
            try:
                value = value[part]
            except (KeyError, TypeError) as error:
                raise KeyError(
                    f"Run manifest input key '{dot_key}' not found in config['data']"
                ) from error
        if isinstance(value, dict):
            raise KeyError(
                f"Run manifest input key '{dot_key}' resolves to a config['data'] "
                "subtree, not a dataset path"
            )
        input_versions[dot_key] = value
    return input_versions


def generate_dict_run_manifest(
    stage: str,
    local_authority: str,
    row_count: int,
    params: dict,
    input_keys: list[str],
) -> dict:
    """
    Generate run manifest dict recording the lineage of one pipeline output.

    Args:
        stage (str): name of the pipeline entrypoint that produced the output,
            e.g. "uprns" or "decision_tree"
        local_authority (str): local authority slug the output was generated for
        row_count (int): number of rows (or geojson features) in the output
        params (dict): CLI arguments the entrypoint was run with
        input_keys (list[str]): dot-separated config["data"] key paths of the
            datasets the stage reads, typically `STAGE_INPUT_KEYS[stage]`

    Returns:
        dict: run manifest with keys stage, local_authority, run_at, git_commit,
            input_versions, row_count, params
    """
    return {
        "stage": stage,
        "local_authority": local_authority,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": get_str_git_commit(),
        "input_versions": generate_dict_input_versions(input_keys),
        "row_count": row_count,
        "params": params,
    }


def get_str_manifest_path(output_path: str) -> str:
    """
    Get the S3 path of the run manifest for an output file.

    The manifest is co-located with the output and named
    `{output_basename}.manifest.json`, e.g. `plymouth_domestic_uprns.parquet`
    -> `plymouth_domestic_uprns.manifest.json`.

    Args:
        output_path (str): S3 path of the output file the manifest describes

    Returns:
        str: S3 path of the companion run manifest
    """
    return output_path.rsplit(".", 1)[0] + MANIFEST_SUFFIX


def save_manifest_to_s3(manifest: dict, output_path: str) -> None:
    """
    Save run manifest as JSON next to the output file it describes.

    A manifest write failure never aborts a pipeline run: any exception raised
    by the write is caught and logged as a warning naming the manifest path,
    and the function returns without raising. Lineage degrades rather than
    failing the run, mirroring the `get_str_git_commit` "unknown" fallback.

    Args:
        manifest (dict): run manifest, as returned by `generate_dict_run_manifest`
        output_path (str): S3 path of the output file the manifest describes

    Returns:
        None
    """
    manifest_path = get_str_manifest_path(output_path)
    logging.info(f"Saving run manifest to {manifest_path}")
    try:
        with fsspec.open(manifest_path, "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.warning(
            "Failed to write run manifest to %s; continuing without it.",
            manifest_path,
            exc_info=True,
        )
