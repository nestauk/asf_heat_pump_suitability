"""
Build and save run manifests recording pipeline output lineage.

Each pipeline entrypoint writes a companion `{output_basename}.manifest.json`
next to every output it saves to S3, recording which input versions, git
commit and parameters produced it. Keeping the output basename in the filename
avoids colliding with the front-end `manifest.json` from
pipeline/run/create_manifest.py.
"""

import json
import logging
import subprocess
from datetime import datetime, timezone

import fsspec

from asf_heat_pump_suitability import PROJECT_DIR, config

MANIFEST_SUFFIX = ".manifest.json"
UNKNOWN_GIT_COMMIT = "unknown"

# The config["data"] datasets each entrypoint reads (directly or via the
# getters/transform modules it calls), recorded as `input_versions` in that
# stage's manifests. Update a list when its stage starts or stops reading a
# dataset. Legacy config["data_source"] (v1) is out of scope.
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
    Get the git commit hash of the checked-out code.

    Runs in the project directory so the hash reflects the imported package,
    not the caller's working directory.

    Returns:
        str: 40-character commit hash, or "unknown" if git is unavailable
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
    Resolve config["data"] key paths to their dataset path strings, whose
    dated prefixes carry the input versions.

    Args:
        input_keys (list[str]): dot-separated key paths into `config["data"]`,
            typically one of the `STAGE_INPUT_KEYS` lists

    Returns:
        dict: mapping of each key to its path string

    Raises:
        KeyError: if a key is missing or resolves to a config subtree — a typo
            in a curated list must fail loudly, not silently omit lineage
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
        stage (str): pipeline entrypoint that produced the output, e.g. "uprns"
        local_authority (str): local authority slug the output was generated for
        row_count (int): number of rows (or geojson features) in the output
        params (dict): CLI arguments the entrypoint was run with
        input_keys (list[str]): config["data"] key paths of the datasets the
            stage reads, typically `STAGE_INPUT_KEYS[stage]`

    Returns:
        dict: run manifest
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

    Args:
        output_path (str): S3 path of the output file the manifest describes

    Returns:
        str: co-located path ending `.manifest.json`
    """
    return output_path.rsplit(".", 1)[0] + MANIFEST_SUFFIX


def save_manifest_to_s3(manifest: dict, output_path: str) -> None:
    """
    Save run manifest as JSON next to the output file it describes.

    Write failures are logged and swallowed: lineage degrades rather than
    aborting a pipeline run.

    Args:
        manifest (dict): run manifest, as returned by `generate_dict_run_manifest`
        output_path (str): S3 path of the output file the manifest describes
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
