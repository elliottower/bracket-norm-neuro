"""Allen Visual Behavior Neuropixels data loading.

153 sessions, 81 mice, visual cortex (V1, LM, AL, PM, AM) + subcortical (LGd, LP).
Change-detection task with passive viewing and active epochs.

Requires: pip install allensdk (heavy dependency, in optional [allen] extra).
S3 bucket: visual-behavior-neuropixels-data
"""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "allen"


def get_cache(cache_dir: Path | None = None):
    """Get AllenSDK VisualBehaviorNeuropixelsProjectCache.

    Uses from_s3_cache which creates a local cache and downloads the manifest
    from S3 via boto3 (anonymous access, no credentials needed).
    """
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorNeuropixelsProjectCache,
    )

    cache = cache_dir or CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initializing Allen VBN cache at {cache}")
    return VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=str(cache))


def list_sessions(min_units: int = 20, area: str | None = None) -> list[dict]:
    """List available sessions, optionally filtered by area and unit count."""
    cache = get_cache()
    sessions_table = cache.get_ecephys_session_table()

    logger.info(f"Sessions table: {len(sessions_table)} rows, columns: {list(sessions_table.columns)[:10]}")

    results = []
    for sid, row in sessions_table.iterrows():
        structure_col = None
        for col in ["ecephys_structure_acronyms", "structure_acronyms", "targeted_structures"]:
            if col in sessions_table.columns:
                structure_col = col
                break

        structures = row.get(structure_col, []) if structure_col else []
        if isinstance(structures, str):
            cleaned = structures.strip("[] ")
            structures = [s.strip().strip("'\"") for s in cleaned.split(",")]
        elif hasattr(structures, "tolist"):
            structures = structures.tolist()
        structures = [s for s in structures if s] if structures else []

        info = {
            "session_id": sid,
            "mouse_id": row.get("mouse_id"),
            "session_type": row.get("session_type"),
            "structure_acronyms": structures,
        }
        if area and area not in structures:
            continue
        results.append(info)

    logger.info(f"Found {len(results)} sessions" + (f" with area {area}" if area else ""))
    return results


def load_session_units(session_id: int, area: str, bin_size: float = 0.01) -> dict | None:
    """Load spike-binned activity for units in a specific area.

    Returns dict with:
        activity: (n_trials, n_units, n_bins)
        stimulus_presentations: DataFrame of stimulus info
        unit_ids: array of unit IDs
    Or None if area not present.
    """
    cache = get_cache()
    session = cache.get_ecephys_session(ecephys_session_id=session_id)

    units = session.get_units()
    area_col = None
    for col in ["ecephys_structure_acronym", "structure_acronym", "targeted_structure"]:
        if col in units.columns:
            area_col = col
            break

    if area_col is None:
        logger.warning(f"No structure column in units table: {list(units.columns)}")
        return None

    area_units = units[units[area_col] == area]

    if len(area_units) < 5:
        logger.warning(f"Only {len(area_units)} units in {area} for session {session_id}")
        return None

    unit_ids = area_units.index.values
    spike_times = session.spike_times

    stim = session.stimulus_presentations
    stim = stim[stim["stimulus_name"].isin(["natural_scenes", "gabors", "drifting_gratings"])]

    if len(stim) == 0:
        return None

    n_trials = len(stim)
    n_units = len(unit_ids)
    pre_time = 0.1
    post_time = 0.5
    n_bins = int((pre_time + post_time) / bin_size)
    counts = np.zeros((n_trials, n_units, n_bins), dtype=np.float32)

    for t, (_, row) in enumerate(stim.iterrows()):
        onset = row["start_time"]
        for u, uid in enumerate(unit_ids):
            if uid not in spike_times:
                continue
            st = spike_times[uid]
            trial_mask = (st >= onset - pre_time) & (st < onset + post_time)
            for spike_t in st[trial_mask]:
                b = int((spike_t - (onset - pre_time)) / bin_size)
                if 0 <= b < n_bins:
                    counts[t, u, b] += 1

    return {
        "activity": counts,
        "stimulus_presentations": stim,
        "unit_ids": unit_ids,
    }
