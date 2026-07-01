"""IBL Brain-Wide Map data loading via ONE API.

621,733 neurons, 139 mice, 12 labs, 279 brain areas.
Standardized visual decision-making task.

Requires: pip install ONE-api ibllib
First use: ONE will prompt for credentials (password='international').
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BASE_URL = "https://openalyx.internationalbrainlab.org"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "ibl"


_one_instance = None


def get_one(cache_dir: Optional[Path] = None):
    global _one_instance
    if _one_instance is not None:
        return _one_instance

    from one.api import ONE

    cache = cache_dir or CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    # Write params file directly so ONE doesn't try interactive setup on headless containers
    one_dir = Path.home() / ".one"
    one_dir.mkdir(exist_ok=True)
    params_file = one_dir / ".alyx.internationalbrainlab.org"
    if not params_file.exists():
        params_file.write_text(json.dumps({
            "ALYX_URL": BASE_URL,
            "ALYX_LOGIN": "intbrainlab",
            "ALYX_PWD": "international",
            "CACHE_DIR": str(cache),
        }))
        logger.info("Wrote ONE params file to %s", params_file)

    try:
        ONE.setup(base_url=BASE_URL, silent=True)
    except Exception as e:
        logger.warning("ONE.setup failed (non-fatal): %s", e)

    _one_instance = ONE(
        base_url=BASE_URL,
        password="international",
        cache_dir=str(cache),
        silent=True,
        mode="remote",
    )
    return _one_instance


def find_sessions_for_region(region: str, min_neurons: int = 20) -> list[dict]:
    """Find sessions with enough neurons in a target brain region."""
    one = get_one()

    # Use REST API directly — search_insertions can silently return empty
    # when called through certain ONE configurations
    try:
        rest_results = one.alyx.rest("insertions", "list", atlas_acronym=region)
    except Exception as e:
        logger.warning(f"REST insertions list failed for {region}: {e}")
        rest_results = []

    logger.info(f"REST insertions({region}) returned {len(rest_results)} results")

    sessions = []
    for info in rest_results[:20]:
        try:
            pid = info.get("id")
            eid = info.get("session") or info.get("session_info", {}).get("id")
            if eid is None:
                logger.warning(f"  pid={pid}: no session ID in response")
                continue
            probe_name = info.get("name", "probe00")
            sessions.append(
                {
                    "pid": pid,
                    "eid": eid,
                    "probe": probe_name,
                    "lab": info.get("session_info", {}).get("lab", "unknown"),
                    "subject": info.get("session_info", {}).get("subject", "unknown"),
                }
            )
        except Exception as e:
            logger.warning(f"  pid={pid}: failed to parse insertion: {e}")
            continue
    logger.info(f"Found {len(sessions)} usable insertions for {region}")
    return sessions


def _to_str_array(raw) -> np.ndarray:
    """Convert any region-label container to a 1-D numpy array of Python str.

    Handles: numpy bytes arrays (b'VISp'), pandas Series, lists, object arrays,
    numpy U-string arrays, and scalar fallbacks. This is the single place where
    we normalise whatever the ONE API hands back.
    """
    # Unwrap pandas Series / Index
    if hasattr(raw, "values"):
        raw = raw.values

    arr = np.asarray(raw).ravel()

    if arr.dtype.kind == "S":
        # numpy bytes array (e.g. dtype='|S8') -- decode to str
        return np.array([v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in arr])

    if arr.dtype.kind == "O":
        # object array -- items could be str, bytes, or anything
        return np.array([
            v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in arr
        ])

    if arr.dtype.kind == "U":
        # already a proper unicode string array
        return arr

    # Numeric or other unexpected dtype -- convert to str as last resort
    return np.array([str(v) for v in arr])


def load_session(eid: str, probe: str = "probe00") -> dict:
    """Load spikes, clusters, and trials for a session.

    Returns dict with keys: spike_times, spike_clusters, cluster_regions,
    trial_intervals, trial_choice, trial_contrast_left, trial_contrast_right,
    trial_feedback_type.
    """
    one = get_one()

    # Try multiple collection paths — IBL data lives under different paths
    # depending on the processing pipeline version
    collections_to_try = [
        f"alf/{probe}/pykilosort",
        f"alf/{probe}",
    ]

    spikes = None
    clusters = None
    for collection in collections_to_try:
        try:
            spikes = one.load_object(eid, "spikes", collection=collection)
            clusters = one.load_object(eid, "clusters", collection=collection)
            logger.info("load_session(%s): loaded from collection=%s", eid, collection)
            break
        except Exception as e:
            logger.debug("load_session(%s): collection=%s failed: %s", eid, collection, e)
            continue

    if spikes is None or clusters is None:
        raise RuntimeError(
            f"Could not load spikes/clusters for {eid} probe={probe}. "
            f"Tried collections: {collections_to_try}"
        )
    trials = one.load_object(eid, "trials")

    # --- Debug logging for clusters object (diagnosing 0-region bug) ---
    clusters_type = type(clusters).__name__
    clusters_keys = list(clusters.keys()) if hasattr(clusters, "keys") else dir(clusters)
    logger.info(
        "load_session(%s) clusters: type=%s, keys=%s",
        eid, clusters_type, clusters_keys,
    )

    # Try multiple possible attribute names for brain region labels
    raw_regions = None
    for key in ("acronym", "atlas_id", "brainLocationIds_ccf_2017", "brain_region"):
        if hasattr(clusters, "get"):
            val = clusters.get(key)
        else:
            val = getattr(clusters, key, None)
        if val is not None:
            logger.info(
                "  clusters['%s']: type=%s, dtype=%s, len=%s, sample=%s",
                key,
                type(val).__name__,
                getattr(val, "dtype", "N/A"),
                len(val) if hasattr(val, "__len__") else "N/A",
                list(val[:5]) if hasattr(val, "__getitem__") and hasattr(val, "__len__") and len(val) > 0 else "empty",
            )
            if raw_regions is None:
                raw_regions = val
                logger.info("  -> using '%s' as region labels", key)

    if raw_regions is None:
        logger.info(
            "  No region labels in clusters (keys: %s), trying channels fallback", clusters_keys
        )
        for collection in collections_to_try:
            try:
                channels = one.load_object(eid, "channels", collection=collection)
                ch_keys = list(channels.keys()) if hasattr(channels, "keys") else dir(channels)
                logger.info("  channels keys: %s", ch_keys)
                ch_regions = None
                for ch_key in ("acronym", "brainLocationAcronyms_ccf_2017", "brain_region"):
                    val = channels.get(ch_key) if hasattr(channels, "get") else getattr(channels, ch_key, None)
                    if val is not None:
                        ch_regions = _to_str_array(val)
                        logger.info("  -> using channels['%s'] (%d entries, sample=%s)", ch_key, len(ch_regions), list(ch_regions[:3]))
                        break
                if ch_regions is None:
                    atlas_ids = channels.get("brainLocationIds_ccf_2017") if hasattr(channels, "get") else getattr(channels, "brainLocationIds_ccf_2017", None)
                    if atlas_ids is None:
                        atlas_ids = channels.get("atlas_id") if hasattr(channels, "get") else getattr(channels, "atlas_id", None)
                    if atlas_ids is not None:
                        try:
                            from iblatlas.regions import BrainRegions
                            br = BrainRegions()
                            atlas_ids_int = np.asarray(atlas_ids).astype(int)
                            ch_regions = np.array([br.id2acronym(int(aid))[0] for aid in atlas_ids_int])
                            logger.info("  -> mapped %d atlas IDs to acronyms via BrainRegions (sample=%s)", len(ch_regions), list(ch_regions[:5]))
                        except Exception as e:
                            logger.warning("  iblatlas BrainRegions mapping failed: %s", e)
                if ch_regions is not None and len(ch_regions) > 0:
                    cluster_channels = clusters.get("channels") if hasattr(clusters, "get") else getattr(clusters, "channels", None)
                    if cluster_channels is not None:
                        cluster_channels = np.asarray(cluster_channels).astype(int)
                        valid = cluster_channels < len(ch_regions)
                        raw_regions = np.where(valid, ch_regions[np.clip(cluster_channels, 0, len(ch_regions) - 1)], "unknown")
                        logger.info("  mapped %d/%d clusters to regions via channels", int(valid.sum()), len(cluster_channels))
                    break
            except Exception as e:
                logger.debug("  channels fallback failed for collection=%s: %s", collection, e)

    if raw_regions is None:
        logger.warning(
            "  No region labels found in clusters or channels!"
        )
        cluster_regions = np.array([], dtype=str)
    else:
        cluster_regions = _to_str_array(raw_regions)

    logger.info(
        "  final cluster_regions: dtype=%s, shape=%s, sample=%s",
        cluster_regions.dtype,
        cluster_regions.shape,
        list(cluster_regions[:5]) if len(cluster_regions) > 0 else "empty",
    )

    return {
        "spike_times": spikes.times,
        "spike_clusters": spikes.clusters,
        "cluster_regions": cluster_regions,
        "trial_intervals": trials.intervals,
        "trial_choice": trials.choice,
        "trial_contrast_left": trials.contrastLeft,
        "trial_contrast_right": trials.contrastRight,
        "trial_feedback_type": trials.feedbackType,
    }


def bin_spikes(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    intervals: np.ndarray,
    bin_size: float = 0.01,
    pre_time: float = 0.5,
    post_time: float = 1.0,
) -> np.ndarray:
    """Bin spikes into (n_trials, n_neurons, n_bins) array.

    Args:
        spike_times: (n_spikes,) spike timestamps
        spike_clusters: (n_spikes,) cluster IDs per spike
        intervals: (n_trials, 2) trial start/end times
        bin_size: bin width in seconds
        pre_time: time before trial onset to include
        post_time: time after trial onset to include

    Returns:
        (n_trials, n_neurons, n_bins) spike count tensor
    """
    cluster_ids = np.unique(spike_clusters)
    n_neurons = len(cluster_ids)
    n_bins = int((pre_time + post_time) / bin_size)
    n_trials = len(intervals)

    counts = np.zeros((n_trials, n_neurons, n_bins), dtype=np.float32)
    cluster_to_idx = {c: i for i, c in enumerate(cluster_ids)}

    for t, (start, _end) in enumerate(intervals):
        t_start = start - pre_time
        t_end = start + post_time
        mask = (spike_times >= t_start) & (spike_times < t_end)
        trial_spikes = spike_times[mask]
        trial_clusters = spike_clusters[mask]

        for spike_t, cluster in zip(trial_spikes, trial_clusters):
            bin_idx = int((spike_t - t_start) / bin_size)
            if 0 <= bin_idx < n_bins:
                neuron_idx = cluster_to_idx.get(cluster)
                if neuron_idx is not None:
                    counts[t, neuron_idx, bin_idx] += 1

    return counts


def filter_by_region(
    counts: np.ndarray, cluster_regions: np.ndarray, target_region: str
) -> np.ndarray:
    """Select neurons from a specific brain region.

    Returns (n_trials, n_region_neurons, n_bins).
    """
    # Normalise to a clean 1-D str array (handles bytes, Series, object arrays)
    regions = _to_str_array(cluster_regions)
    target = str(target_region).strip()

    # Match exact region OR layer-specific sub-regions from Allen CCF
    # Atlas IDs resolve to layer labels like VISp4/VISp5/VISp6a; "VISp" should match all
    # A sub-region match requires the suffix to start with a digit (layer) or "-" (e.g. DG-mo)
    def _matches(r: str) -> bool:
        r = r.strip()
        if r == target:
            return True
        if r.startswith(target) and len(r) > len(target):
            suffix_char = r[len(target)]
            return suffix_char.isdigit() or suffix_char == "-"
        return False

    mask = np.array([_matches(str(r)) for r in regions], dtype=bool)

    n_match = int(mask.sum())
    logger.debug(
        "filter_by_region(%s): %d/%d neurons match (regions dtype=%s, sample=%s)",
        target, n_match, len(regions), regions.dtype,
        list(regions[:5]) if len(regions) > 0 else "empty",
    )
    if n_match == 0:
        unique_regions = np.unique(regions)
        logger.warning(
            "filter_by_region(%s): 0 matches! unique regions (%d): %s",
            target, len(unique_regions),
            list(unique_regions[:20]),
        )

    return counts[:, mask, :]
