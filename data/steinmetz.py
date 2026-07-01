"""Steinmetz et al. 2019 data loading.

~30,000 neurons, 42 brain regions, 39 sessions, 10 mice.
Multi-region simultaneous recording: all 42 regions from the same animal.

Data: 3 .npz files on OSF, pre-processed spike counts in 10ms bins.
For raw spike times (needed for holonomy experiments), use ONE API instead.
"""
import logging
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

URLS = [
    "https://osf.io/agvxh/download",
    "https://osf.io/uv3mw/download",
    "https://osf.io/ehmw2/download",
]
FILENAMES = [
    "steinmetz_part0.npz",
    "steinmetz_part1.npz",
    "steinmetz_part2.npz",
]


def _download_from_osf(url: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from OSF: {url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(local_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=local_path.name) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return local_path


def download(force: bool = False) -> list[Path]:
    """Download Steinmetz .npz files from OSF."""
    cache_dir = Path(__file__).parent.parent / "data" / "cache" / "steinmetz"
    paths = []
    for url, fname in zip(URLS, FILENAMES):
        local = cache_dir / fname
        if not force and local.exists():
            logger.info(f"Cached: {local}")
            paths.append(local)
            continue
        _download_from_osf(url, local)
        paths.append(local)
    return paths


def load_all() -> list[dict]:
    """Load all sessions. Returns list of session dicts.

    Each dict has keys matching the .npz fields:
        spks: (n_neurons, n_bins, n_trials) spike counts
        brain_area: (n_neurons,) region labels
        trough_to_peak: (n_neurons,) waveform features
        contrast_left, contrast_right: (n_trials,)
        response: (n_trials,) -1/0/1
        feedback_type: (n_trials,) -1/1
        response_time: (n_trials,)
        mouse_name, date_exp: session metadata
    """
    paths = download()
    sessions = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        alldat = data["dat"]
        for sess in alldat:
            sessions.append(dict(sess))
    logger.info(f"Loaded {len(sessions)} Steinmetz sessions")
    return sessions


def get_region_activity(
    session: dict, region: str, time_slice: slice | None = None
) -> np.ndarray | None:
    """Extract spike counts for neurons in a specific region.

    Args:
        session: dict from load_all()
        region: brain region acronym (e.g., 'MOs', 'VISp', 'ACA')
        time_slice: optional slice for time bins (default: full trial)

    Returns:
        (n_trials, n_region_neurons, n_bins) or None if region not present
    """
    mask = session["brain_area"] == region
    if mask.sum() == 0:
        return None
    spks = session["spks"][mask]  # (n_neurons, n_bins, n_trials)
    if time_slice is not None:
        spks = spks[:, time_slice, :]
    return spks.transpose(2, 0, 1)  # (n_trials, n_neurons, n_bins)


def get_choice_labels(session: dict) -> np.ndarray:
    """Get binary choice labels aligned to the spike trial dimension."""
    n_trials_spks = session["spks"].shape[2]
    response = session["response"]
    n = min(n_trials_spks, len(response))
    return (response[:n] > 0).astype(int)


def list_regions(session: dict, min_neurons: int = 10) -> list[str]:
    """List brain regions with at least min_neurons in this session."""
    areas = session["brain_area"]
    unique, counts = np.unique(areas, return_counts=True)
    return [a for a, c in zip(unique, counts) if c >= min_neurons]
