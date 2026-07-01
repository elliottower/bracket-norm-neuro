"""Shared data loading and helpers for batch3 bundle experiments.

All experiments in this batch run on cached Steinmetz data, CPU-only.
They call load_bundle_data() which returns everything needed.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import svd, polar
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from geometry.subspace import fit_lda_subspace, fit_pca_subspace

RESULTS_DIR = PROJECT_ROOT / "results_bundle" / "batch3"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "steinmetz"

# Computed from Zatka-Haas et al. (2021) raw .mat file (exp47b).
# total_effect = sqrt(Δleft² + Δright² + Δnogo²) per laser coordinate,
# averaged across coordinates mapped to each Steinmetz region.
SILENCING_EFFECTS = {
    "PL": 0.3333, "ORB": 0.3085, "VISpm": 0.2248, "MOp": 0.1869,
    "SSp": 0.1862, "VISl": 0.1722, "SSs": 0.1600, "MOs": 0.1529,
    "ACA": 0.1451, "RSP": 0.1421, "VISp": 0.1414, "VISam": 0.0818,
}

# Decision-related window: 375–875ms post-stimulus (matches exp57 / main paper).
TIME_WINDOW = slice(15, 35)
MIN_NEURONS = 10
MIN_TRIALS_PER_CLASS = 15


def save_results(name, data, results_dir=None):
    rd = results_dir or RESULTS_DIR
    rd.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = rd / f"{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def correlate_with_silencing(node_values):
    matched = []
    for area, effect in SILENCING_EFFECTS.items():
        if area in node_values:
            matched.append((node_values[area], effect))
    if len(matched) < 4:
        return None
    x, y = zip(*matched)
    rho, p = spearmanr(x, y)
    return {
        "spearman_rho": float(rho), "p_value": float(p),
        "n_matched": len(matched),
        "matched_regions": [a for a in SILENCING_EFFECTS if a in node_values],
    }


def procrustes_rotation(P_from, P_to):
    M = P_from.T @ P_to
    U, _, Vt = svd(M)
    return U @ Vt


def edge_distance_from_R(R):
    k = R.shape[0]
    det = np.linalg.det(R)
    if det < 0:
        return np.pi
    trace = np.trace(R)
    cos_val = np.clip((trace - (k - 2)) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_val))


def get_R(restriction_maps, s, t):
    if (s, t) in restriction_maps:
        return restriction_maps[(s, t)]
    elif (t, s) in restriction_maps:
        return restriction_maps[(t, s)].T
    return None


def find_triangles(edges, adj):
    seen = set()
    triangles = []
    for s, t in edges:
        common = adj.get(s, set()) & adj.get(t, set())
        for w in common:
            tri = tuple(sorted([s, t, w]))
            if tri not in seen:
                seen.add(tri)
                triangles.append(tri)
    return triangles


def build_coboundary(restriction_maps, all_areas, k_dim):
    areas = sorted(all_areas)
    area_idx = {a: i for i, a in enumerate(areas)}
    n_areas = len(areas)
    edges = list(restriction_maps.keys())
    n_edges = len(edges)
    k = k_dim

    delta = np.zeros((n_edges * k, n_areas * k))
    for e_idx, (s, t) in enumerate(edges):
        R = restriction_maps[(s, t)]
        i_s, i_t = area_idx[s], area_idx[t]
        delta[e_idx*k:(e_idx+1)*k, i_s*k:(i_s+1)*k] = R
        delta[e_idx*k:(e_idx+1)*k, i_t*k:(i_t+1)*k] = -np.eye(k)

    return delta, edges, areas


def build_scalar_simplicial(all_areas, edges, triangles):
    """Build the SCALAR simplicial chain complex (no connection / bundle structure).

    δ₀: C⁰(nodes) → C¹(edges)     — standard graph coboundary
    δ₁: C¹(edges) → C²(triangles)  — standard triangle coboundary

    These always satisfy δ₁ ∘ δ₀ = 0 (exactness of the cochain complex).
    Use this for Hodge decomposition of scalar 1-forms (e.g. edge distances).

    Returns (delta_0, delta_1) as dense matrices.
    """
    areas = sorted(all_areas)
    area_idx = {a: i for i, a in enumerate(areas)}
    edge_list = list(edges)
    edge_idx = {e: i for i, e in enumerate(edge_list)}

    n_v = len(areas)
    n_e = len(edge_list)
    n_t = len(triangles)

    # δ₀: (n_edges, n_nodes) — δ₀(f)[e(s,t)] = f(t) - f(s)
    delta_0 = np.zeros((n_e, n_v))
    for e_i, (s, t) in enumerate(edge_list):
        delta_0[e_i, area_idx[s]] = -1.0
        delta_0[e_i, area_idx[t]] = 1.0

    # δ₁: (n_triangles, n_edges) — δ₁(ω)[tri(a,b,c)] = ω(a,b) + ω(b,c) - ω(a,c)
    delta_1 = np.zeros((n_t, n_e))
    edge_set = set(edge_list)
    for t_i, tri in enumerate(triangles):
        a, b, c = tri
        for (s, t), sign in [((a, b), 1.0), ((b, c), 1.0), ((a, c), -1.0)]:
            if (s, t) in edge_idx:
                delta_1[t_i, edge_idx[(s, t)]] = sign
            elif (t, s) in edge_idx:
                delta_1[t_i, edge_idx[(t, s)]] = -sign

    return delta_0, delta_1, areas, edge_list


def hodge_decompose_edge(delta_0, delta_1, omega):
    """Full Hodge decomposition of a scalar edge 1-form omega.

    omega = grad + curl + harmonic

    Uses the SCALAR simplicial complex (delta_0, delta_1 must satisfy δ₁∘δ₀=0).
    Fractions sum to 1.0 (within numerical precision).

    Returns dict with norms and fractions.
    """
    total_norm2 = float(np.linalg.norm(omega)**2)
    if total_norm2 < 1e-20:
        return {"grad_frac": 0, "curl_frac": 0, "harm_frac": 0, "total_norm": 0}

    # Gradient component: projection onto im(delta_0)
    # im(δ₀) is spanned by the column space of δ₀.T (== row space of δ₀)
    # Equivalently: eigenvectors of δ₀.T @ δ₀ with nonzero eigenvalue, projected via δ₀
    L_down = delta_0 @ delta_0.T  # (n_e, n_e) "down Laplacian"
    evals_down, evecs_down = np.linalg.eigh(L_down)
    tol = 1e-8 * max(1.0, abs(evals_down).max())
    grad_mask = evals_down > tol
    if grad_mask.sum() > 0:
        V_grad = evecs_down[:, grad_mask]
        grad_part = V_grad @ (V_grad.T @ omega)
    else:
        grad_part = np.zeros_like(omega)

    # Curl component: projection onto im(delta_1.T)
    if delta_1.shape[0] > 0:
        L_up = delta_1.T @ delta_1  # (n_e, n_e) "up Laplacian"
        evals_up, evecs_up = np.linalg.eigh(L_up)
        tol_up = 1e-8 * max(1.0, abs(evals_up).max())
        curl_mask = evals_up > tol_up
        if curl_mask.sum() > 0:
            V_curl = evecs_up[:, curl_mask]
            curl_part = V_curl @ (V_curl.T @ omega)
        else:
            curl_part = np.zeros_like(omega)
    else:
        curl_part = np.zeros_like(omega)

    # Harmonic: residual (should be in ker(L₁) = ker(δ₀.T) ∩ ker(δ₁))
    harm_part = omega - grad_part - curl_part

    grad_norm2 = float(np.linalg.norm(grad_part)**2)
    curl_norm2 = float(np.linalg.norm(curl_part)**2)
    harm_norm2 = float(np.linalg.norm(harm_part)**2)

    return {
        "grad_frac": grad_norm2 / total_norm2,
        "curl_frac": curl_norm2 / total_norm2,
        "harm_frac": harm_norm2 / total_norm2,
        "grad_norm": float(np.sqrt(grad_norm2)),
        "curl_norm": float(np.sqrt(curl_norm2)),
        "harm_norm": float(np.sqrt(harm_norm2)),
        "total_norm": float(np.sqrt(total_norm2)),
    }


def load_bundle_data(k_dim=2):
    """Load Steinmetz, compute projections, restriction maps, everything."""
    print(f"[{datetime.now(timezone.utc).isoformat()}] Loading Steinmetz data...")
    parts = [np.load(CACHE_DIR / f"steinmetz_part{i}.npz", allow_pickle=True)["dat"] for i in range(3)]
    sessions = np.concatenate(parts)
    print(f"  {len(sessions)} sessions")

    print(f"[{datetime.now(timezone.utc).isoformat()}] Computing projections (k={k_dim})...")
    session_projs = []
    region_coords = {}
    region_neuron_counts = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        spks = sess["spks"]
        response = sess["response"]
        areas = sess["brain_area"]
        ccf = sess["ccf"]

        choice_mask = (response == 1) | (response == -1)
        if choice_mask.sum() < 2 * MIN_TRIALS_PER_CLASS:
            session_projs.append({})
            continue

        activity = spks[:, choice_mask, TIME_WINDOW].mean(axis=2).T
        labels = response[choice_mask]

        if min((labels == -1).sum(), (labels == 1).sum()) < MIN_TRIALS_PER_CLASS:
            session_projs.append({})
            continue

        sess_proj = {}
        for area in np.unique(areas):
            neuron_mask = areas == area
            if neuron_mask.sum() < MIN_NEURONS:
                continue

            area_str = str(area)
            area_activity = activity[:, neuron_mask]

            try:
                lda_basis = fit_lda_subspace(area_activity, labels, k=k_dim)
                pca_basis = fit_pca_subspace(area_activity, labels, k=k_dim)
                sess_proj[area_str] = {
                    "lda_proj": area_activity @ lda_basis,
                    "pca_proj": area_activity @ pca_basis,
                }
            except Exception:
                continue

            if area_str not in region_coords:
                region_coords[area_str] = []
                region_neuron_counts[area_str] = []
            region_coords[area_str].append(ccf[neuron_mask].mean(axis=0))
            region_neuron_counts[area_str].append((neuron_mask.sum(), sess_idx))

        session_projs.append(sess_proj)

    region_info = {}
    for area in region_coords:
        coords = np.mean(region_coords[area], axis=0)
        n_sessions = len(region_neuron_counts[area])
        region_info[area] = {"coords": coords, "n_sessions": n_sessions}

    all_areas = sorted(region_info.keys())
    n_valid = sum(1 for sp in session_projs if sp)
    print(f"  {len(all_areas)} regions, {n_valid} sessions with data")

    print(f"[{datetime.now(timezone.utc).isoformat()}] Building adjacency...")
    areas_list = sorted(region_info.keys())
    coords = np.array([region_info[a]["coords"] for a in areas_list])
    dists = squareform(pdist(coords))
    edges = []
    adj = {a: set() for a in areas_list}
    for i in range(len(areas_list)):
        for j in range(i + 1, len(areas_list)):
            if dists[i, j] < 2000.0:
                edges.append((areas_list[i], areas_list[j]))
                adj[areas_list[i]].add(areas_list[j])
                adj[areas_list[j]].add(areas_list[i])
    print(f"  {len(edges)} edges")

    print(f"[{datetime.now(timezone.utc).isoformat()}] Computing restriction maps...")
    lda_maps, lda_n = _compute_rmaps(session_projs, edges, k_dim, "lda")
    pca_maps, pca_n = _compute_rmaps(session_projs, edges, k_dim, "pca")
    print(f"  LDA: {len(lda_maps)} edges, PCA: {len(pca_maps)} edges")

    return {
        "sessions": sessions,
        "session_projs": session_projs,
        "region_info": region_info,
        "all_areas": all_areas,
        "edges": edges,
        "adj": adj,
        "lda_maps": lda_maps,
        "pca_maps": pca_maps,
        "k_dim": k_dim,
    }


def _compute_rmaps(session_projs, edges, k_dim, method):
    proj_key = f"{method}_proj"
    rmaps = {}
    n_sess = {}
    for s, t in edges:
        R_sum = np.zeros((k_dim, k_dim))
        n = 0
        for sp in session_projs:
            if s not in sp or t not in sp:
                continue
            P_s = sp[s][proj_key]
            P_t = sp[t][proj_key]
            P_s_n = P_s / (np.linalg.norm(P_s, axis=0, keepdims=True) + 1e-10)
            P_t_n = P_t / (np.linalg.norm(P_t, axis=0, keepdims=True) + 1e-10)
            R_sum += procrustes_rotation(P_s_n, P_t_n)
            n += 1
        if n > 0:
            U, _, Vt = svd(R_sum)
            rmaps[(s, t)] = U @ Vt
            n_sess[(s, t)] = n
    return rmaps, n_sess
