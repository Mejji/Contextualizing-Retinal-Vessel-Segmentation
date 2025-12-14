
#!/usr/bin/env python3
"""Graph parity validator: compares two graphs (or two folders of graphs)
and decides PASS/FAIL with clear metrics.

Usage examples:
  # Compare two gpickle graphs
  python graph_validator.py --baseline /path/to/original.graph_res --candidate /path/to/new.graph_res

  # Compare two folders containing .graph_res files (paired by basename)
  python graph_validator.py --baseline_dir /path/to/originals --candidate_dir /path/to/news --report_csv parity_report.csv

Decision rule (default):
  PASS if (edge_jaccard_tol >= 0.98) AND (node_match_tol >= 0.99) AND (deg_l1 <= 0.05).
  Otherwise FAIL.

You can override thresholds via CLI flags.
"""
import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import networkx as nx
import pickle as pkl
try:
    from networkx.readwrite import gpickle as nx_gpickle
except Exception:
    nx_gpickle = None

try:
    from scipy.spatial import cKDTree
    HAS_CKDTREE = True
except Exception:
    HAS_CKDTREE = False


def read_graph(path: str) -> nx.Graph:
    # Robust read across NetworkX versions
    if hasattr(nx, 'read_gpickle'):
        G = nx.read_gpickle(path)
    elif nx_gpickle is not None and hasattr(nx_gpickle, 'read_gpickle'):
        G = nx_gpickle.read_gpickle(path)
    else:
        # Fallback to raw pickle
        with open(path, 'rb') as f:
            G = pkl.load(f)
    # Normalize attributes
    for n, d in list(G.nodes(data=True)):
        # Some pickles may store y/x as float; cast to int
        if 'y' in d and 'x' in d:
            try:
                yy = int(round(float(d['y'])))
                xx = int(round(float(d['x'])))
            except Exception:
                # Fall back if unavailable
                yy = d.get('y', 0)
                xx = d.get('x', 0)
            G.nodes[n]['y'] = yy
            G.nodes[n]['x'] = xx
        else:
            raise ValueError(f"Node {n} in {path} lacks 'y'/'x' attributes.")
    return G


def ids_coords_from_graph(G: nx.Graph) -> Tuple[List, np.ndarray, np.ndarray, Dict]:
    """Return (ids, ys, xs, id2idx)."""
    ids = list(G.nodes())
    ys = np.array([int(G.nodes[n]['y']) for n in ids], dtype=np.int32)
    xs = np.array([int(G.nodes[n]['x']) for n in ids], dtype=np.int32)
    id2idx = {n: i for i, n in enumerate(ids)}
    return ids, ys, xs, id2idx


def build_exact_node_set(ys: np.ndarray, xs: np.ndarray) -> Set[Tuple[int, int]]:
    return set(map(tuple, np.stack([ys, xs], axis=1).tolist()))


def greedy_match_with_tolerance(
        base_coords: np.ndarray,
        cand_coords: np.ndarray,
        tol: int) -> List[Optional[int]]:
    """Greedy one-to-one matcher mapping each candidate node (index in cand_coords)
    to an index in base_coords if within <= tol pixels (Chebyshev metric).
    Returns list of indices into base_coords, or None if unmatched.
    """
    N0 = base_coords.shape[0]
    N1 = cand_coords.shape[0]
    if tol <= 0:
        # exact matching via hash map
        base_map = {tuple(p): i for i, p in enumerate(base_coords.tolist())}
        out = [base_map.get(tuple(p), None) for p in cand_coords.tolist()]
        return out

    if HAS_CKDTREE and N0 > 0 and N1 > 0:
        # Use KDTree with Chebyshev distance (p=inf). cKDTree supports p=inf.
        tree = cKDTree(base_coords.astype(np.float32))
        idxs = tree.query_ball_point(cand_coords.astype(np.float32), r=float(tol), p=np.inf)
        # Greedy: process by fewest options first
        order = sorted(range(N1), key=lambda i: len(idxs[i]))
        used = set()
        mapping: List[Optional[int]] = [None] * N1
        for i in order:
            options = [j for j in idxs[i] if j not in used]
            if not options:
                continue
            # Choose the nearest (break ties by index)
            dists = np.max(np.abs(base_coords[options] - cand_coords[i]), axis=1)
            best = options[int(np.argmin(dists))]
            mapping[i] = best
            used.add(best)
        return mapping

    # Fallback O(N0*N1): acceptable for small graphs
    mapping: List[Optional[int]] = [None] * N1
    used = set()
    for i in range(N1):
        # Search in square window (2*tol+1)^2 around cand_coords[i]
        y, x = cand_coords[i]
        best_j = None
        best_d = None
        for j in range(N0):
            if j in used:
                continue
            d = max(abs(int(base_coords[j, 0]) - int(y)), abs(int(base_coords[j, 1]) - int(x)))
            if d <= tol and (best_d is None or d < best_d):
                best_d = d; best_j = j
        if best_j is not None:
            mapping[i] = best_j
            used.add(best_j)
    return mapping


def edge_set_from_graph(G: nx.Graph,
                        id2idx: Dict,
                        ys: np.ndarray,
                        xs: np.ndarray) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    edges = set()
    for u, v in G.edges():
        iu = id2idx[u]; iv = id2idx[v]
        a = (int(ys[iu]), int(xs[iu])); b = (int(ys[iv]), int(xs[iv]))
        if a <= b:
            edges.add((a, b))
        else:
            edges.add((b, a))
    return edges


def map_edges_via_node_mapping(
        G: nx.Graph,
        id2idx_cand: Dict,
        mapping_cand_to_base_idx: List[Optional[int]],
        base_coords: np.ndarray) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Map candidate graph edges to base coordinate space using the candidate->base node mapping."""
    edges = set()
    for u, v in G.edges():
        iu = id2idx_cand[u]; iv = id2idx_cand[v]
        mu = mapping_cand_to_base_idx[iu]; mv = mapping_cand_to_base_idx[iv]
        if mu is None or mv is None or mu == mv:
            continue
        a = (int(base_coords[mu, 0]), int(base_coords[mu, 1]))
        b = (int(base_coords[mv, 0]), int(base_coords[mv, 1]))
        if a <= b:
            edges.add((a, b))
        else:
            edges.add((b, a))
    return edges


def degree_distribution_L1(G0: nx.Graph, G1: nx.Graph) -> float:
    deg0 = np.array([d for _, d in G0.degree()], dtype=np.int32)
    deg1 = np.array([d for _, d in G1.degree()], dtype=np.int32)
    m = int(max(deg0.max(initial=0), deg1.max(initial=0)))
    h0 = np.bincount(deg0, minlength=m+1).astype(np.float64)
    h1 = np.bincount(deg1, minlength=m+1).astype(np.float64)
    if h0.sum() > 0: h0 /= h0.sum()
    if h1.sum() > 0: h1 /= h1.sum()
    return float(np.abs(h0 - h1).sum())


def jaccard(A: Set, B: Set) -> float:
    if not A and not B:
        return 1.0
    union = len(A | B)
    inter = len(A & B)
    return inter / union if union > 0 else 0.0


def compare_two_graphs(baseline_path: str,
                       candidate_path: str,
                       pixel_tol_soft: int = 1,
                       edge_thresh: float = 0.98,
                       node_thresh: float = 0.99,
                       deg_l1_thresh: float = 0.05) -> Tuple[dict, bool]:
    """Return metrics dict and boolean PASS/FAIL."""
    G0 = read_graph(baseline_path)
    G1 = read_graph(candidate_path)

    ids0, ys0, xs0, id2idx0 = ids_coords_from_graph(G0)
    ids1, ys1, xs1, id2idx1 = ids_coords_from_graph(G1)

    coords0 = np.stack([ys0, xs0], axis=1)
    coords1 = np.stack([ys1, xs1], axis=1)

    # Node match (exact and soft)
    set0 = build_exact_node_set(ys0, xs0)
    set1 = build_exact_node_set(ys1, xs1)
    exact_matches = len(set0 & set1)
    node_match_exact = exact_matches / max(1, len(set1))

    mapping_soft = greedy_match_with_tolerance(coords0, coords1, tol=pixel_tol_soft)
    matched_soft = sum(1 for m in mapping_soft if m is not None)
    node_match_soft = matched_soft / max(1, len(mapping_soft))

    # Edge Jaccard exact
    E0_exact = edge_set_from_graph(G0, id2idx0, ys0, xs0)
    E1_exact = edge_set_from_graph(G1, id2idx1, ys1, xs1)
    edge_jacc_exact = jaccard(E0_exact, E1_exact)

    # Edge Jaccard with soft mapping (map candidate edges into baseline coord space)
    E1_mapped = map_edges_via_node_mapping(G1, id2idx1, mapping_soft, coords0)
    edge_jacc_soft = jaccard(E0_exact, E1_mapped)

    # Degree distribution L1
    deg_l1 = degree_distribution_L1(G0, G1)

    # Connected components difference
    cc_diff = abs(nx.number_connected_components(G0) - nx.number_connected_components(G1))

    # Final decision (soft metrics)
    passed = (edge_jacc_soft >= edge_thresh) and (node_match_soft >= node_thresh) and (deg_l1 <= deg_l1_thresh)

    metrics = {
        'baseline': os.path.basename(baseline_path),
        'candidate': os.path.basename(candidate_path),
        'nodes_baseline': G0.number_of_nodes(),
        'nodes_candidate': G1.number_of_nodes(),
        'edges_baseline': G0.number_of_edges(),
        'edges_candidate': G1.number_of_edges(),
        'node_match_exact': round(node_match_exact, 6),
        'node_match_soft': round(node_match_soft, 6),
        'edge_jacc_exact': round(edge_jacc_exact, 6),
        'edge_jacc_soft': round(edge_jacc_soft, 6),
        'deg_l1': round(deg_l1, 6),
        'cc_diff': int(cc_diff),
        'result': 'PASS' if passed else 'FAIL'
    }
    return metrics, passed


def pair_graphs_by_basename(dirA: str, dirB: str) -> List[Tuple[str, str]]:
    filesA = {os.path.basename(p): os.path.join(dirA, p) for p in os.listdir(dirA) if p.endswith('.graph_res')}
    filesB = {os.path.basename(p): os.path.join(dirB, p) for p in os.listdir(dirB) if p.endswith('.graph_res')}
    common = sorted(set(filesA.keys()) & set(filesB.keys()))
    return [(filesA[k], filesB[k]) for k in common]


def main():
    ap = argparse.ArgumentParser(description='Validate graph parity between baseline and candidate graphs.')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--baseline', type=str, help='Path to baseline .graph_res file')
    g.add_argument('--baseline_dir', type=str, help='Directory of baseline .graph_res files')

    h = ap.add_mutually_exclusive_group(required=True)
    h.add_argument('--candidate', type=str, help='Path to candidate .graph_res file')
    h.add_argument('--candidate_dir', type=str, help='Directory of candidate .graph_res files')

    ap.add_argument('--pixel_tol_soft', type=int, default=1, help='Pixel tolerance for soft node/edge matching (Chebyshev distance). Default=1')
    ap.add_argument('--edge_thresh', type=float, default=0.98, help='PASS threshold for soft edge Jaccard. Default=0.98')
    ap.add_argument('--node_thresh', type=float, default=0.99, help='PASS threshold for soft node match rate. Default=0.99')
    ap.add_argument('--deg_l1_thresh', type=float, default=0.05, help='PASS threshold for degree-distribution L1 distance. Default=0.05')
    ap.add_argument('--report_csv', type=str, default=None, help='Optional CSV path to save per-file metrics')
    args = ap.parse_args()

    if args.baseline and args.candidate:
        pairs = [(args.baseline, args.candidate)]
    elif args.baseline_dir and args.candidate_dir:
        pairs = pair_graphs_by_basename(args.baseline_dir, args.candidate_dir)
        if not pairs:
            print('No matching .graph_res basenames found between the two directories.')
            return
    else:
        raise SystemExit('You must provide either two files or two directories.')

    all_pass = True
    rows = []
    for base_path, cand_path in pairs:
        m, ok = compare_two_graphs(base_path, cand_path,
                                   pixel_tol_soft=args.pixel_tol_soft,
                                   edge_thresh=args.edge_thresh,
                                   node_thresh=args.node_thresh,
                                   deg_l1_thresh=args.deg_l1_thresh)
        rows.append(m)
        all_pass = all_pass and ok
        print(f"{os.path.basename(base_path)} vs {os.path.basename(cand_path)} -> {m['result']} | "
              f"nodes: {m['nodes_baseline']} vs {m['nodes_candidate']} | edges: {m['edges_baseline']} vs {m['edges_candidate']} | "
              f"edge_jacc_soft={m['edge_jacc_soft']:.4f} | node_match_soft={m['node_match_soft']:.4f} | deg_l1={m['deg_l1']:.4f}")

    if args.report_csv:
        with open(args.report_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Report written to {args.report_csv}")

    print("\nOVERALL:", "PASS" if all_pass else "FAIL")


if __name__ == '__main__':
    main()
