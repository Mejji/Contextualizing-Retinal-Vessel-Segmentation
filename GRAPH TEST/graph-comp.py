# graph-val.py — exact validator + parity stats (no pass/fail)
# Prints (1) detailed per-pair logs, (2) a final ASCII summary table across images,
# and also writes CSV/JSON artifacts you can reuse.

import os, sys, pickle, math, time, hashlib, json, csv, argparse
from typing import Any, Dict, List, Tuple, Optional, Iterable
import networkx as nx
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLOW_SUFFIX = "-shin.graph_res"
FAST_SUFFIX = "-ours.graph_res"

# ---------- Defaults (CLI-overridable) ----------
N_WORKERS = max(1, cpu_count() - 1)
PRINT_EVERY_SOURCES = 50
WEIGHT_TOL_DEFAULT = 1e-6
# ------------------------------------------------

# ---------------- Robust graph extraction ----------------
def _is_int_seq(x: Any) -> bool:
    try:
        if isinstance(x, (bytes, str)): return False
        it = list(x)
        if len(it) == 0: return False
        for v in it:
            if not isinstance(v, int) and not (hasattr(v, "dtype") and str(getattr(v, "dtype")).startswith("int")):
                return False
        return True
    except Exception:
        return False

def _coerce_simple_undirected(G: Any) -> nx.Graph:
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return nx.Graph(G)
    if isinstance(G, nx.DiGraph):
        return nx.Graph(G)
    if isinstance(G, nx.Graph):
        return G.copy()
    raise TypeError

def _build_from_edges(nodes: Optional[Iterable[int]], edges: Iterable[Tuple[int,int]]) -> nx.Graph:
    H = nx.Graph()
    if nodes is not None:
        H.add_nodes_from(nodes)
    for e in edges:
        if len(e) >= 2:
            H.add_edge(int(e[0]), int(e[1]))
    return H

def _try_extract_from_dict(d: dict) -> Optional[nx.Graph]:
    for k in ("graph", "G", "nx_graph"):
        if k in d and isinstance(d[k], (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            return _coerce_simple_undirected(d[k])
    if "edges" in d and isinstance(d["edges"], (list, tuple)):
        return _build_from_edges(d.get("nodes"), d["edges"])
    if "src" in d and "dst" in d and _is_int_seq(d["src"]) and _is_int_seq(d["dst"]):
        src, dst = list(d["src"]), list(d["dst"])
        nmax = max(src + dst) + 1 if (src and dst) else 0
        return _build_from_edges(range(nmax), zip(src, dst))
    for v in d.values():
        if isinstance(v, dict):
            g = _try_extract_from_dict(v)
            if g is not None: return g
    return None

def _try_extract_from_tuple_like(obj: Any) -> Optional[nx.Graph]:
    seq = list(obj)
    for item in seq:
        if isinstance(item, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            return _coerce_simple_undirected(item)
    for item in seq:
        if isinstance(item, dict):
            g = _try_extract_from_dict(item)
            if g is not None: return g
    if len(seq) >= 2 and isinstance(seq[0], (list, set, tuple)) and isinstance(seq[1], (list, tuple)):
        if seq[1] and isinstance(seq[1][0], (list, tuple)) and len(seq[1][0]) >= 2:
            return _build_from_edges(seq[0], seq[1])
    int_arrays = [s for s in seq if _is_int_seq(s)]
    if len(int_arrays) >= 2:
        a = list(int_arrays[0]); b = list(int_arrays[1])
        if len(a) == len(b) and len(a) > 0:
            nmax = max(max(a), max(b)) + 1
            return _build_from_edges(range(nmax), zip(a, b))
    return None

def _extract_graph(obj: Any) -> nx.Graph:
    if isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        return _coerce_simple_undirected(obj)
    if isinstance(obj, dict):
        g = _try_extract_from_dict(obj)
        if g is not None: return g
    if isinstance(obj, (tuple, list)):
        g = _try_extract_from_tuple_like(obj)
        if g is not None: return g
    raise TypeError(f"Could not extract a NetworkX graph from object type {type(obj)}")

def load_graph(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return _extract_graph(obj)

# ---------------- Parity stats (no pass/fail) ----------------
def _edge_set(G: nx.Graph):
    return {tuple(sorted(e)) for e in G.edges()}

def _weight_map(G: nx.Graph):
    return {tuple(sorted((u, v))): float(G[u][v].get('weight', 1.0)) for u, v in G.edges()}

def _attr_arrays(G: nx.Graph):
    ids = sorted(G.nodes())
    ys = [int(G.nodes[i].get('y', -1)) for i in ids]
    xs = [int(G.nodes[i].get('x', -1)) for i in ids]
    return ids, ys, xs

def _sha256_of_int_list(lst: List[int]) -> str:
    h = hashlib.sha256()
    for v in lst:
        h.update(int(v).to_bytes(4, byteorder='little', signed=True))
    return h.hexdigest()

def parity_stats(Gs: nx.Graph, Gf: nx.Graph, tol_weight: float) -> Dict[str, Any]:
    ids_s, ys_s, xs_s = _attr_arrays(Gs)
    ids_f, ys_f, xs_f = _attr_arrays(Gf)

    nodes_equal = (ids_s == ids_f)
    y_equal = (ys_s == ys_f)
    x_equal = (xs_s == xs_f)

    Es, Ef = _edge_set(Gs), _edge_set(Gf)
    inter = Es & Ef
    union = Es | Ef
    jacc = len(inter) / len(union) if union else 1.0

    Ws, Wf = _weight_map(Gs), _weight_map(Gf)
    mae, wmax, n_bad, worst = 0.0, 0.0, 0, []
    if inter:
        diffs = []
        for e in inter:
            d = abs(Ws[e] - Wf[e])
            diffs.append(d)
            if d > wmax: wmax = d
            if d > tol_weight and len(worst) < 5:
                worst.append((e, Ws[e], Wf[e], d))
            if d > tol_weight: n_bad += 1
        mae = sum(diffs)/len(diffs)

    return {
        "num_nodes": len(ids_s),
        "nodes_equal": nodes_equal,
        "y_equal": y_equal,
        "x_equal": x_equal,
        "y_hash": _sha256_of_int_list(ys_s),
        "x_hash": _sha256_of_int_list(xs_s),
        "edges_slow": len(Es),
        "edges_fast": len(Ef),
        "edge_jaccard": jacc,
        "weight_mae": mae,
        "weight_max": wmax,
        "weight_outside_tol": n_bad,
        "weight_worst": worst,
    }

# ---------------- Exact metrics with multiprocessing ----------------
def _format_dur(sec: float) -> str:
    if not (sec >= 0) or math.isinf(sec) or math.isnan(sec): return "unknown"
    sec = int(round(sec))
    m, s = divmod(sec, 60); h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

def _bfs_sum_and_ecc(args):
    H_nodes, H_edges, sources = args
    H = nx.Graph()
    H.add_nodes_from(H_nodes)
    H.add_edges_from(H_edges)
    local_sum = 0
    local_pairs = 0
    local_diam = 0
    for s in sources:
        sp = nx.single_source_shortest_path_length(H, s)
        local_sum += sum(sp.values())
        local_pairs += (len(sp) - 1)
        if sp:
            far = max(sp.values())
            if far > local_diam:
                local_diam = far
    return local_sum, local_pairs, local_diam, len(sources)

def exact_aspl_and_diameter(H: nx.Graph, label: str = "", quiet: bool = False) -> Tuple[float, int]:
    nodes = list(H.nodes())
    node_idx = {u: i for i, u in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in H.edges()]
    N = len(nodes)
    if N <= 1:
        return float("nan"), 0

    chunks: List[List[int]] = [[] for _ in range(max(1, N_WORKERS))]
    for i in range(N): chunks[i % len(chunks)].append(i)

    args = []
    for c in chunks:
        if not c: continue
        args.append((list(range(N)), edges, c))

    start = time.time()
    done_src = 0
    sum_dist = 0
    total_pairs = 0
    diam = 0

    with Pool(processes=len(args)) as pool:
        for (loc_sum, loc_pairs, loc_diam, nsrc) in pool.imap_unordered(_bfs_sum_and_ecc, args):
            sum_dist += loc_sum
            total_pairs += loc_pairs
            diam = max(diam, loc_diam)
            done_src += nsrc
            if not quiet and (done_src % PRINT_EVERY_SOURCES == 0 or done_src == N):
                frac = done_src / N
                elapsed = time.time() - start
                eta = elapsed * (1 - frac) / max(frac, 1e-9)
                print(f"    [{label}] sources {done_src}/{N} | elapsed={_format_dur(elapsed)} | ETA={_format_dur(eta)}",
                      flush=True)

    aspl = sum_dist / total_pairs if total_pairs > 0 else float("nan")
    return aspl, diam

def metrics_exact(G: nx.Graph, who: str = "", quiet: bool = False) -> Dict[str, float]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0
    avg_degree = (2.0 * m / n) if n > 0 else 0.0

    comps = [list(c) for c in nx.connected_components(G)]
    comps_ge2 = [c for c in comps if len(c) >= 2]

    total_pairs = sum((len(c) * (len(c) - 1)) // 2 for c in comps_ge2)
    sum_len = 0.0
    diam = 0

    acc_pairs = 0
    t0 = time.time()
    for idx, nodes in enumerate(comps_ge2, start=1):
        H = G.subgraph(nodes).copy()
        k = H.number_of_nodes()
        pairs = k * (k - 1) // 2
        if not quiet:
            print(f"{who} CC {idx}/{len(comps_ge2)} | nodes={k} (exact)", flush=True)
        aspl_c, diam_c = exact_aspl_and_diameter(H, label=f"{who}CC{idx}", quiet=quiet)
        sum_len += aspl_c * pairs
        diam = max(diam, diam_c)
        acc_pairs += pairs
        if not quiet:
            frac = acc_pairs / total_pairs if total_pairs else 1.0
            elapsed = time.time() - t0
            eta = elapsed * (1 - frac) / max(frac, 1e-9)
            print(f"{who}   progress={frac*100:5.1f}% | elapsed={_format_dur(elapsed)} | ETA={_format_dur(eta)}",
                  flush=True)

    aspl = (sum_len / total_pairs) if total_pairs > 0 else float("nan")

    return {
        "nodes": n,
        "edges": m,
        "density": density,
        "avg_degree": avg_degree,
        "avg_path_len": aspl,
        "diameter": diam,
        "num_components": len(comps),
        "num_components_ge2": len(comps_ge2),
        "num_isolates": sum(1 for c in comps if len(c) == 1),
    }

def pct_diff(a: float, b: float) -> float:
    if a == 0 or (isinstance(a, float) and (math.isnan(a) or math.isinf(a))):
        return float("nan")
    return 100.0 * (b - a) / a

# ---------------- CLI & Reporting ----------------
def parse_cli():
    ap = argparse.ArgumentParser(description="Validate Shin vs Fast graphs with exact metrics.")
    ap.add_argument("--weight-tol", type=float, default=WEIGHT_TOL_DEFAULT, help="tolerance for weight parity stats")
    ap.add_argument("--no-progress", action="store_true", help="suppress per-CC progress bars")
    ap.add_argument("--cols", type=str, default="", help="comma-separated columns for the final summary table")
    return ap.parse_args()

DEFAULT_COLS = [
    "pair",
    "edge_jaccard",
    "shin_nodes","ours_nodes","delta_pct_nodes",
    "shin_edges","ours_edges","delta_pct_edges",
    "shin_avg_path_len","ours_avg_path_len","delta_pct_avg_path_len",
    "shin_diameter","ours_diameter","delta_pct_diameter",
]

def _fmt_float(x, prec=6):
    return "NaN" if (isinstance(x, float) and math.isnan(x)) else (f"{x:.{prec}f}" if isinstance(x, float) else str(x))

def report_pair(base: str, slow_path: str, fast_path: str, weight_tol: float, quiet: bool) -> Dict[str, Any]:
    print("=" * 80)
    print(f"PAIR: {base}")
    print(f"slow (Shin): {slow_path}")
    print(f"fast (Ours): {fast_path}")

    Gs = load_graph(slow_path)
    Gf = load_graph(fast_path)

    # ---------- PARITY STATS ----------
    print("\nPARITY STATS (nodes/coords/edges/weights)")
    pst = parity_stats(Gs, Gf, tol_weight=weight_tol)
    print(f"  Nodes equal: {pst['nodes_equal']} | |V|={pst['num_nodes']}")
    print(f"  Coords equal: y={pst['y_equal']} x={pst['x_equal']}")
    print(f"  Coord hashes: y={pst['y_hash'][:12]}… x={pst['x_hash'][:12]}…")
    print(f"  Edge Jaccard: {pst['edge_jaccard']:.4f} | |E| slow={pst['edges_slow']}, fast={pst['edges_fast']}")
    print(f"  Weight MAE: {_fmt_float(pst['weight_mae'], 9)} | MAX: {_fmt_float(pst['weight_max'], 9)} "
          f"| outside tol: {pst['weight_outside_tol']}")
    if pst["weight_outside_tol"] > 0 and pst["weight_worst"]:
        print("  Worst edges (u,v, w_slow, w_fast, |Δ|):")
        for (e, ws, wf, d) in pst["weight_worst"]:
            print(f"    {e}  {ws:.9f}  {wf:.9f}  {d:.9f}")
    # ---------- /PARITY STATS ----------

    print("\n[Shin] computing EXACT metrics …", flush=True)
    ms = metrics_exact(Gs, who="[Shin]", quiet=quiet)
    print("\n[Ours] computing EXACT metrics …", flush=True)
    mf = metrics_exact(Gf, who="[Ours]", quiet=quiet)

    fields = [
        ("nodes", "Nodes (|V|)", 0),
        ("edges", "Edges (|E|)", 0),
        ("density", "Graph density", 6),
        ("avg_degree", "Average degree", 6),
        ("avg_path_len", "Avg shortest path length (ASPL)", 6),
        ("diameter", "Graph diameter", 0),
        ("num_components", "# components (all)", 0),
        ("num_components_ge2", "# components (size ≥ 2)", 0),
        ("num_isolates", "# isolates", 0),
    ]

    print("\nMetric".ljust(40), "Shin (slow)".ljust(20), "Ours (fast)".ljust(20), "Δ% vs Shin")
    print("-" * 80)
    for key, label, prec in fields:
        s = ms[key]; f = mf[key]
        print(label.ljust(40), _fmt_float(s, prec).ljust(20), _fmt_float(f, prec).ljust(20), f"{pct_diff(s, f):+.3f}%")
    print()

    # machine-friendly record for table/CSV
    record = {
        "pair": base,
        "edge_jaccard": float(pst["edge_jaccard"]),
    }
    for k in ("nodes","edges","density","avg_degree","avg_path_len","diameter",
              "num_components","num_components_ge2","num_isolates"):
        record[f"shin_{k}"] = ms[k]
        record[f"ours_{k}"] = mf[k]
        record[f"delta_pct_{k}"] = pct_diff(ms[k], mf[k]) if isinstance(ms[k], (int,float)) else float("nan")
    return record

# ---------------- Pair discovery & main ----------------
def find_pairs_here(folder: str):
    files = [f for f in os.listdir(folder) if f.endswith(".graph_res")]
    slow = {f[:-len(SLOW_SUFFIX)]: os.path.join(folder, f)
            for f in files if f.endswith(SLOW_SUFFIX)}
    fast = {f[:-len(FAST_SUFFIX)]: os.path.join(folder, f)
            for f in files if f.endswith(FAST_SUFFIX)}
    common = sorted(set(slow.keys()) & set(fast.keys()))
    return [(b, slow[b], fast[b]) for b in common], slow, fast

def _write_csv_json(rows: List[Dict[str,Any]], csv_path: str, json_path: str):
    if not rows:
        return
    fieldnames = sorted(set().union(*[r.keys() for r in rows]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int,float)) and not (math.isnan(x) or math.isinf(x))]
    return sum(xs)/len(xs) if xs else float("nan")

def _parse_cols(s: str) -> List[str]:
    if not s: return DEFAULT_COLS
    cols = [c.strip() for c in s.split(",") if c.strip()]
    return cols or DEFAULT_COLS

def _render_ascii_table(rows: List[Dict[str,Any]], cols: List[str]) -> str:
    # make strings & col widths
    S = [[str("NaN" if (isinstance(r.get(c), float) and (math.isnan(r[c]) or math.isinf(r[c]))) else r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), max((len(row[i]) for row in S), default=0)) for i, c in enumerate(cols)]
    def line(char="-"):
        return "+" + "+".join(char*(w+2) for w in widths) + "+"
    out = []
    out.append(line("="))
    # header
    hdr = "|" + "|".join(f" {c.ljust(widths[i])} " for i, c in enumerate(cols)) + "|"
    out.append(hdr)
    out.append(line("="))
    # rows
    for r in S:
        out.append("|" + "|".join(f" {r[i].ljust(widths[i])} " for i in range(len(cols))) + "|")
    out.append(line("="))
    return "\n".join(out)

def main():
    args = parse_cli()
    weight_tol = float(args.weight_tol)
    cols_out = _parse_cols(args.cols)

    pairs, slow_only, fast_only = find_pairs_here(SCRIPT_DIR)
    if not pairs:
        print("No '<base>-shin.graph_res' / '<base>-ours.graph_res' pairs found here.")
        if slow_only: print("Found shin-only:", ", ".join(sorted(slow_only.keys())))
        if fast_only: print("Found ours-only:", ", ".join(sorted(fast_only.keys())))
        sys.exit(1)

    compiled: List[Dict[str,Any]] = []

    for base, slow_path, fast_path in pairs:
        rec = report_pair(base, slow_path, fast_path, weight_tol=weight_tol, quiet=args.no_progress)
        compiled.append(rec)

    # ---------- FINAL PER-IMAGE SUMMARY TABLE ----------
    print("=" * 80)
    print("FINAL SUMMARY TABLE (per image)")
    table_rows = []
    for r in compiled:
        row = {c: "" for c in cols_out}
        for c in cols_out:
            if c in r:
                row[c] = r[c]
            elif c.startswith("shin_") or c.startswith("ours_") or c.startswith("delta_pct_"):
                row[c] = r.get(c, "")
            elif c == "pair":
                row[c] = r["pair"]
        table_rows.append(row)
    print(_render_ascii_table(table_rows, cols_out))

    # ---------- COMPILED OVERALL SUMMARY ----------
    print("\nOVERALL SUMMARY")
    print(f"- pairs: {len(compiled)}")
    mean_jacc = _mean([r["edge_jaccard"] for r in compiled])
    print(f"- mean edge Jaccard: {mean_jacc:.6f}")

    keys = ["nodes","edges","density","avg_degree","avg_path_len","diameter",
            "num_components","num_components_ge2","num_isolates"]
    print("\nMean Δ% vs Shin (across pairs):")
    for k in keys:
        m = _mean([r[f"delta_pct_{k}"] for r in compiled])
        print(f"  {k:>20}: {m:+.3f}%")

    # ---------- SAVE ARTIFACTS ----------
    csv_path = os.path.join(SCRIPT_DIR, "graph_val_summary.csv")
    json_path = os.path.join(SCRIPT_DIR, "graph_val_summary.json")
    _write_csv_json(compiled, csv_path, json_path)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
