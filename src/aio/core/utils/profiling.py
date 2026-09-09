"""Cost accounting (spec §2.7): valid triples, chunked-path work, MAC estimate, timing."""
from __future__ import annotations
import time
import torch

def triple_count(mask):                            # valid triples: T = Σ_b C(n_b,2)(n_b-2)
    n = mask.sum(1).long()
    return int((n * (n - 1) // 2 * (n - 2)).clamp_min(0).sum())


def chunked_path_work(anc, mask):
    """Slots executed by the *chunked anchor path* (anchors × padded N) vs valid triples.  Not an
    accounting of the dense-matmul path (all pairs before gathering), extra pools, bonds, or checkpoint
    recomputation — use measured profiling for comparisons."""
    n = mask.sum(1).long(); A, N = anc.shape[0], mask.shape[1]
    valid = int((n[anc[:, 0]] - 2).clamp_min(0).sum()) if A else 0
    return {"anchors": A, "slots_executed": A * N, "valid_triples": valid,
            "candidate_padding_waste": 1 - valid / max(A * N, 1)}


def mac_estimate(triples, D, passes=4.0):          # forming t (2), score (1), pool (1)
    return passes * triples * D


def time_fn(fn, iters=10, warmup=3):
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for _ in range(warmup): fn()
    ts = []
    for _ in range(iters):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]