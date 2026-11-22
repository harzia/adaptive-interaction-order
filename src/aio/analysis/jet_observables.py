"""Hand-computed jet observables from constituent four-momenta (NumPy only).

Every observable is a function of the padded constituent four-momenta that the
tagger itself receives, so nothing here adds information the model was not
given. With the JetClass-II kin config those momenta are scaled so the jet has
pT = 500; dimensionful outputs (masses, kT) are therefore in units of
jet pT / 500 and never reveal the absolute jet pT.

Inputs are batches ``p[B, N, 4] = (px, py, pz, E)`` with a boolean ``mask[B, N]``.
Angles use rapidity and wrapped azimuth, as FastJet does.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np

R0 = 0.8
TINY = 1e-12
GROUPS = ("basic", "subjets", "nsub", "ecf", "pull", "lund", "softdrop")
LUND_EMISSIONS = 6
LUND_KT_THRESHOLDS = (1.0, 5.0, 20.0)  # scaled units: kT / jet pT = 0.002, 0.01, 0.04
SOFTDROP_ZCUT = 0.1


# --------------------------------------------------------------------------- kinematics
def kinematics(p):
    """Return (pt, rapidity, phi) for four-momenta ``p[..., 4]``."""
    px, py, pz, e = np.moveaxis(p, -1, 0)
    pt = np.hypot(px, py)
    y = 0.5 * np.log(np.maximum(e + pz, TINY) / np.maximum(e - pz, TINY))
    return pt, y, np.arctan2(py, px)


def mass(p):
    """Invariant mass; values below the float cancellation noise of E^2 - p^2 are zero."""
    px, py, pz, e = np.moveaxis(p, -1, 0)
    m2 = e * e - px * px - py * py - pz * pz
    return np.sqrt(np.where(m2 > 1e-10 * e * e, m2, 0.0))


def wrap(dphi):
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def delta_r2(y1, phi1, y2, phi2):
    return (y1 - y2) ** 2 + wrap(phi1 - phi2) ** 2


def _ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


# --------------------------------------------------------------------------- clustering
def cluster(p, mask, power, snapshots=(4, 3, 2)):
    """Sequential recombination (generalised kT, E-scheme, no beam distance).

    ``power`` 1 is kT and 0 is Cambridge/Aachen. Slot ``i`` keeps the merged
    pseudojet and slot ``j`` is retired. Returns the merge history and, for each
    K in ``snapshots``, the state when exactly K pseudojets remain.
    """
    p = np.where(mask[..., None], p, 0.0).astype(np.float64)
    B, N, _ = p.shape
    rows = np.arange(B)
    P, alive = p.copy(), mask.copy()
    label = np.where(mask, np.arange(N)[None, :], -1)
    count = mask.sum(1)
    pt, y, phi = kinematics(P)
    weight = pt ** (2 * power) if power else np.ones_like(pt)
    D = np.minimum(weight[:, :, None], weight[:, None, :]) * delta_r2(
        y[:, :, None], phi[:, :, None], y[:, None, :], phi[:, None, :])
    D[~(alive[:, :, None] & alive[:, None, :])] = np.inf
    D[:, np.arange(N), np.arange(N)] = np.inf

    steps = max(N - 1, 0)
    hist = dict(i=np.full((B, steps), -1), j=np.full((B, steps), -1), d=np.zeros((B, steps)),
                pa=np.zeros((B, steps, 4)), pb=np.zeros((B, steps, 4)),
                valid=np.zeros((B, steps), dtype=bool))
    snap = {K: dict(ok=np.zeros(B, dtype=bool), label=np.full((B, N), -1),
                    P=np.zeros((B, N, 4)), alive=np.zeros((B, N), dtype=bool)) for K in snapshots}

    def take(which):
        for K, s in snap.items():
            hit = which[count[which] == K]
            if len(hit):
                s["ok"][hit], s["label"][hit], s["P"][hit], s["alive"][hit] = True, label[hit], P[hit], alive[hit]

    take(rows)
    for step in range(steps):
        b = np.nonzero(count >= 2)[0]
        if not len(b):
            break
        flat = D.reshape(B, -1)
        idx = flat.argmin(1)[b]
        i, j = idx // N, idx % N
        hist["i"][b, step], hist["j"][b, step], hist["d"][b, step] = i, j, flat[b, idx]
        hist["pa"][b, step], hist["pb"][b, step], hist["valid"][b, step] = P[b, i], P[b, j], True
        P[b, i] += P[b, j]
        P[b, j] = 0.0
        alive[b, j] = False
        label[b] = np.where(label[b] == j[:, None], i[:, None], label[b])
        pt[b, i], y[b, i], phi[b, i] = kinematics(P[b, i])
        weight[b, i] = pt[b, i] ** (2 * power) if power else 1.0
        d = np.minimum(weight[b, i][:, None], weight[b]) * delta_r2(
            y[b, i][:, None], phi[b, i][:, None], y[b], phi[b])
        d[~alive[b]] = np.inf
        d[np.arange(len(b)), i] = np.inf
        D[b, i, :], D[b, :, i] = d, d
        D[b, j, :], D[b, :, j] = np.inf, np.inf
        count[b] -= 1
        take(b)
    return dict(hist=hist, snap=snap, n=mask.sum(1))


def exclusive_subjets(snapshot, K):
    """Subjet momenta sorted by pT and each particle's subjet index (-1 if none)."""
    B, N, _ = snapshot["P"].shape
    extra = max(K - N, 0)   # batches narrower than K have no valid K-subjet state
    P = np.pad(snapshot["P"], ((0, 0), (0, extra), (0, 0)))
    pt = np.where(np.pad(snapshot["alive"], ((0, 0), (0, extra))), kinematics(P)[0], -1.0)
    order = np.argsort(-pt, axis=1, kind="stable")[:, :K]
    sub = np.take_along_axis(P, order[..., None], axis=1)
    member = np.full((B, N), -1)
    for k in range(K):
        member[snapshot["label"] == order[:, k][:, None]] = k
    member[snapshot["label"] < 0] = -1
    ok = snapshot["ok"]
    sub[~ok] = 0.0
    member[~ok] = -1
    return sub, member


# --------------------------------------------------------------------------- observable blocks
def _basic(p, mask, pt, y, phi, jet, names, cols):
    jpt, jy, jphi = kinematics(jet)
    z = _ratio(pt, pt.sum(1, keepdims=True))
    dr = np.sqrt(delta_r2(y, phi, jy[:, None], jphi[:, None])) * mask
    top = -np.sort(-z, axis=1)[:, :3]
    top = np.pad(top, ((0, 0), (0, 3 - top.shape[1])))
    values = dict(n_const=mask.sum(1).astype(np.float64), jet_mass=mass(jet),
                  width=(z * dr).sum(1), lha=(z * np.sqrt(dr / R0)).sum(1),
                  thrust=(z * (dr / R0) ** 2).sum(1), ptd=np.sqrt((z * z).sum(1)),
                  z1=top[:, 0], z2=top[:, 1], z3=top[:, 2])
    for k, v in values.items():
        names.append(("basic", k, k in ("jet_mass", "width", "lha", "thrust")))
        cols.append(v)


def _subjets(kt, pt, names, cols):
    out = {}
    for K in (2, 3, 4):
        sub, member = exclusive_subjets(kt["snap"][K], K)
        spt, sy, sphi = kinematics(sub)
        total = spt.sum(1, keepdims=True)
        for k in range(K):
            names.append(("subjets", f"kt{K}_m{k+1}", True))
            cols.append(mass(sub[:, k]))
            names.append(("subjets", f"kt{K}_z{k+1}", False))
            cols.append(_ratio(spt, total)[:, k])
            names.append(("subjets", f"kt{K}_n{k+1}", False))
            cols.append((member == k).sum(1).astype(np.float64))
        for a in range(K):
            for b in range(a + 1, K):
                names.append(("subjets", f"kt{K}_dr{a+1}{b+1}", True))
                cols.append(np.sqrt(delta_r2(sy[:, a], sphi[:, a], sy[:, b], sphi[:, b])) * kt["snap"][K]["ok"])
                names.append(("subjets", f"kt{K}_m{a+1}{b+1}", True))
                cols.append(mass(sub[:, a] + sub[:, b]))
        out[K] = (sub, member)
    hist, n = kt["hist"], kt["n"]
    rows, steps = np.arange(len(n)), hist["d"].shape[1]
    for K in (2, 3, 4):  # kT scale of the merge that takes K pseudojets to K - 1
        step = n - K
        d = np.zeros(len(n))
        if steps:
            d = np.where(step >= 0, hist["d"][rows, np.clip(step, 0, steps - 1)], 0.0)
        names.append(("subjets", f"kt_d{K-1}{K}", True))
        cols.append(np.sqrt(d))
    return out


def _nsub(subjets, jet, pt, y, phi, mask, names, cols):
    norm = pt.sum(1) * R0
    axes = {1: jet[:, None, :]}
    axes.update({K: subjets[K][0] for K in (2, 3, 4)})
    tau = {}
    for K in (1, 2, 3, 4):
        apt, ay, aphi = kinematics(axes[K])
        dr = np.sqrt(delta_r2(y[:, :, None], phi[:, :, None], ay[:, None, :], aphi[:, None, :]))
        dr = np.where((apt > 0)[:, None, :], dr, np.inf).min(2)
        dr = np.where(np.isfinite(dr), dr, 0.0)
        tau[K] = _ratio((pt * dr * mask).sum(1), norm)
        names.append(("nsub", f"tau{K}", True))
        cols.append(tau[K])
    for K in (2, 3, 4):
        names.append(("nsub", f"tau{K}{K-1}", False))
        cols.append(_ratio(tau[K], tau[K - 1]))


@lru_cache(maxsize=None)
def _combinations(n, r):
    index = np.array(list(combinations(range(n), r)), dtype=int).reshape(-1, r)
    return tuple(index[:, c] for c in range(r))


def _top_by_pt(pt, y, phi, mask, k):
    order = np.argsort(-np.where(mask, pt, -1.0), axis=1, kind="stable")[:, :k]
    take = lambda a: np.take_along_axis(a, order, axis=1)
    m = take(mask)
    return take(pt) * m, take(y), take(phi), m


def _ecf(pt, y, phi, mask, names, cols, top4=24, top3=32, top4g=16):
    """Energy correlators. e2 and e3 use every particle; e4 and the generalised
    correlators use the hardest ``top4`` / ``top3`` / ``top4g`` particles, with
    energy fractions still normalised to the full jet."""
    total = pt.sum(1, keepdims=True)
    z = _ratio(pt, total) * mask
    R = np.sqrt(delta_r2(y[:, :, None], phi[:, :, None], y[:, None, :], phi[:, None, :]))
    R = R * (mask[:, :, None] & mask[:, None, :])
    pt4, y4, phi4, m4 = _top_by_pt(pt, y, phi, mask, top4)
    z4 = _ratio(pt4, total)
    R4 = np.sqrt(delta_r2(y4[:, :, None], phi4[:, :, None], y4[:, None, :], phi4[:, None, :]))
    R4 = R4 * (m4[:, :, None] & m4[:, None, :])
    e = {}
    for beta in (1, 2):
        Rb = R if beta == 1 else R * R
        e2 = 0.5 * np.einsum("bi,bij,bj->b", z, Rb, z)
        A = z[:, :, None] * Rb                         # tr(A^3) = 6 * sum_{i<j<k} zzz RRR
        e3 = np.einsum("bij,bji->b", A @ A, A) / 6.0
        Rb4 = R4 if beta == 1 else R4 * R4
        # e4 = sum_l z_l tr(A_l^3) / 24 with A_l[i, j] = z_i R_il R_ij
        Al = (z4[:, None, :] * Rb4)[:, :, :, None] * Rb4[:, None, :, :]
        e4 = np.einsum("bl,bl->b", z4, np.einsum("blij,blji->bl", Al @ Al, Al)) / 24.0
        e[beta] = (e2, e3, e4)
        for k, v in zip((2, 3, 4), (e2, e3, e4)):
            names.append(("ecf", f"e{k}_b{beta}", True))
            cols.append(v)
        names.append(("ecf", f"C2_b{beta}", True))
        cols.append(_ratio(e3, e2 ** 2))
        names.append(("ecf", f"D2_b{beta}", True))
        cols.append(_ratio(e3, e2 ** 3))
        names.append(("ecf", f"C3_b{beta}", True))
        cols.append(_ratio(e4 * e2, e3 ** 2))
        names.append(("ecf", f"D3_b{beta}", True))
        cols.append(_ratio(e4 * e2 ** 3, e3 ** 3))
    # Generalised correlators with the product of the v smallest angles (Moult, Necib, Thaler).
    e2_1 = e[1][0]
    pt3, y3, phi3, _ = _top_by_pt(pt, y, phi, mask, top3)
    z3 = _ratio(pt3, total)
    R3 = np.sqrt(delta_r2(y3[:, :, None], phi3[:, :, None], y3[:, None, :], phi3[:, None, :]))
    i, j, k = _combinations(R3.shape[1], 3)
    if len(i):
        a, b, c = R3[:, i, j], R3[:, i, k], R3[:, j, k]
        low, high = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
        zzz = z3[:, i] * z3[:, j] * z3[:, k]
        e3_1 = (zzz * low).sum(1)
        e3_2 = (zzz * _ratio(a * b * c, high)).sum(1)   # product of the two smallest angles
    else:
        e3_1 = e3_2 = np.zeros(len(pt))
    names.append(("ecf", "N2_b1", True))
    cols.append(_ratio(e3_2, e2_1 ** 2))
    ptg, yg, phig, _ = _top_by_pt(pt, y, phi, mask, top4g)
    zg = _ratio(ptg, total)
    Rg = np.sqrt(delta_r2(yg[:, :, None], phig[:, :, None], yg[:, None, :], phig[:, None, :]))
    a, b, c, d = _combinations(Rg.shape[1], 4)
    if len(a):
        ang = np.stack((Rg[:, a, b], Rg[:, a, c], Rg[:, a, d], Rg[:, b, c], Rg[:, b, d], Rg[:, c, d]), -1)
        ang = np.partition(ang, 1, axis=-1)
        e4_2 = (zg[:, a] * zg[:, b] * zg[:, c] * zg[:, d] * ang[..., 0] * ang[..., 1]).sum(1)
    else:
        e4_2 = np.zeros(len(pt))
    names.append(("ecf", "N3_b1", True))
    cols.append(_ratio(e4_2, e3_1 ** 2))


def _pull_vector(pt, y, phi, weight_mask, axis):
    apt, ay, aphi = kinematics(axis)
    dy, dphi = (y - ay[:, None]) * weight_mask, wrap(phi - aphi[:, None]) * weight_mask
    r = np.hypot(dy, dphi)
    scale = _ratio(pt * r * weight_mask, apt[:, None])
    return (scale * dy).sum(1), (scale * dphi).sum(1)


def _pull(subjets, jet, pt, y, phi, mask, names, cols):
    ty, tphi = _pull_vector(pt, y, phi, mask, jet)
    names.append(("pull", "jet_pull", True))
    cols.append(np.hypot(ty, tphi))
    sub, member = subjets[2]
    spt, sy, sphi = kinematics(sub)
    for a, b in ((0, 1), (1, 0)):
        ty, tphi = _pull_vector(pt, y, phi, member == a, sub[:, a])
        mag = np.hypot(ty, tphi)
        cy, cphi = sy[:, b] - sy[:, a], wrap(sphi[:, b] - sphi[:, a])
        cos = _ratio(ty * cy + tphi * cphi, mag * np.hypot(cy, cphi))
        names.append(("pull", f"pull{a+1}_mag", True))
        cols.append(mag)
        names.append(("pull", f"pull{a+1}{b+1}_angle", False))
        cols.append(np.arccos(np.clip(cos, -1.0, 1.0)) * (mag > 0))


def primary_lund(ca):
    """Primary Lund declustering: follow the harder branch from the full jet.

    Returns arrays ``[B, steps]`` in declustering order (widest first) and a count.
    """
    hist = ca["hist"]
    B, steps = hist["valid"].shape
    out = {k: np.zeros((B, steps)) for k in ("delta", "kt", "z", "m")}
    count = np.zeros(B, dtype=int)
    current = np.full(B, -1)
    for step in range(steps - 1, -1, -1):
        valid = hist["valid"][:, step]
        first = valid & (current < 0)
        current[first] = hist["i"][first, step]
        hit = np.nonzero(valid & (hist["i"][:, step] == current))[0]
        if not len(hit):
            continue
        pa, pb = hist["pa"][hit, step], hist["pb"][hit, step]
        pta, ya, phia = kinematics(pa)
        ptb, yb, phib = kinematics(pb)
        a_harder = pta >= ptb
        delta = np.sqrt(delta_r2(ya, phia, yb, phib))
        soft = np.minimum(pta, ptb)
        c = count[hit]
        out["delta"][hit, c], out["kt"][hit, c] = delta, soft * delta
        out["z"][hit, c], out["m"][hit, c] = _ratio(soft, pta + ptb), mass(pa + pb)
        count[hit] += 1
        current[hit] = np.where(a_harder, hist["i"][hit, step], hist["j"][hit, step])
    return out, count


def _lund_and_softdrop(ca, names, cols):
    lund, count = primary_lund(ca)
    B, steps = lund["kt"].shape
    pad = lambda a: np.pad(a, ((0, 0), (0, max(LUND_EMISSIONS - steps, 0))))[:, :LUND_EMISSIONS]
    present = np.arange(max(steps, 1))[None, :] < count[:, None] if steps else np.zeros((B, 0), bool)
    for e in range(LUND_EMISSIONS):
        for key, log in (("delta", True), ("kt", True), ("z", False)):
            names.append(("lund", f"lund{e+1}_{key}", log))
            cols.append(pad(lund[key])[:, e])
    kt = np.where(present, lund["kt"], -1.0) if steps else np.full((B, 1), -1.0)
    hardest = kt.argmax(1)
    rows = np.arange(B)
    for key, log in (("delta", True), ("kt", True), ("z", False)):
        names.append(("lund", f"lund_hardest_{key}", log))
        cols.append(lund[key][rows, hardest] * (count > 0) if steps else np.zeros(B))
    for threshold in LUND_KT_THRESHOLDS:
        names.append(("lund", f"lund_n_kt{threshold:g}", False))
        cols.append((kt > threshold).sum(1).astype(np.float64))
    names.append(("lund", "lund_n_primary", False))
    cols.append(count.astype(np.float64))
    # Soft drop with beta = 0: first primary splitting with z >= zcut.
    passed = (present & (lund["z"] >= SOFTDROP_ZCUT)) if steps else np.zeros((B, 1), bool)
    found = passed.any(1)
    first = passed.argmax(1)
    pick = lambda a: (a[rows, first] * found) if steps else np.zeros(B)
    for name, value, log in (("sd_mass", pick(lund["m"]), True), ("sd_zg", pick(lund["z"]), False),
                             ("sd_rg", pick(lund["delta"]), True),
                             ("sd_ndrop", np.where(found, first, count).astype(np.float64), False)):
        names.append(("softdrop", name, log))
        cols.append(value)


# --------------------------------------------------------------------------- entry points
def compute_batch(p, mask):
    """All observables for one padded batch. Returns (values[B, F], names).

    ``names`` is a list of ``(group, name, log_scale)``; ``log_scale`` marks
    non-negative heavy-tailed quantities the caller should take the log of.
    """
    p = np.asarray(p, dtype=np.float64)
    pt_all = np.hypot(p[..., 0], p[..., 1])
    mask = np.asarray(mask, dtype=bool) & (pt_all > 1e-9) & (p[..., 3] > 0)
    p = np.where(mask[..., None], p, 0.0)
    pt, y, phi = kinematics(p)
    pt = pt * mask
    jet = p.sum(1)
    names, cols = [], []
    _basic(p, mask, pt, y, phi, jet, names, cols)
    subjets = _subjets(cluster(p, mask, 1), pt, names, cols)
    _nsub(subjets, jet, pt, y, phi, mask, names, cols)
    _ecf(pt, y, phi, mask, names, cols)
    _pull(subjets, jet, pt, y, phi, mask, names, cols)
    _lund_and_softdrop(cluster(p, mask, 0, snapshots=()), names, cols)
    values = np.stack(cols, 1)
    return np.where(np.isfinite(values), values, 0.0), names


def compute(vectors, offsets, chunk=256):
    """Observables for ragged jets: ``vectors[total, 4]`` and ``offsets[n_jets + 1]``.

    Jets are processed in multiplicity order so each padded batch is tight.
    """
    counts = np.diff(offsets)
    order = np.argsort(counts, kind="stable")
    out, names = None, None
    for start in range(0, len(order), chunk):
        jets = order[start:start + chunk]
        width = max(int(counts[jets].max()), 1)
        p = np.zeros((len(jets), width, 4))
        mask = np.arange(width)[None, :] < counts[jets][:, None]
        p[mask] = np.concatenate([vectors[offsets[j]:offsets[j + 1]] for j in jets]) if mask.any() else 0.0
        values, names = compute_batch(p, mask)
        if out is None:
            out = np.zeros((len(counts), values.shape[1]), dtype=np.float32)
        out[jets] = values
    if out is None:
        _, names = compute_batch(np.zeros((1, 1, 4)), np.zeros((1, 1), bool))
        out = np.zeros((0, len(names)), dtype=np.float32)
    return out, names
