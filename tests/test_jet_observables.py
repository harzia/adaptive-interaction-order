"""Check the batched observables against slow per-jet reference code."""
from itertools import combinations

import numpy as np
import pytest

from aio.analysis import jet_observables as jo


def random_jets(rng, sizes, width=128):
    p = np.zeros((len(sizes), width, 4))
    mask = np.zeros((len(sizes), width), dtype=bool)
    for b, n in enumerate(sizes):
        pt = np.exp(rng.uniform(np.log(0.3), np.log(200.0), n))
        y = 0.4 + 0.5 * rng.normal(size=n).clip(-1.5, 1.5)
        phi = 2.9 + 0.5 * rng.normal(size=n).clip(-1.5, 1.5)   # straddles the +-pi seam
        m = rng.uniform(0.0, 0.3, n) * (rng.random(n) < 0.5)
        mt = np.hypot(pt, m)
        p[b, :n] = np.stack((pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y), mt * np.cosh(y)), 1)
        mask[b, :n] = True
    return p, mask


def table(p, mask):
    values, names = jo.compute_batch(p, mask)
    return {name: values[:, i] for i, (_, name, _) in enumerate(names)}


# ----------------------------------------------------------------------------- references
def kin(v):
    pt = np.hypot(v[0], v[1])
    return pt, 0.5 * np.log((v[3] + v[2]) / (v[3] - v[2])), np.arctan2(v[1], v[0])


def dr(a, b):
    _, ya, pa = kin(a)
    _, yb, pb = kin(b)
    d = (pa - pb + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(ya - yb, d)


def inv_mass(v):
    return np.sqrt(max(v[3] ** 2 - v[0] ** 2 - v[1] ** 2 - v[2] ** 2, 0.0))


def ref_cluster(parts, power):
    """Return a list of merges and, per number of remaining clusters, the clusters."""
    nodes = [dict(p=v.copy(), members=[i], children=None) for i, v in enumerate(parts)]
    live = list(range(len(nodes)))
    states = {len(live): [nodes[i] for i in live]}
    scales = {}
    while len(live) > 1:
        best = None
        for a, b in combinations(live, 2):
            wa, wb = kin(nodes[a]["p"])[0] ** (2 * power), kin(nodes[b]["p"])[0] ** (2 * power)
            d = min(wa, wb) * dr(nodes[a]["p"], nodes[b]["p"]) ** 2
            if best is None or d < best[0]:
                best = (d, a, b)
        d, a, b = best
        scales[len(live)] = d
        nodes.append(dict(p=nodes[a]["p"] + nodes[b]["p"], members=nodes[a]["members"] + nodes[b]["members"],
                          children=(nodes[a], nodes[b])))
        live = [i for i in live if i not in (a, b)] + [len(nodes) - 1]
        states[len(live)] = [nodes[i] for i in live]
    return states, scales, nodes[-1]


def ref_primary(root):
    out, node = [], root
    while node["children"]:
        a, b = node["children"]
        pa, pb = kin(a["p"])[0], kin(b["p"])[0]
        hard, soft = (a, b) if pa >= pb else (b, a)
        delta = dr(a["p"], b["p"])
        out.append(dict(delta=delta, kt=min(pa, pb) * delta, z=min(pa, pb) / (pa + pb), m=inv_mass(node["p"])))
        node = hard
    return out


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(7)
    sizes = [5, 6, 8, 9, 11, 13, 14, 14, 12, 7]
    p, mask = random_jets(rng, sizes, width=16)
    return p, mask, sizes, table(p, mask)


def test_exclusive_kt_subjets_scales_and_nsubjettiness(sample):
    p, mask, sizes, t = sample
    for b, n in enumerate(sizes):
        parts = p[b, :n]
        states, scales, root = ref_cluster(parts, 1)
        pt_i = np.array([kin(v)[0] for v in parts])
        for K in (2, 3, 4):
            subs = sorted(states[K], key=lambda s: -kin(s["p"])[0])
            for k, s in enumerate(subs):
                assert t[f"kt{K}_m{k+1}"][b] == pytest.approx(inv_mass(s["p"]), rel=1e-6, abs=1e-4)
                assert t[f"kt{K}_n{k+1}"][b] == len(s["members"])
                assert t[f"kt{K}_z{k+1}"][b] == pytest.approx(kin(s["p"])[0] / sum(kin(x["p"])[0] for x in subs))
            for a, c in combinations(range(K), 2):
                assert t[f"kt{K}_dr{a+1}{c+1}"][b] == pytest.approx(dr(subs[a]["p"], subs[c]["p"]), rel=1e-6)
                assert t[f"kt{K}_m{a+1}{c+1}"][b] == pytest.approx(inv_mass(subs[a]["p"] + subs[c]["p"]), rel=1e-6)
            assert t[f"kt_d{K-1}{K}"][b] == pytest.approx(np.sqrt(scales[K]), rel=1e-6)
            tau = sum(w * min(dr(v, s["p"]) for s in subs) for w, v in zip(pt_i, parts)) / (pt_i.sum() * jo.R0)
            assert t[f"tau{K}"][b] == pytest.approx(tau, rel=1e-6)
        tau1 = sum(w * dr(v, root["p"]) for w, v in zip(pt_i, parts)) / (pt_i.sum() * jo.R0)
        assert t["tau1"][b] == pytest.approx(tau1, rel=1e-6)
        assert t["tau21"][b] == pytest.approx(t["tau2"][b] / t["tau1"][b])
        assert t["jet_mass"][b] == pytest.approx(inv_mass(root["p"]), rel=1e-6)


def test_energy_correlators_match_explicit_sums(sample):
    p, mask, sizes, t = sample
    for b, n in enumerate(sizes):
        parts = p[b, :n]
        z = np.array([kin(v)[0] for v in parts])
        z = z / z.sum()
        R = np.array([[dr(a, c) if i != j else 0.0 for j, c in enumerate(parts)] for i, a in enumerate(parts)])
        for beta in (1, 2):
            e2 = sum(z[i] * z[j] * R[i, j] ** beta for i, j in combinations(range(n), 2))
            e3 = sum(z[i] * z[j] * z[k] * (R[i, j] * R[i, k] * R[j, k]) ** beta for i, j, k in combinations(range(n), 3))
            e4 = sum(np.prod(z[list(q)]) * np.prod([R[a, c] for a, c in combinations(q, 2)]) ** beta
                     for q in combinations(range(n), 4))
            assert t[f"e2_b{beta}"][b] == pytest.approx(e2, rel=1e-8)
            assert t[f"e3_b{beta}"][b] == pytest.approx(e3, rel=1e-8)
            assert t[f"e4_b{beta}"][b] == pytest.approx(e4, rel=1e-7)
            assert t[f"D2_b{beta}"][b] == pytest.approx(e3 / e2 ** 3, rel=1e-7)
            assert t[f"C3_b{beta}"][b] == pytest.approx(e4 * e2 / e3 ** 2, rel=1e-6)
        e2 = sum(z[i] * z[j] * R[i, j] for i, j in combinations(range(n), 2))
        one_e3 = sum(np.prod(z[list(q)]) * min(R[a, c] for a, c in combinations(q, 2)) for q in combinations(range(n), 3))
        two_e3 = sum(np.prod(z[list(q)]) * np.prod(sorted(R[a, c] for a, c in combinations(q, 2))[:2])
                     for q in combinations(range(n), 3))
        two_e4 = sum(np.prod(z[list(q)]) * np.prod(sorted(R[a, c] for a, c in combinations(q, 2))[:2])
                     for q in combinations(range(n), 4))
        assert t["N2_b1"][b] == pytest.approx(two_e3 / e2 ** 2, rel=1e-7)
        assert t["N3_b1"][b] == pytest.approx(two_e4 / one_e3 ** 2, rel=1e-7)


def test_primary_lund_plane_and_soft_drop(sample):
    p, mask, sizes, t = sample
    for b, n in enumerate(sizes):
        _, _, root = ref_cluster(p[b, :n], 0)
        primary = ref_primary(root)
        assert t["lund_n_primary"][b] == len(primary)
        for e, s in enumerate(primary[:jo.LUND_EMISSIONS]):
            for key in ("delta", "kt", "z"):
                assert t[f"lund{e+1}_{key}"][b] == pytest.approx(s[key], rel=1e-6)
        hardest = max(primary, key=lambda s: s["kt"])
        assert t["lund_hardest_kt"][b] == pytest.approx(hardest["kt"], rel=1e-6)
        for threshold in jo.LUND_KT_THRESHOLDS:
            assert t[f"lund_n_kt{threshold:g}"][b] == sum(s["kt"] > threshold for s in primary)
        passing = [(i, s) for i, s in enumerate(primary) if s["z"] >= jo.SOFTDROP_ZCUT]
        if passing:
            i, s = passing[0]
            assert t["sd_mass"][b] == pytest.approx(s["m"], rel=1e-6)
            assert t["sd_zg"][b] == pytest.approx(s["z"], rel=1e-6)
            assert t["sd_rg"][b] == pytest.approx(s["delta"], rel=1e-6)
            assert t["sd_ndrop"][b] == i
        else:
            assert t["sd_mass"][b] == 0 and t["sd_ndrop"][b] == len(primary)


def test_pull_matches_definition(sample):
    p, mask, sizes, t = sample
    for b, n in enumerate(sizes):
        parts = p[b, :n]
        states, _, root = ref_cluster(parts, 1)

        def pull(members, axis):
            apt, ay, aphi = kin(axis)
            vec = np.zeros(2)
            for i in members:
                pt, y, phi = kin(parts[i])
                r = np.array([y - ay, (phi - aphi + np.pi) % (2 * np.pi) - np.pi])
                vec += pt * np.hypot(*r) / apt * r
            return vec

        assert t["jet_pull"][b] == pytest.approx(np.hypot(*pull(range(n), root["p"])), rel=1e-6)
        subs = sorted(states[2], key=lambda s: -kin(s["p"])[0])
        for a, c in ((0, 1), (1, 0)):
            vec = pull(subs[a]["members"], subs[a]["p"])
            _, ya, pa = kin(subs[a]["p"])
            _, yc, pc = kin(subs[c]["p"])
            link = np.array([yc - ya, (pc - pa + np.pi) % (2 * np.pi) - np.pi])
            assert t[f"pull{a+1}_mag"][b] == pytest.approx(np.hypot(*vec), rel=1e-6, abs=1e-12)
            if np.hypot(*vec) > 0:
                angle = np.arccos(np.clip(vec @ link / (np.hypot(*vec) * np.hypot(*link)), -1, 1))
                assert t[f"pull{a+1}{c+1}_angle"][b] == pytest.approx(angle, abs=1e-6)


def test_invariant_under_permutation_padding_rotation_and_boost():
    rng = np.random.default_rng(3)
    p, mask = random_jets(rng, [9, 17, 30, 4], width=32)
    base, names = jo.compute_batch(p, mask)
    order = np.stack([rng.permutation(32) for _ in range(4)])
    shuffled, _ = jo.compute_batch(np.take_along_axis(p, order[..., None], 1), np.take_along_axis(mask, order, 1))
    np.testing.assert_allclose(shuffled, base, rtol=1e-6, atol=1e-9)
    padded, _ = jo.compute_batch(np.pad(p, ((0, 0), (0, 20), (0, 0))), np.pad(mask, ((0, 0), (0, 20))))
    np.testing.assert_allclose(padded, base, rtol=1e-9, atol=1e-12)
    angle, rapidity = 1.234, 0.7
    c, s, ch, sh = np.cos(angle), np.sin(angle), np.cosh(rapidity), np.sinh(rapidity)
    moved = p.copy()
    moved[..., 0], moved[..., 1] = c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]
    moved[..., 2], moved[..., 3] = ch * p[..., 2] + sh * p[..., 3], sh * p[..., 2] + ch * p[..., 3]
    transformed, _ = jo.compute_batch(moved, mask)
    np.testing.assert_allclose(transformed, base, rtol=1e-5, atol=1e-7)
    assert len({name for _, name, _ in names}) == len(names)
    assert {group for group, _, _ in names} == set(jo.GROUPS)


def test_small_and_empty_jets_are_finite_and_ragged_entry_point_matches():
    rng = np.random.default_rng(11)
    sizes = [0, 1, 2, 3, 4, 40, 128, 60, 2, 33]
    p, mask = random_jets(rng, sizes)
    values, names = jo.compute_batch(p, mask)
    assert np.isfinite(values).all() and values.shape == (len(sizes), len(names))
    assert not values[0].any()
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    ragged, ragged_names = jo.compute(p[mask], offsets, chunk=3)
    assert ragged_names == names
    np.testing.assert_allclose(ragged, values, rtol=2e-5, atol=1e-6)
    empty, _ = jo.compute(np.zeros((0, 4)), np.zeros(1, dtype=int))
    assert empty.shape == (0, len(names))
