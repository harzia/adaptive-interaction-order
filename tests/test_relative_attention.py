from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from aio.core.geometry.relative_attention import RelativeAttention, rotate_planes
from aio.models.jets.relative_particle_transformer import geometry_coordinates, pair_features
from aio.models.jets.particle_transformer import Attention, pairwise_lv_fts_pp
from integrations.weaver.RelativeParT import get_model
from scripts.verify_relative_attention import kernel_fits


def batch():
    torch.manual_seed(43)
    x = torch.randn(3, 7, 9)
    v = torch.randn(3, 4, 9)
    v[:, 3] = v[:, :3].norm(dim=1) + .2
    mask = torch.ones(3, 1, 9, dtype=torch.bool)
    mask[1, :, -3:] = False
    return x[:, -2:].clone(), x, v, mask


def model(preset, **kwargs):
    names = ["pf_points", "pf_features", "pf_vectors", "pf_mask"]
    config = SimpleNamespace(input_names=names, input_dicts={"pf_features": range(7)},
                             input_shapes={k: (1, c, 9) for k, c in zip(names, (2, 7, 4, 1))})
    net, info = get_model(config, preset=preset, num_classes=188, trim=False, **kwargs)
    assert info["input_names"] == names
    return net


@pytest.mark.parametrize("score", ["linear", "rotary"])
def test_pairwise_definition_scores_messages_and_common_shift(score, monkeypatch):
    torch.manual_seed(9)
    att = RelativeAttention(12, 2, score=score, bank="small").double().eval()
    x = torch.randn(2, 5, 12, dtype=torch.double)
    pos = torch.rand(2, 5, 3, dtype=torch.double)
    pair = torch.randn(2, 5, 5, 4, dtype=torch.double)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    pad[1, 3:] = True
    def heads(layer):
        return layer(x).unflatten(-1, (2, -1)).transpose(1, 2)
    q, k, v = map(heads, (att.q_proj, att.k_proj, att.v_proj))
    dc = att.content
    delta = pos[:, None, :, :] - pos[:, :, None, :]
    def transported(planes, omega):
        angle = torch.einsum("bijc,hmc->bhijm", delta, omega)
        return rotate_planes(planes[:, :, None], angle.cos(), angle.sin())
    logits = q[..., :dc] @ k[..., :dc].transpose(-1, -2)
    if score == "linear":
        logits = logits / dc**.5 + att.pair_proj(pair).permute(0, 3, 1, 2)
    else:
        with torch.no_grad():
            att.gain[0].fill_(.6)
        extra = (q[..., None, dc:] * transported(k[..., dc:], att.omega_score)).sum(-1)
        logits = (logits + att.gain * extra) / att.score_width**.5
    prob = logits.masked_fill(pad[:, None, None], -torch.inf).softmax(-1)
    msg = (prob.unsqueeze(-1) * transported(v[..., dc:], att.omega_value)).sum(-2)
    expected = att.out_proj(torch.cat((prob @ v[..., :dc], msg), -1).transpose(1, 2).flatten(2))
    scores = []
    sdpa = F.scaled_dot_product_attention
    def observe(q, k, v, mask, dropout):
        actual = q @ k.transpose(-1, -2) / q.shape[-1]**.5
        scores.append(actual.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else actual + mask)
        return sdpa(q, k, v, mask, dropout)
    monkeypatch.setattr(F, "scaled_dot_product_attention", observe)
    for translated in (pos, pos + torch.tensor([.31, -1.2, 2.5])):
        out, _ = att(x, x, x, coords=translated, key_padding_mask=pad, pair=pair)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(scores[-1].softmax(-1), prob)
    torch.testing.assert_close(scores[0], scores[1])


def test_zero_planes_equal_plain_attention():
    plain = Attention(12, 2).double().eval()
    linear = RelativeAttention(12, 2, rotate_values=False).double().eval()
    with torch.no_grad():
        for proj, w, b in zip((linear.q_proj, linear.k_proj, linear.v_proj),
                              plain.in_proj.weight.chunk(3), plain.in_proj.bias.chunk(3)):
            proj.weight.copy_(w)
            proj.bias.copy_(b)
        linear.out_proj.load_state_dict(plain.out_proj.state_dict())
        linear.pair_proj.weight.zero_()
        linear.pair_proj.bias.zero_()
    x = torch.randn(3, 5, 12, dtype=torch.double)
    pad = torch.zeros(3, 5, dtype=torch.bool)
    pad[0, -1] = True
    a = plain(x, x, x, key_padding_mask=pad)[0]
    b = linear(x, x, x, coords=x[..., :3], pair=x.new_zeros(3, 5, 5, 4), key_padding_mask=pad)[0]
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("preset", ["H", "E", "H_small", "E_small", "C", "D", "F", "G"])
def test_weaver_forward_backward_and_permutation_padding(preset):
    net = model(preset, num_layers=2).eval()
    points, x, v, mask = batch()
    expected = net(points, x, v, mask)
    perm = torch.randperm(9)
    inputs = [t[..., perm] for t in (points, x, v, mask)]
    for t in inputs[:3]:
        t.masked_fill_(~inputs[3], float("nan"))
    torch.testing.assert_close(net(*inputs), expected, atol=2e-5, rtol=2e-5)
    net.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(points, x, v, mask)
        loss = F.cross_entropy(out, torch.tensor([0, 1, 2]))
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters() if p.requires_grad)
    torch.optim.AdamW(net.parameters()).step()


@pytest.mark.parametrize("preset", ["H", "E"])
def test_class_origin_and_ablation_switches(preset):
    pts, x, v, mask = batch()
    net = model(preset, num_layers=2, class_at_origin=True, directions=4, momentum=False,
                freeze_frequencies=True, scale_invariant_pairs=(preset == "H"), export_embed=True).eval()
    assert "blocks.0.attn.omega_value" in dict(net.mod.named_buffers())
    assert net.mod.blocks[0].attn.omega_value.shape == (4, 16, 2)
    assert net(pts, x, v, mask).shape == (3, 236)
    frozen = net.mod.cls_blocks[0].attn.omega_value
    assert not frozen.requires_grad
    fixed = model(preset, num_layers=2, angular=False)
    assert fixed.mod.blocks[0].attn.omega_value.shape == (4, 3, 1)


def test_exact_pair_features_and_unstandardized_coordinates():
    pts, _, v, _ = batch()
    pts[:, 1].fill_(-torch.pi)
    coords = geometry_coordinates(pts, v)
    torch.testing.assert_close(coords[..., 2].exp(), v[:, :2].norm(dim=1) / 500)
    torch.testing.assert_close(coords[..., 1], torch.full_like(coords[..., 1], torch.pi))
    ref = pairwise_lv_fts_pp(v.unsqueeze(-1), v.unsqueeze(-2)).movedim(1, -1)
    torch.testing.assert_close(pair_features(v, v), ref)
    torch.testing.assert_close(pair_features(v, v, True), ref[..., 1:3])


@pytest.mark.parametrize("preset,counts", [("H", [280136, 361424]), ("H_small", [265952, 333056]),
                                          ("E", [327864, 456880]), ("E_small", [285024, 371200])])
def test_proposal_sizes(preset, counts):
    for d, count in zip((2, 4), counts):
        assert sum(p.numel() for p in model(preset, num_layers=d).parameters()) == count


def test_kernel_fits():
    result = kernel_fits(18000)
    for key, reference in {"angular_3": .949, "angular_equal": .876, "angular_4": .964,
                           "harmonics_l2": .681, "momentum_abs": .994, "momentum_lnz": .996}.items():
        assert result[key] == pytest.approx(reference, abs=.012)


def test_geometry_uses_same_trim_indices_and_rotary_skips_pairs(monkeypatch):
    from aio.models.jets import relative_particle_transformer as rel
    pts, x, v, mask = batch()
    net = model("E", num_layers=2).eval()
    idx = torch.tensor([3, 0, 4, 2])
    def unexpected_pair(*args):
        raise AssertionError("Rotary arm built a per-pair feature tensor")
    monkeypatch.setattr(rel, "pair_features", unexpected_pair)
    expected = net(pts[..., idx], x[..., idx], v[..., idx], mask[..., idx])
    class FixedTrim(torch.nn.Module):
        def forward(self, x, packed, mask):
            return x[..., idx], packed[..., idx], mask[..., idx], None
    net.mod.trimmer = FixedTrim()
    torch.testing.assert_close(net(pts, x, v, mask), expected)


def test_launcher_suite_order_and_checkpoint_args(tmp_path, monkeypatch, capsys):
    import sys
    from scripts.run_relative_experiment import main
    monkeypatch.delenv("COMMENT", raising=False)
    monkeypatch.setenv("OUTPUT_PATH", str(tmp_path / "unused"))
    monkeypatch.setattr(sys, "argv", ["run", "suite", "--dry-run", "--seed", "124"])
    main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert all(f"relative_{arm}_seed124" in line for arm, line in zip("CHBE", lines))
    monkeypatch.setenv("COMMENT", "my_hybrid")
    monkeypatch.setattr(sys, "argv", ["run", "H", "--dry-run", "--num-epochs=2", "--train-percentage=25",
                                      "--weights", "my checkpoint.pt", "-o", "num_layers", "2"])
    main()
    import shlex
    cmd = shlex.split(capsys.readouterr().out)
    assert cmd[:2] == ["EPOCHS=2", "COMMENT=my_hybrid"]
    assert cmd[cmd.index("--train-percentage") + 1] == "25"
    assert cmd[-5:] == ["--weights", "my checkpoint.pt", "-o", "num_layers", "2"]
    assert not (tmp_path / "unused").exists()
