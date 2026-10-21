"""Frozen-expert feasibility tests for adaptive representation order.

The router sees input geometry only. Soft mixtures evaluate every expert; hard
dispatch evaluates one expert per jet. Neither is a regional order router.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

FEATURE_NAMES = (
    "log1p_constituents", "mass_over_pt", "pt_dispersion", "largest_pt_fraction",
    "top3_pt_fraction", "pt_entropy", "girth", "radial_second_moment",
    "radial_fourth_moment", "angular_covariance_trace", "angular_anisotropy",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def jet_gate_features(vectors, mask):
    """Cheap dimensionless shapes from px,py,pz,E [B,4,N], mask [B,1,N].

    These are detector/beam-referenced rapidity-azimuth observables, NOT general
    Lorentz invariants. No labels, checkpoint embeddings, or expert predictions.
    Input momenta may already have been rescaled by the Weaver configuration.
    """
    if vectors.ndim != 3 or vectors.shape[1] != 4 or mask.shape != (len(vectors), 1, vectors.shape[2]):
        raise ValueError("Expected vectors [B,4,N] and mask [B,1,N]")
    valid = mask[:, 0].bool()
    p = vectors.double().transpose(1, 2).masked_fill(~valid[..., None], 0)
    if not torch.isfinite(p).all() or not valid.any(1).all():
        raise ValueError("Nonempty jets with finite valid momenta required")
    pt = p[..., :2].square().sum(-1).sqrt()
    total_pt = pt.sum(1)
    if (total_pt <= 0).any():
        raise ValueError("Each jet must have positive scalar transverse momentum")
    p = p / total_pt[:, None, None]
    weights = pt / total_pt[:, None]
    total = p.sum(1)
    jet_pt = total[:, :2].square().sum(-1).sqrt().clamp_min(1e-12)
    mass = (total[:, 3].square() - total[:, :3].square().sum(-1)).clamp_min(0).sqrt()
    rapidity = .5 * ((p[..., 3] + p[..., 2]).clamp_min(1e-12).log()
                      - (p[..., 3] - p[..., 2]).clamp_min(1e-12).log())
    jet_y = .5 * ((total[:, 3] + total[:, 2]).clamp_min(1e-12).log()
                   - (total[:, 3] - total[:, 2]).clamp_min(1e-12).log())
    dphi = torch.atan2(p[..., 1], p[..., 0]) - torch.atan2(total[:, 1], total[:, 0])[:, None]
    dphi = (dphi + torch.pi) % (2 * torch.pi) - torch.pi
    coords = torch.stack((rapidity - jet_y[:, None], dphi), -1).masked_fill(~valid[..., None], 0)
    radius2 = coords.square().sum(-1)
    center = (weights[..., None] * coords).sum(1)
    delta = coords - center[:, None]
    cov = torch.einsum("bn,bni,bnj->bij", weights, delta, delta)
    trace = cov[:, 0, 0] + cov[:, 1, 1]
    anisotropy = ((cov[:, 0, 0] - cov[:, 1, 1]).square() + 4 * cov[:, 0, 1].square()).sqrt() / trace.clamp_min(1e-12)
    features = torch.stack((
        valid.sum(1).double().log1p(), mass / jet_pt, weights.square().sum(1).sqrt(),
        weights.max(1).values, weights.topk(min(3, weights.shape[1]), dim=1).values.sum(1),
        -(weights * weights.clamp_min(1e-12).log()).sum(1),
        (weights * radius2.sqrt()).sum(1), (weights * radius2).sum(1),
        (weights * radius2.square()).sum(1), trace, anisotropy,
    ), -1).float()
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite gate features")
    return features


def input_identities(data):
    """Source/selected-row IDs plus hashes of unpadded, unordered model inputs.

    Source IDs need the SAME YAML selection across caches. Content hashes also
    catch copied/renamed input files and constituent permutations. No labels are
    included in either identity. They cannot identify all generator-level kinship.
    """
    features, vectors, mask = (data[k].cpu() for k in ("features", "vectors", "mask"))
    n = len(features)
    contents = []
    for row in range(n):
        valid = mask[row, 0].bool()
        values = torch.cat((features[row, :, valid], vectors[row, :, valid]), 0).T.float().numpy()
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Cannot identify empty/nonfinite input")
        values = values[np.lexsort(values.T[::-1])].astype("<f4", copy=True)
        values[values == 0] = 0  # canonicalize negative zero
        digest = hashlib.sha256(str(values.shape).encode() + values.tobytes()).hexdigest()
        contents.append(digest)
    source_keys = ("source_file_index", "source_selected_index")
    if all(k in data for k in source_keys):
        files = data.get("metadata", {}).get("files", [])
        source_ids = []
        for file_index, selected_index in zip(*(data[k].tolist() for k in source_keys)):
            if not 0 <= file_index < len(files) or selected_index < 0:
                raise ValueError("Invalid source identity metadata")
            identity = json.dumps([str(Path(files[file_index]).resolve()), selected_index])
            source_ids.append(hashlib.sha256(identity.encode()).hexdigest())
        if len(source_ids) != n:
            raise ValueError("Source identity count mismatch")
    elif any(k in data for k in source_keys):
        raise ValueError("Incomplete source identity metadata")
    else:
        raise ValueError("Input export needs source_file_index/source_selected_index and metadata.files")
    return np.asarray(source_ids, dtype="S64"), np.asarray(contents, dtype="S64")


def resolve_experts(manifest_path):
    from aio.models.jets.representation_experiments import experiment_config, experiment_family
    path = Path(manifest_path)
    document = json.loads(path.read_text())
    classes = document.get("class_names")
    if not isinstance(classes, list) or len(classes) < 2 or any(not isinstance(c, str) for c in classes) or len(set(classes)) != len(classes):
        raise ValueError("Manifest needs explicit unique class_names in checkpoint output order")
    experts = []
    for item in document["experts"]:
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        config = {**experiment_config(item["preset"]), **item.get("model_overrides", {})}
        if any(k in config for k in ("input_dim", "num_classes", "export_embed", "geometry_checkpoint", "freeze_geometry")):
            raise ValueError("Overrides must describe the backbone; input/classes come from the cache")
        config["for_inference"] = False
        experts.append(dict(name=item["name"], preset=item["preset"],
                            family=experiment_family(item["preset"]), model=config,
                            checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_sha256(checkpoint)))
    names = [e["name"] for e in experts]
    if len(names) < 2 or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("At least two uniquely named experts required")
    return dict(class_names=classes, experts=experts)


def load_expert(spec, input_dim, num_classes, device="cpu"):
    if file_sha256(spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise ValueError("Expert checkpoint has changed since caching/fitting")
    if spec["family"] == "region":
        from aio.models.jets.region_transformer import RegionTransformer
        cls = RegionTransformer
    elif spec["family"] == "part":
        from aio.models.jets.particle_transformer import ParticleTransformer
        cls = ParticleTransformer
    else:
        raise ValueError("Unknown expert family")
    model = cls(input_dim=input_dim, num_classes=num_classes, **spec["model"])
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    cleaned = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.", "mod.")):
            key = key.split(".", 1)[1]
        if key in cleaned:
            raise ValueError("Duplicate checkpoint key after unwrapping")
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return model.to(device).eval().requires_grad_(False)


class ExpertCache:
    """Read-only memory-mapped probabilities: never load the full cache on GPU."""
    fields = ("probabilities", "features", "labels", "event_ids", "content_ids")

    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "manifest.json").read_text())
        if self.metadata.get("format") != "aio-frozen-experts-v1":
            raise ValueError("Unknown/incomplete cache format")
        for field in self.fields:
            setattr(self, field, np.load(self.path / (field + ".npy"), mmap_mode="r", allow_pickle=False))
        n, e, c = self.probabilities.shape
        if n == 0 or e != len(self.metadata["experts"]) or c != len(self.metadata["class_names"]):
            raise ValueError("Cache probability dimensions disagree with manifest")
        if self.features.shape != (n, len(self.metadata["feature_names"])) or any(getattr(self, k).shape != (n,) for k in ("labels", "event_ids", "content_ids")):
            raise ValueError("Mismatched cache rows/features")
        if self.metadata["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Unknown gate feature schema")
        if self.labels.dtype.kind not in "iu" or self.event_ids.dtype != np.dtype("S64") or self.content_ids.dtype != np.dtype("S64"):
            raise ValueError("Invalid label/identity dtype")
        for field in ("event_ids", "content_ids"):
            ids = getattr(self, field)
            if np.unique(ids).size != n or (ids == b"").any():
                raise ValueError(f"Duplicate or empty {field} in cache {path}")
        for start in range(0, n, 4096):
            p, x, y = self.probabilities[start:start+4096], self.features[start:start+4096], self.labels[start:start+4096]
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=2e-5, rtol=0):
                raise ValueError("Cache must contain normalized finite probabilities, not logits")
            if not np.isfinite(x).all() or (y < 0).any() or (y >= c).any():
                raise ValueError("Invalid features or labels")

    def __len__(self):
        return len(self.labels)

    def batches(self, batch_size, *, order=None, device="cpu"):
        indices = np.arange(len(self)) if order is None else order
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start+batch_size]
            yield (torch.from_numpy(np.array(getattr(self, key)[ix], copy=True)).to(device)
                   for key in ("features", "probabilities", "labels"))


def validate_splits(*caches):
    signature_keys = ("class_names", "feature_names", "preprocessing_sha256", "input_dim")
    def signature(cache):
        # A checkpoint copied to another path is still the same expert.
        experts = [{k: e[k] for k in ("name", "family", "model", "checkpoint_sha256")} for e in cache.metadata["experts"]]
        return {**{k: cache.metadata[k] for k in signature_keys}, "experts": experts}
    if any(signature(c) != signature(caches[0]) for c in caches[1:]):
        raise ValueError("Caches differ in experts/order, classes, preprocessing, or features")
    for i, left in enumerate(caches):
        for right in caches[i+1:]:
            for field in ("event_ids", "content_ids"):
                if np.intersect1d(getattr(left, field), getattr(right, field)).size:
                    raise ValueError(f"Overlapping {field} between {left.path} and {right.path}")


def true_class_losses(probabilities, labels):
    p = probabilities.gather(-1, labels.long()[:, None, None].expand(-1, probabilities.shape[1], 1)).squeeze(-1)
    return -p.clamp_min(1e-12).log()


class InputGate(nn.Module):
    def __init__(self, mean, scale, num_experts, hidden=16):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).float().clone())
        self.register_buffer("scale", torch.as_tensor(scale).float().clamp_min(1e-6).clone())
        self.net = (nn.Sequential(nn.Linear(len(mean), hidden), nn.SiLU(), nn.Linear(hidden, num_experts))
                    if hidden else nn.Sequential(nn.Linear(len(mean), num_experts)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net((features.float() - self.mean) / self.scale).softmax(-1)


class ConstantGate(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_experts))

    def forward(self, features):
        return self.logits.softmax(-1).expand(len(features), -1)


def routed_probabilities(probabilities, weights, hard=False):
    if hard:
        return probabilities[torch.arange(len(weights), device=weights.device), weights.argmax(-1)]
    return (probabilities * weights[..., None]).sum(1)


@torch.no_grad()
def hard_dispatch(experts, gate, features, vectors, mask):
    """Only run chosen experts, in sub-batches; restore original event order.

    All expert weights remain resident. Dispatch overhead and uneven sub-batches
    mean latency MUST be measured, not inferred from the chosen-expert count.
    """
    if gate.training or any(model.training for model in experts):
        raise ValueError("Frozen routing requires eval mode")
    choices = gate(jet_gate_features(vectors, mask)).argmax(-1)
    if choices.max() >= len(experts):
        raise ValueError("Gate/expert dimension mismatch")
    output = None
    for index, expert in enumerate(experts):
        selected = (choices == index).nonzero(as_tuple=True)[0]
        if not len(selected):
            continue
        probabilities = expert(features[selected], vectors[selected], mask[selected]).float().softmax(-1)
        if output is None:
            output = probabilities.new_empty((len(features), probabilities.shape[-1]))
        output[selected] = probabilities
    return output, choices
