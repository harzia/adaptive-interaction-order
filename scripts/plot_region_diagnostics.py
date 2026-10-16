#!/usr/bin/env python3
"""Plot region quality, tensor utility versus structure, and matched-event CE.

These are checkpoint diagnostics, not ROC estimates or independent-region error
bars. Optional kinematic cuts use original observers saved by the input exporter.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, NullLocator, FuncFormatter
from integrations.weaver.eval.jetclass2_eval import SIGNAL_GROUPS, QCD_INDICES


GROUPS = {k.removeprefix("label_"): v for k, v in SIGNAL_GROUPS.items()}
GROUPS["QCD"] = QCD_INDICES.tolist()


def save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / (name + ".png"), dpi=160)
    plt.close(fig)


def binned_rows(x, utility, active, selected, group, feature, bins=8):
    """Descriptive means only. Regions within one jet are not independent trials."""
    jets = np.broadcast_to(np.arange(len(selected))[:, None], x.shape)
    valid = active & selected[:, None] & np.isfinite(x) & np.isfinite(utility)
    xx, yy, jj = x[valid], utility[valid], jets[valid]
    if not len(xx):
        return []
    edges = np.unique(np.quantile(xx, np.linspace(0, 1, bins + 1)))
    if len(edges) == 1:
        edges = np.array([edges[0], np.nextafter(edges[0], np.inf)])
    rows = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        keep = (xx >= lo) & ((xx <= hi) if index == len(edges) - 2 else (xx < hi))
        if keep.any():
            rows.append(dict(group=group, feature=feature, lower=float(lo), upper=float(hi),
                             mean_structure=float(xx[keep].mean()), mean_delta_ce=float(yy[keep].mean()),
                             num_regions=int(keep.sum()), num_jets=int(len(np.unique(jj[keep])))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS),
                        default=["X_YY_gggg", "X_YY_QQgg", "X_YY_QQQQ", "X_YY_QQg", "X_YY_ggg"])
    parser.add_argument("--pt-range", nargs=2, type=float)
    parser.add_argument("--mass-range", nargs=2, type=float)
    parser.add_argument("--multiplicity-range", nargs=2, type=int,
                        help="Input valid-constituent count, after truncation")
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("--examples must be nonnegative")
    for bounds in (args.pt_range, args.mass_range, args.multiplicity_range):
        if bounds is not None and bounds[0] >= bounds[1]:
            parser.error("Ranges must have lower < upper")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded, all_rows, summary_rows = [], [], []
    for index, path in enumerate(args.reports):
        report = json.loads(path.read_text())
        detail = Path(report["details"])
        if not detail.is_file():
            detail = path.parent / detail.name
        tensors = torch.load(detail, map_location="cpu", weights_only=True)
        d = {k: v.detach().numpy() for k, v in tensors.items()}
        if "labels" not in d:
            parser.error(f"Labels required for grouped plots: {path}")
        selected = np.ones(len(d["labels"]), dtype=bool)
        for bounds, key in ((args.pt_range, "observer__jet_pt"),
                            (args.mass_range, "observer__jet_sdmass"),
                            (args.multiplicity_range, "constituent_count")):
            if bounds is not None:
                if key not in d:
                    parser.error(f"{path}: missing {key}; re-export inputs with observers")
                selected &= (d[key] >= bounds[0]) & (d[key] < bounds[1])
        if not selected.any():
            parser.error(f"No events pass filters: {path}")
        name = f"{index}_{report['preset']}"
        loaded.append((report, d, selected, name))
        if "assignments" in d:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
            for ax, key, label in zip(axes,
                    ["hard_occupied_regions", "mean_region_overlap", "normalized_assignment_entropy"],
                    ["Occupied regions (argmax)", "Assignment-column overlap", "Assignment entropy / log K"]):
                ax.hist(d[key][selected], bins=20)
                ax.set_xlabel(label)
                ax.set_ylabel("Jets")
            fig.suptitle(report["preset"])
            save(fig, args.output_dir, name + "_assignments")
        for group in args.groups:
            chosen = selected & np.isin(d["labels"], GROUPS[group])
            if not chosen.any():
                continue
            row = dict(model=name, group=group, num_jets=int(chosen.sum()))
            for key, value in d.items():
                if value.ndim == 1 and (key == "base_ce" or key.endswith("_delta_ce")):
                    row[key] = float(value[chosen].mean())
            summary_rows.append(row)
            if "region_off_delta_ce" not in d:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            for ax, feature in zip(axes, ["angular_covariance_trace", "angular_anisotropy"]):
                active = d["region_mask"] & d["angular_shape_valid"]
                rows = binned_rows(d[feature], d["region_off_delta_ce"], active, chosen, group, feature)
                all_rows.extend(dict(model=name, **r) for r in rows)
                if rows:
                    ax.plot([r["mean_structure"] for r in rows], [r["mean_delta_ce"] for r in rows], "o-")
                if feature.endswith("trace"):
                    ax.set_xscale("log")
                    ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.axhline(0, color="grey", lw=.7)
                ax.set_xlabel("Angular spread tr(C)" if feature.endswith("trace") else "Angular anisotropy")
                ax.set_ylabel("CE(region tensors off) − CE(original)")
            fig.suptitle(f"{report['preset']} · {group} · {chosen.sum()} jets; descriptive bins")
            save(fig, args.output_dir, name + "_" + group + "_utility")
        if "lund_split_valid" in d:
            keep = d["lund_split_valid"] & selected[:, None] & (d["lund_delta"] > 0) & (d["lund_kt_fraction"] > 0)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            if keep.any():
                colors = d.get("region_off_delta_ce", np.zeros_like(d["lund_delta"]))[keep]
                limit = max(float(np.abs(colors).max()), 1e-8)
                points = ax.scatter(np.log(1 / d["lund_delta"][keep]), np.log(d["lund_kt_fraction"][keep]),
                                    c=colors, s=8, cmap="coolwarm", norm=TwoSlopeNorm(0, -limit, limit))
                fig.colorbar(points, ax=ax, label="Region-off ΔCE (0 if not evaluated)")
            ax.set(xlabel="log(1 / Δ)", ylabel="log(kt / sum constituent pt)",
                   title="Internal splitting of cut nodes; leaves excluded")
            save(fig, args.output_dir, name + "_lund_cut")
        if "assignments" in d and args.examples:
            # Examples are explicitly chosen by global tensor-removal effect;
            # include both ends, not only favorable cases.
            ids = np.flatnonzero(selected & np.isin(d["labels"], sum((GROUPS[g] for g in args.groups), [])))
            if "all_tensors_off_delta_ce" in d:
                ids = ids[np.argsort(d["all_tensors_off_delta_ce"][ids])]
                # Alternating largest/smallest effects, with no repeated examples.
                picked = []
                for rank in range(min(args.examples, len(ids))):
                    picked.append(ids[-1 - rank // 2] if rank % 2 == 0 else ids[rank // 2])
                ids = picked
            else:
                ids = ids[:args.examples]
            for jet in ids:
                keep = d["mask"][jet].astype(bool)
                a = d["assignments"][jet, keep]
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
                axes[0].scatter(d["rapidity_offset"][jet, keep], d["phi_offset"][jet, keep],
                                s=15 + 300 * np.sqrt(d["constituent_pt_fraction"][jet, keep]),
                                c=a.argmax(-1), cmap="tab20", vmin=0, vmax=max(a.shape[1]-1, 1))
                axes[0].set(xlabel="y − jet y", ylabel="wrapped φ − jet φ", title="Hard labels for visualization only")
                order = np.argsort(d["rapidity_offset"][jet, keep])
                image = axes[1].imshow(a[order].T, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
                fig.colorbar(image, ax=axes[1], label="Assignment weight")
                axes[1].set(xlabel="Constituents sorted by y", ylabel="Region", title="Actual pooling weights")
                fig.suptitle(f"{report['preset']} · sample {jet} · label {d['labels'][jet]}")
                save(fig, args.output_dir, name + f"_jet{jet}")
    # Paired loss comparison requires identical input files and event order.
    # The first report is the reference; positive means it has lower CE.
    if len(loaded) > 1:
        reference, base, base_selected, base_name = loaded[0]
        for report, d, selected, name in loaded[1:]:
            same = (reference.get("input_sha256") and reference.get("input_sha256") == report.get("input_sha256")
                    and np.array_equal(base["labels"], d["labels"]))
            if not same or "base_ce" not in base or "base_ce" not in d:
                print(f"Skipping paired CE for {name}: inputs/order not verified")
                continue
            for group in args.groups:
                keep = base_selected & selected & np.isin(d["labels"], GROUPS[group])
                if keep.any():
                    summary_rows.append(dict(model=f"{name} minus {base_name}", group=group,
                                             num_jets=int(keep.sum()),
                                             paired_ce_difference=float((d["base_ce"][keep] - base["base_ce"][keep]).mean())))
    for filename, rows in (("utility_bins.csv", all_rows), ("group_summary.csv", summary_rows)):
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (args.output_dir / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    (args.output_dir / "selection.json").write_text(json.dumps(dict(
        reports=[str(p) for p in args.reports], groups=args.groups, pt_range=args.pt_range,
        mass_range=args.mass_range, multiplicity_range=args.multiplicity_range,
        note="Exploratory, label-dependent checkpoint interventions. Bins are descriptive, not independent-region confidence intervals. No ROC estimates from this inspection sample."), indent=2) + "\n")
    print(f"Saved plots and descriptive tables to {args.output_dir}")


if __name__ == "__main__":
    main()
