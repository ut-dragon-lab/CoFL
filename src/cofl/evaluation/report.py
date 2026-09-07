"""Shareable offline evaluation figures and a self-contained HTML index."""

from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path
from urllib.parse import quote

import numpy as np

from cofl.data.profiles import field_query_grid
from cofl.fields import query_valid_mask


def _angular_error(prediction, target, mask, epsilon):
    """Match the field metric: undefined zero targets, 180° for zero predictions."""
    result = np.full(mask.shape, np.nan, dtype=np.float64)
    pnorm, tnorm = np.linalg.norm(prediction, axis=-1), np.linalg.norm(target, axis=-1)
    directional = mask & (tnorm > epsilon)
    result[directional] = 180.0
    nonzero = directional & (pnorm > epsilon)
    p = prediction[nonzero] / pnorm[nonzero, None]
    t = target[nonzero] / tnorm[nonzero, None]
    result[nonzero] = np.degrees(np.arccos(np.clip(np.sum(p * t, axis=-1), -1, 1)))
    return result


def _coordinates(ax, profile, scale):
    ax.set_aspect("equal")
    if profile == "image_field_v1":
        ax.set(
            xlim=(0, 1),
            ylim=(1, 0),
            xlabel="Right (image fraction)",
            ylabel="Down (image fraction)",
        )
    else:
        # Positive body-left is left on the page; forward is up.
        ax.set(xlim=(scale, -scale), ylim=(0, scale), xlabel="Left (m)", ylabel="Forward (m)")
    ax.tick_params(labelsize=8, colors="#526175")
    for spine in ax.spines.values():
        spine.set_color("#ccd5e0")


def _field_panel(fig, ax, grid, values, mask, *, profile, scale, vmax, title):
    ground = profile == "ground_sector_v1"
    x, y = (grid[..., 1], grid[..., 0]) if ground else (grid[..., 0], grid[..., 1])
    magnitude = np.linalg.norm(values, axis=-1)
    mesh = ax.pcolormesh(
        x,
        y,
        np.ma.masked_where(~mask, magnitude),
        shading="nearest",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
    )
    stride = max(1, int(np.ceil(max(mask.shape) / 18)))
    selected = np.zeros_like(mask)
    selected[::stride, ::stride] = True
    selected &= mask & (magnitude > 0)
    unit = values[selected] / magnitude[selected, None]
    u, v = (unit[:, 1], unit[:, 0]) if ground else (unit[:, 0], unit[:, 1])
    ax.quiver(
        x[selected],
        y[selected],
        u,
        v,
        color="white",
        edgecolors="#203347",
        linewidths=0.35,
        angles="xy",
        scale_units="xy",
        scale=22 / scale,
        width=0.004,
        pivot="mid",
    )
    _coordinates(ax, profile, scale)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("m / policy time" if ground else "image fraction / policy time", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)


def _path(values, name):
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[-1] != 2
        or not len(result)
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite [N,2] trajectory")
    return result


def _observation_panel(ax, sample, record, profile):
    image = np.asarray(sample["image"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("sample image must have shape [H,W,3]")
    if profile == "ground_sector_v1":
        ax.imshow(image)
        ax.set_axis_off()
        action = record.get("metrics", {}).get("action")
        ax.set_title("Observation · egocentric RGB", loc="left", fontweight="bold", fontsize=12)
        if action:
            names = ("STOP", "FORWARD", "LEFT", "RIGHT")
            target = names[action["target"]] if action.get("target") is not None else "unlabeled"
            predicted = names[action["prediction"]]
            ax.text(
                0,
                -0.05,
                f"Action target: {target}   ·   Prediction: {predicted}",
                transform=ax.transAxes,
                fontsize=9,
                wrap=True,
            )
        return
    ax.imshow(image, extent=(0, 1, 1, 0))
    extras = sample.get("annotation_extras", {})
    reference = extras.get("trajectory_state")
    if reference is not None:
        reference = _path(reference, "trajectory_state")
        ax.plot(*reference.T, color="#15b8a6", linewidth=2.5, label="GT path")
    prediction = record.get("diagnostics", {}).get("prediction_trajectory")
    if prediction is not None:
        prediction = _path(prediction, "prediction_trajectory")
        ax.plot(
            *prediction.T, color="#ff5e58", linewidth=2.3, linestyle="--", label="Predicted path"
        )
    start = reference[0] if reference is not None else extras.get("start")
    for point, label, color, marker in (
        (start, "Start", "#ffffff", "o"),
        (extras.get("goal"), "Goal", "#ffd166", "*"),
    ):
        if point is not None:
            value = np.asarray(point)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{label} must be a finite image point [2]")
            ax.scatter(
                *value,
                color=color,
                edgecolors="#17263a",
                marker=marker,
                s=100 if marker == "*" else 55,
                linewidth=1,
                zorder=5,
                label=label,
            )
    _coordinates(ax, profile, 1)
    ax.set_title(
        "Observation · paths and independent goal", loc="left", fontweight="bold", fontsize=12
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8, framealpha=0.9)


def render_sample(path: Path, sample: dict, prediction: np.ndarray | None, record: dict) -> Path:
    """Write a PNG in native image coordinates or Cartesian forward/left metres.

    ``sample['field']`` is CHW; prediction is HWC on the same native grid.
    Masked cells may contain NaN. Arrows display direction, and both field
    panels share one magnitude color scale. Scores come from the record; the
    figure makes no independent navigation-success or collision judgment.
    """
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    geometry = sample["geometry"]
    profile = geometry["profile"]
    grid = field_query_grid(geometry).astype(np.float64)
    target = np.moveaxis(np.asarray(sample["field"], dtype=np.float64), 0, -1)
    mask = np.asarray(sample["mask"])
    if target.shape != grid.shape or mask.shape != grid.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("Native field [2,H,W] and boolean mask [H,W] must match the geometry")
    mask = mask & query_valid_mask(profile, geometry)
    if not np.isfinite(target[mask]).all():
        raise ValueError("Target field must be finite on valid cells")
    predicted = None if prediction is None else np.asarray(prediction, dtype=np.float64)
    if predicted is not None and (
        predicted.shape != target.shape or not np.isfinite(predicted[mask]).all()
    ):
        raise ValueError("Prediction must have matching [H,W,2] shape and finite valid cells")
    scale = float(geometry["normalization_scale_m"]) if profile == "ground_sector_v1" else 1.0
    magnitudes = [np.linalg.norm(target[mask], axis=-1) * scale]
    if predicted is not None:
        magnitudes.append(np.linalg.norm(predicted[mask], axis=-1) * scale)
    nonempty = [values.max() for values in magnitudes if values.size]
    vmax = max(nonempty, default=0.0) or 1.0
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Sample figures must use a .png path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rc_context(
        {
            "font.family": "DejaVu Sans",
            "text.usetex": False,
            "text.parse_math": False,
            "axes.facecolor": "#e8edf3",
        }
    ):
        fig = Figure(figsize=(13, 10), facecolor="#f8fafc")
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(left=0.08, right=0.94, top=0.83, bottom=0.15, wspace=0.33, hspace=0.36)
        identity = str(sample.get("sample_id", record.get("sample_id", "Sample")))
        scene = record.get("diagnostics", {}).get("scene_id", sample.get("scene_id", ""))
        title = "CoFL" if profile == "image_field_v1" else "CoFL-S"
        fig.text(
            0.06, 0.96, f"{title} · offline evaluation", fontsize=19, weight="bold", color="#17263a"
        )
        fig.text(
            0.06,
            0.926,
            textwrap.shorten(f"{identity}  ·  {scene}", width=140),
            fontsize=9,
            color="#526175",
        )
        instruction = textwrap.fill(
            textwrap.shorten(str(sample.get("instruction", "")), width=210), width=105
        )
        fig.text(0.06, 0.9, instruction, fontsize=11, color="#263b53", va="top")
        _observation_panel(axes[0, 0], sample, record, profile)
        _field_panel(
            fig,
            axes[0, 1],
            grid * scale,
            target * scale,
            mask,
            profile=profile,
            scale=scale,
            vmax=vmax,
            title="Ground-truth field",
        )
        if predicted is None:
            for ax, name in ((axes[1, 0], "Predicted field"), (axes[1, 1], "Angular error")):
                ax.set_title(name, loc="left", fontweight="bold", fontsize=12)
                ax.text(
                    0.5,
                    0.5,
                    "Prediction unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="#526175",
                )
                _coordinates(ax, profile, scale)
        else:
            _field_panel(
                fig,
                axes[1, 0],
                grid * scale,
                predicted * scale,
                mask,
                profile=profile,
                scale=scale,
                vmax=vmax,
                title="Predicted field",
            )
            epsilon = (
                record.get("metrics", {}).get("field", {}).get("magnitude_epsilon_native", 1e-8)
            )
            if not np.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("Field magnitude epsilon must be finite and positive")
            error = _angular_error(predicted, target, mask, epsilon)
            x, y = (
                (grid[..., 1] * scale, grid[..., 0] * scale)
                if profile == "ground_sector_v1"
                else (grid[..., 0], grid[..., 1])
            )
            mesh = axes[1, 1].pcolormesh(
                x, y, np.ma.masked_invalid(error), shading="nearest", cmap="magma", vmin=0, vmax=180
            )
            _coordinates(axes[1, 1], profile, scale)
            axes[1, 1].set_title(
                "Angular error",
                loc="left",
                fontweight="bold",
                fontsize=12,
            )
            colorbar = fig.colorbar(mesh, ax=axes[1, 1], fraction=0.046, pad=0.04)
            colorbar.set_label("degrees", fontsize=8)
            colorbar.ax.tick_params(labelsize=8)
        metrics = record.get("metrics", {})
        unit = (
            "m / policy time" if profile == "ground_sector_v1" else "image fraction / policy time"
        )
        for position, group, labels in (
            (
                0.076,
                "field",
                (
                    ("vector_l2_mean", f"Vector L2 ({unit})"),
                    ("angular_error_deg_mean", "Angular error (deg)"),
                ),
            ),
            (
                0.056,
                "image_navigation",
                (
                    ("fge", "FGE (image fraction)"),
                    ("cr", "CR (0/1)"),
                    ("plr", "PLR"),
                    ("curv", "Curvature (rad/turn)"),
                ),
            ),
        ):
            selected = [
                f"{label}: {_display(metrics[group][key])}"
                for key, label in labels
                if key in (metrics.get(group) or {})
            ]
            fig.text(0.06, position, "  ·  ".join(selected), fontsize=8.5, color="#263b53")
        note = "Arrows: direction. Color: magnitude (shared scale). Gray: masked / undefined. Angles exclude zero targets; zero predictions receive 180°."
        fig.text(0.06, 0.031, note, fontsize=8, color="#526175")
        error = record.get("error") or record.get("diagnostics", {}).get("error")
        if error:
            fig.text(
                0.06,
                0.016,
                textwrap.shorten(
                    f"{record.get('status', record.get('diagnostics', {}).get('status', 'error'))}: {error}",
                    width=165,
                ),
                fontsize=8,
                color="#a72f35",
            )
        fig.savefig(
            path,
            format="png",
            dpi=160,
            metadata={
                "Title": f"{title} evaluation: {identity}",
                "Description": str(sample.get("instruction", "")),
            },
        )
        fig.clear()
    return path


def _display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _rows(mapping, prefix=""):
    for key, value in mapping.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _rows(value, label)
        else:
            yield label, value


def _table(mapping):
    return (
        "<table><tbody>"
        + "".join(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(_display(value))}</td></tr>"
            for label, value in _rows(mapping)
        )
        + "</tbody></table>"
    )


def _metric_cards(overall):
    cards = []
    field = overall.get("field")
    if field:
        unit = str(field.get("vector_error_unit", "")).replace("_per_", " / ").replace("_", " ")
        cards.extend(
            (
                ("Field vector L2", field.get("vector_l2_mean"), f"{unit} · vector weighted"),
                (
                    "Field angular error",
                    field.get("angular_error_deg_mean"),
                    "degrees · directional-target weighted",
                ),
            )
        )
    navigation = overall.get("image_navigation", {})
    for key, label, unit in (
        ("fge", "Final goal error", "image fraction"),
        ("cr", "Collision rate", "fraction of paths"),
        ("plr", "Path length ratio", "dimensionless"),
        ("curv", "Path curvature", "rad / resampled turn"),
    ):
        if key in navigation:
            stats = navigation[key]
            cards.append(
                (label, stats["mean"], f"{unit} · n={stats['count']} · SD={_display(stats['std'])}")
            )
    action = overall.get("action")
    if action:
        cards.append(
            (
                "Action accuracy",
                action.get("accuracy"),
                f"fraction correct · n={action.get('count', 0)}",
            )
        )
        cards.append(
            (
                "Action cross-entropy",
                action["cross_entropy"]["mean"],
                "natural-log units · annotation mean",
            )
        )
    return "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(_display(value))}</strong><small>{html.escape(note)}</small></div>'
        for label, value, note in cards
    )


def _group_table(groups):
    """One row per group; detailed sample statistics remain in summary.json."""
    columns = [("Samples", ("samples",)), ("Failed", ("failed_samples",))]
    for group, definitions in (
        (
            "field",
            (("Vector L2", ("vector_l2_mean",)), ("Angle (deg)", ("angular_error_deg_mean",))),
        ),
        (
            "image_navigation",
            (
                ("FGE", ("fge", "mean")),
                ("CR", ("cr", "mean")),
                ("PLR", ("plr", "mean")),
                ("Curvature", ("curv", "mean")),
            ),
        ),
        ("action", (("Action accuracy", ("accuracy",)),)),
    ):
        if any(group in value for value in groups.values()):
            columns.extend((label, (group, *path)) for label, path in definitions)
    rows = []
    for name, values in groups.items():
        cells = [f"<th>{html.escape(str(name))}</th>"]
        for _, path in columns:
            value = values
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            cells.append(f"<td>{html.escape(_display(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    headers = "<th>Group</th>" + "".join(f"<th>{html.escape(label)}</th>" for label, _ in columns)
    return f'<div class="scroll"><table class="groups"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def write_report(
    output_dir: Path, summary: dict, protocol: dict, figures: list[dict], worst: list[dict]
) -> Path:
    """Write report.html with escaped text, local images, and artifact links.

    Figure paths are absolute within output_dir or relative to output_dir.
    The report uses no scripts, external fonts, CDN, or network resources.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery = []
    for item in figures:
        path = Path(item["path"])
        path = path if path.is_absolute() else output_dir / path
        try:
            relative = path.resolve().relative_to(output_dir.resolve())
        except ValueError as error:
            raise ValueError("Figure paths must stay inside the report directory") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        href = html.escape(quote(relative.as_posix(), safe="/"), quote=True)
        identity = html.escape(str(item["sample_id"]), quote=True)
        instruction = html.escape(str(item.get("instruction", "")))
        gallery.append(
            f'<figure><a href="{href}"><img src="{href}" alt="Evaluation for {identity}" loading="lazy"></a><figcaption><strong>{identity}</strong><p>{instruction}</p></figcaption></figure>'
        )
    overall = summary["overall"]
    cards_html = _metric_cards(overall)
    coverage = f"{summary['evaluated_samples']} / {summary['selected_samples']} annotations evaluated · {'Complete' if summary['complete'] else 'Incomplete'} · {overall.get('failed_samples', 0)} failed predictions"
    breakdowns = "".join(
        f"<details><summary>{label} · {len(summary.get(key, {}))} groups</summary>{_group_table(summary[key])}</details>"
        for key, label in (("by_scene", "By scene"), ("by_category", "By category"))
        if summary.get(key)
    )
    links = "".join(
        f'<a href="{name}">{name}</a>'
        for name in ("summary.json", "results.jsonl", "metrics.csv", "protocol.json")
    )
    worst_html = "".join(
        f"<details><summary>{html.escape(str(row.get('sample_id', f'Sample {index + 1}')))}</summary>{_table(row)}</details>"
        for index, row in enumerate(worst)
    )
    protocol_json = html.escape(json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoFL · Offline evaluation</title><style>
:root{{color-scheme:light;font-family:system-ui,-apple-system,sans-serif;color:#193047;background:#f1f5f9}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1240px;margin:auto;padding:48px 24px 72px}}
header{{border-bottom:1px solid #d5e0ea;padding-bottom:26px}}.eyebrow{{color:#127b83;letter-spacing:.12em;text-transform:uppercase;font-size:12px;font-weight:700}}
h1{{font-size:clamp(28px,5vw,46px);letter-spacing:-.04em;margin:10px 0}}h2{{font-size:23px;margin:36px 0 16px}}p{{line-height:1.6;color:#53677a}}a{{color:#087782;text-decoration:none}}a:hover{{text-decoration:underline}}
nav{{display:flex;flex-wrap:wrap;gap:10px}}nav a{{background:white;border:1px solid #d5e0ea;border-radius:8px;padding:9px 12px;font-size:13px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:24px 0}}.card{{background:white;padding:18px;border:1px solid #dce5ed;border-radius:12px;overflow-wrap:anywhere}}
.card span{{display:block;color:#63768a;font-size:12px}}.card strong{{display:block;font-size:28px;margin:10px 0}}.card small{{display:block;color:#63768a;font-size:11px;line-height:1.5}}table{{width:100%;border-collapse:collapse;background:white;font-size:13px}}th,td{{padding:11px 15px;text-align:left;vertical-align:top;border-bottom:1px solid #e5ebf1;overflow-wrap:anywhere}}th{{width:48%;font-weight:500;color:#52677c}}td{{font-variant-numeric:tabular-nums}}.scroll{{overflow-x:auto}}.groups th{{width:auto;white-space:nowrap}}.groups td{{white-space:nowrap}}.coverage{{padding:12px 15px;background:#e4f1f2;border-radius:8px}}
details{{background:white;border:1px solid #dce5ed;border-radius:10px;overflow:hidden;margin:10px 0}}summary{{cursor:pointer;padding:15px;font-weight:600;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;padding:0 18px 18px;font-size:12px;line-height:1.6}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,470px),1fr));gap:22px}}figure{{margin:0;background:white;border:1px solid #dce5ed;border-radius:14px;overflow:hidden}}img{{display:block;width:100%;height:auto}}figcaption{{padding:16px 20px;overflow-wrap:anywhere;font-size:14px}}figcaption p{{margin-bottom:0}}.muted{{font-size:13px}}footer{{margin-top:36px;color:#6c7f91;font-size:12px}}
@media print{{main{{padding:0}}details{{break-inside:avoid}}figure{{break-inside:avoid}}nav{{display:none}}}}
</style></head><body><main><header><div class="eyebrow">CoFL · Research evaluation</div><h1>Offline evaluation</h1>
<p>Recorded metrics, representative predictions, and the exact evaluation protocol.</p><nav>{links}</nav></header>
<p class="coverage">{html.escape(coverage)}</p>
<section class="cards">{cards_html}</section>
<h2>Metrics and coverage</h2><p class="muted">Field errors use valid-vector weights; angular metrics use directional-target weights. Navigation metrics use equal annotation weights. SD is the sample standard deviation (ddof=1). Unavailable values are shown as —. CR includes failed predictions as unsafe; boundary clamping is reported separately.</p>
<details><summary>Full aggregate metrics, counts, and units</summary>{_table(overall)}</details>{breakdowns}
<details><summary>Evaluation protocol</summary><pre>{protocol_json}</pre></details>
<h2>Selected samples</h2><p class="muted">Figures show a selected subset, not the score distribution. Shared field color scales aid comparison; aggregate conclusions come from the metrics above. Click a figure to open the full-resolution PNG.</p>
<section class="gallery">{"".join(gallery) or "<p>No sample figures were selected.</p>"}</section>
<h2>Worst samples</h2><p class="muted">Ranking follows the recorded metric and selection policy; consult the protocol for its scope.</p>{worst_html or "<p>No ranked samples are available.</p>"}
<footer>Standalone report · No network resources required · Complete per-sample records are available in JSONL and CSV.</footer>
</main></body></html>"""
    path = output_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
