"""Shared tabular exports and explicit sample- and vector-weighted statistics."""

import numpy as np
import pandas as pd

FIELD_WEIGHTS = {
    "vector_l2_mean": "valid_count",
    "magnitude_error_mean": "valid_count",
    "angular_error_deg_mean": "directional_count",
    "directional_accuracy_30deg": "directional_count",
}
NAVIGATION_METRICS = ("fge", "cr", "plr", "curv")


def result_table(records):
    rows = []
    for record in records:
        diagnostics = record["diagnostics"]
        row = {"sample_id": record["sample_id"]}
        for key in (
            "status",
            "error",
            "scene_id",
            "category",
            "annotation_key",
            "instruction",
            "observation_id",
            "episode_id",
            "clamped_steps",
        ):
            row[key] = diagnostics.get(key)
        for task, metrics in record["metrics"].items():
            for key, value in metrics.items():
                if value is None or isinstance(value, (int, float, bool, str)):
                    row[f"{task}.{key}"] = value
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _statistics(values, total):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "mean": float(values.mean()) if len(values) else None,
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "count": len(values),
        "undefined_count": total - len(values),
    }


def _column(frame, name):
    return frame[name] if name in frame else pd.Series(index=frame.index, dtype=float)


def _aggregate(frame, tasks):
    result = {
        "samples": len(frame),
        "successful_samples": int(_column(frame, "status").eq("ok").sum()),
        "failed_samples": int(_column(frame, "status").eq("failed").sum()),
    }
    if "field" in tasks:
        field = {"sample_statistics": {}}
        for metric, weight_name in FIELD_WEIGHTS.items():
            values = _column(frame, f"field.{metric}")
            weights = _column(frame, f"field.{weight_name}")
            selected = values.notna() & weights.gt(0)
            denominator = weights[selected].sum()
            field[metric] = (
                float((values[selected] * weights[selected]).sum() / denominator)
                if denominator
                else None
            )
            field["sample_statistics"][metric] = _statistics(values, len(frame))
        for name in (
            "valid_count",
            "directional_count",
            "total_count",
            "masked_count",
            "zero_target_count",
            "zero_prediction_directional_count",
        ):
            field[name] = int(_column(frame, f"field.{name}").sum())
        units = _column(frame, "field.vector_error_unit").dropna().unique().tolist()
        field["vector_error_unit"] = units[0] if len(units) == 1 else None
        field["aggregation"] = (
            "Selected-vector weighted errors; directional-target weighted angles and accuracy. Sample statistics use equal annotation weights and sample std (ddof=1)."
        )
        result["field"] = field
    if "image_navigation" in tasks:
        navigation = {
            name: _statistics(_column(frame, f"image_navigation.{name}"), len(frame))
            for name in NAVIGATION_METRICS
        }
        navigation.update(
            coordinate_unit="image_fraction",
            curv_unit="radians_per_resampled_turn",
            aggregation="Equal weight per annotation; std uses ddof=1. CR counts failed predictions as unsafe.",
        )
        navigation["clamped_samples"] = int(_column(frame, "clamped_steps").gt(0).sum())
        result["image_navigation"] = navigation
    if "action" in tasks:
        target = _column(frame, "action.target")
        prediction = _column(frame, "action.prediction")
        selected = target.notna() & prediction.notna()
        matrix = np.zeros((4, 4), dtype=np.int64)
        if selected.any():
            np.add.at(matrix, (target[selected].astype(int), prediction[selected].astype(int)), 1)
        result["action"] = {
            "count": int(selected.sum()),
            "unscored_count": int((~selected).sum()),
            "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else None,
            "cross_entropy": _statistics(_column(frame, "action.cross_entropy"), len(frame)),
            "confusion_matrix": matrix.tolist(),
            "class_order": ["STOP", "FORWARD", "LEFT", "RIGHT"],
            "confusion_matrix_axes": "rows=target, columns=prediction",
        }
    return result


def summarize_results(writer, *, selected_samples, tasks):
    table = result_table(writer.records())
    table.to_csv(writer.output_dir / "metrics.csv", index=False)
    summary = {
        "selected_samples": selected_samples,
        "evaluated_samples": len(table),
        "complete": len(table) == selected_samples,
        "overall": _aggregate(table, tasks),
    }
    summary["by_scene"] = (
        {
            str(key): _aggregate(group, tasks)
            for key, group in table.groupby("scene_id", sort=True, dropna=False)
        }
        if "scene_id" in table
        else {}
    )
    summary["by_category"] = (
        {
            str(key): _aggregate(group, tasks)
            for key, group in table.groupby("category", sort=True, dropna=False)
        }
        if "category" in table
        else {}
    )
    writer.write_summary(summary)
    return summary, table
