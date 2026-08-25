"""Runtime validation for the standalone versioned dataset contract."""

from __future__ import annotations

from typing import Any

from dataset_create.trajectory import SCHEMA_VERSION


class SchemaError(ValueError):
    pass


def _require(mapping: dict[str, Any], fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise SchemaError(f"{context} is missing fields: {missing}")


def validate_trajectory(trajectory: dict[str, Any]) -> None:
    _require(
        trajectory,
        (
            "trajectory_id",
            "scene_id",
            "start_position",
            "start_rotation_xyzw",
            "goal_position",
            "final_position",
            "final_rotation_xyzw",
            "shortest_path",
            "region_sequence",
            "connection_sequence",
            "decision_events",
            "action_ids",
            "metrics",
        ),
        "trajectory",
    )
    if len(trajectory["connection_sequence"]) != len(trajectory["region_sequence"]) - 1:
        raise SchemaError("connection_sequence must connect every adjacent region pair")
    if not trajectory["decision_events"]:
        raise SchemaError("Every retained trajectory must have a decision event")
    if not trajectory["action_ids"] or trajectory["action_ids"][-1] != 0:
        raise SchemaError("replay must terminate with STOP")
    if any(action_id not in (0, 1, 2, 3) for action_id in trajectory["action_ids"]):
        raise SchemaError("trajectory contains an unknown primitive action id")
    metrics = trajectory["metrics"]
    _require(
        metrics,
        (
            "length_m",
            "region_count",
            "decision_region_count",
            "decision_event_count",
        ),
        "trajectory metrics",
    )
    if int(metrics["region_count"]) != len(trajectory["region_sequence"]):
        raise SchemaError("metrics.region_count disagrees with region_sequence")
    if int(metrics["decision_event_count"]) != len(trajectory["decision_events"]):
        raise SchemaError("metrics.decision_event_count disagrees with decision_events")
    decision_regions = {int(event["region_id"]) for event in trajectory["decision_events"]}
    if int(metrics["decision_region_count"]) != len(decision_regions):
        raise SchemaError("metrics.decision_region_count disagrees with decision_events")
    for event in trajectory["decision_events"]:
        _require(
            event,
            (
                "relation_id",
                "region_id",
                "region_sequence_index",
                "action_index",
                "position",
                "rotation_xyzw",
                "incoming",
                "selected",
                "alternatives",
            ),
            "decision event",
        )
        if not event["alternatives"]:
            raise SchemaError("decision event has no visible alternative")
        if not 0 <= int(event["action_index"]) < len(trajectory["action_ids"]):
            raise SchemaError("decision event action_index is outside replay actions")
        sequence_index = int(event["region_sequence_index"])
        if not 0 <= sequence_index < len(trajectory["region_sequence"]):
            raise SchemaError("decision event region_sequence_index is invalid")
        if int(trajectory["region_sequence"][sequence_index]) != int(event["region_id"]):
            raise SchemaError("decision event region does not match region_sequence")
        for name, branch in (
            ("incoming", event["incoming"]),
            ("selected", event["selected"]),
            *[("alternative", value) for value in event["alternatives"]],
        ):
            _require(
                branch,
                ("connection_id", "region_id", "anchor_position"),
                f"{name} branch",
            )


def validate_dataset(dataset: dict[str, Any]) -> None:
    _require(dataset, ("schema_version", "metadata", "episodes"), "dataset")
    if dataset["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"Unsupported schema {dataset['schema_version']}, expected {SCHEMA_VERSION}"
        )
    identifiers = set()
    for trajectory in dataset["episodes"]:
        validate_trajectory(trajectory)
        identifier = trajectory["trajectory_id"]
        if identifier in identifiers:
            raise SchemaError(f"Duplicate trajectory id: {identifier}")
        identifiers.add(identifier)
