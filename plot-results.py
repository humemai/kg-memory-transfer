#!/usr/bin/env python3
"""Render memory-state graphs from a held-out states YAML file.

This is the paper-scoped plotting utility for the RLC 2026 submission. It takes the
`states_q_values_actions_test.yaml` artifact produced by `run-dqn-test.py` and renders
one memory-state graph per step under `./figures/memory_state_graphs/`.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

MEMORY_STATE_OUTPUT_DIR = Path("./figures/memory_state_graphs")
MEMORY_STATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_COLOR_MAP = {
    "room": "#FFE4B5",
    "static": "#87CEFA",
    "moving": "#90EE90",
    "agent": "#D8BFD8",
    "wall": "#D3D3D3",
    "unknown": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--states",
        dest="states_path",
        type=Path,
        required=True,
        help="Path to states_q_values_actions_test.yaml.",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=100,
        help="Maximum number of timesteps to render.",
    )
    return parser.parse_args()


def _load_states_yaml(states_path: Path | None) -> object | None:
    """Load the states YAML file if provided."""

    if not states_path:
        return None

    if not states_path.exists():
        print(f"States file not found: {states_path}")
        return None

    try:
        with states_path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or []
    except yaml.YAMLError as exc:
        print(f"Failed to parse states YAML '{states_path}': {exc}")
        return None


def _extract_state_records(data: object) -> list[dict]:
    """Extract individual state records from the loaded YAML structure."""

    if data is None:
        return []

    records: list[dict] = []

    if isinstance(data, list):
        for record in data:
            if isinstance(record, dict) and record.get("state"):
                records.append(record)
        return records

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for record in value:
                    if isinstance(record, dict) and record.get("state"):
                        records.append(record)
        return records

    return records


def _infer_rooms_from_records(records: list[dict]) -> set[str]:
    rooms: set[str] = set()
    for record in records:
        for entry in record.get("state", []):
            if isinstance(entry, list) and len(entry) >= 3:
                subj, rel, obj = entry[:3]
                if rel in {"north", "east", "south", "west"}:
                    if isinstance(subj, str):
                        rooms.add(subj)
                    if isinstance(obj, str):
                        rooms.add(obj)
                if rel == "at_location" and isinstance(obj, str):
                    rooms.add(obj)
    return rooms


def _infer_objects_from_records(records: list[dict], rooms: set[str]) -> set[str]:
    objects: set[str] = set()
    for record in records:
        for entry in record.get("state", []):
            if isinstance(entry, list) and len(entry) >= 3:
                subj, rel, _ = entry[:3]
                if rel == "at_location" and isinstance(subj, str) and subj != "agent":
                    objects.add(subj)
    # Ensure we don't classify rooms as objects
    return {obj for obj in objects if obj not in rooms}


def _load_environment_metadata(
    states_path: Path | None, records: list[dict]
) -> dict[str, set[str]]:
    """Load room/static/moving names from room-env configs with fallbacks."""

    metadata = {"rooms": set(), "static": set(), "moving": set()}

    inferred_rooms = _infer_rooms_from_records(records)
    inferred_objects = _infer_objects_from_records(records, inferred_rooms)

    def _load_config_metadata(config_path: Path) -> dict[str, set[str]] | None:
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                config = json.load(stream)
            return {
                "rooms": set(config.get("room_names", [])),
                "static": set(config.get("static_names", [])),
                "moving": set(config.get("moving_names", [])),
            }
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Failed to load environment config '{config_path}': {exc}")
            return None

    def _score_candidate(candidate: dict[str, set[str]]) -> tuple[int, int]:
        candidate_objects = candidate.get("static", set()) | candidate.get(
            "moving", set()
        )
        object_overlap = len(inferred_objects & candidate_objects)
        room_overlap = len(inferred_rooms & candidate.get("rooms", set()))
        return (object_overlap, room_overlap)

    def _detect_environment_token(path: Path) -> str | None:
        segments = list(path.parts)
        segments.append(path.stem)
        for segment in segments:
            if "__" in segment:
                parts = segment.split("__")
                if parts:
                    candidate = parts[-1]
                    if candidate:
                        return candidate
        return None

    base_data_dir = (
        Path(__file__).resolve().parents[1] / "room-env" / "room_env" / "data"
    )
    candidate_paths: list[Path] = []

    if states_path is not None and base_data_dir.exists():
        token = _detect_environment_token(states_path)
        if token:
            pattern = f"room-config-{token}-v3.json"
            candidate = base_data_dir / pattern
            if candidate.exists():
                candidate_paths.append(candidate)

    if base_data_dir.exists():
        default_candidate = base_data_dir / "room-config-dev-v3.json"
        if default_candidate.exists():
            candidate_paths.append(default_candidate)

        for config_path in sorted(base_data_dir.glob("room-config-*-v3.json")):
            if config_path not in candidate_paths:
                candidate_paths.append(config_path)

    best_candidate: dict[str, set[str]] | None = None
    best_score = (-1, -1)
    for config_path in candidate_paths:
        loaded = _load_config_metadata(config_path)
        if loaded is None:
            continue
        score = _score_candidate(loaded)
        if score > best_score:
            best_score = score
            best_candidate = loaded

    if best_candidate is not None and best_score > (0, 0):
        metadata["rooms"].update(best_candidate.get("rooms", set()))
        metadata["static"].update(best_candidate.get("static", set()))
        metadata["moving"].update(best_candidate.get("moving", set()))

    metadata["rooms"].update(inferred_rooms)

    if metadata["static"] or metadata["moving"]:
        unresolved = inferred_objects - metadata["static"] - metadata["moving"]
        metadata["static"].update(unresolved)

    return metadata


def _categorize_node(node: str, metadata: dict[str, set[str]]) -> str:
    if node == "agent":
        return "agent"
    if node == "wall":
        return "wall"
    if node in metadata.get("rooms", set()):
        return "room"
    if node in metadata.get("moving", set()):
        return "moving"
    if node in metadata.get("static", set()):
        return "static"
    return "unknown"


def _aggregate_state_edges(state_entries: list) -> dict[tuple[str, str, str], int]:
    edge_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for entry in state_entries:
        if isinstance(entry, list) and len(entry) >= 3:
            subj, rel, obj = entry[:3]
            if isinstance(subj, str) and isinstance(rel, str) and isinstance(obj, str):
                edge_counts[(subj, rel, obj)] += 1
    return edge_counts


def _separate_overlapping_nodes(
    pos: dict, min_distance: float = 0.1, max_iterations: int = 50
) -> dict:
    """Separate overlapping networkx nodes by nudging their positions."""

    nodes = list(pos.keys())
    adjusted_pos = pos.copy()

    for _ in range(max_iterations):
        overlaps_found = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                pos1 = np.array(adjusted_pos[node1])
                pos2 = np.array(adjusted_pos[node2])

                distance = np.linalg.norm(pos1 - pos2)

                if distance < min_distance:
                    overlaps_found = True

                    if distance == 0:
                        angle = random.uniform(0, 2 * np.pi)
                        separation = (
                            np.array([np.cos(angle), np.sin(angle)]) * min_distance
                        )
                    else:
                        direction = (pos1 - pos2) / distance
                        separation = direction * (min_distance - distance) / 2

                    adjusted_pos[node1] = pos1 + separation
                    adjusted_pos[node2] = pos2 - separation

        if not overlaps_found:
            break

    return adjusted_pos


def _draw_memory_state_graph(
    edge_counts: dict[tuple[str, str, str], int],
    metadata: dict[str, set[str]],
    index: int,
    output_path: Path,
) -> None:
    if not edge_counts:
        return

    G = nx.DiGraph()
    for (subj, rel, obj), count in edge_counts.items():
        label = f"{rel} ({count})" if count > 1 else rel
        G.add_edge(subj, obj, label=label, weight=count)

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
    ax.set_facecolor("white")

    pos = nx.kamada_kawai_layout(G, scale=1.6)
    pos = _separate_overlapping_nodes(pos, min_distance=0.25, max_iterations=150)

    node_colors = []
    for node in G.nodes():
        category = _categorize_node(node, metadata)
        color = NODE_COLOR_MAP.get(category, NODE_COLOR_MAP["unknown"])
        node_colors.append(color)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=2600,
        alpha=0.9,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=12,
        font_weight="bold",
        font_color="black",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        arrows=True,
        arrowsize=25,
        arrowstyle="->",
        alpha=0.7,
        width=2,
        connectionstyle="arc3,rad=0.1",
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels,
        ax=ax,
        font_size=12,
        font_color="black",
        font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
    )

    ax.set_title(
        f"Memory State {index:03d}",
        fontsize=16,
        fontweight="bold",
    )
    ax.axis("off")

    if pos:
        xs = np.fromiter((coord[0] for coord in pos.values()), dtype=float)
        ys = np.fromiter((coord[1] for coord in pos.values()), dtype=float)
        if xs.size and ys.size:
            x_span = xs.max() - xs.min()
            y_span = ys.max() - ys.min()
            pad_x = max(x_span * 0.15, 0.2)
            pad_y = max(y_span * 0.15, 0.2)
            ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
            ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

    # fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.resolve()}")


def render_memory_state_graphs(states_data: object, states_path: Path | None) -> None:
    """Render and save memory state graphs for each timestep."""

    records = _extract_state_records(states_data)
    if not records:
        print("No state records found; skipping memory state graph generation.")
        return

    metadata = _load_environment_metadata(states_path, records)

    unknown_nodes: set[str] = set()

    for idx, record in enumerate(records):
        if idx >= 100:
            break

        state_entries = record.get("state")
        if not isinstance(state_entries, list):
            continue

        edge_counts = _aggregate_state_edges(state_entries)
        if not edge_counts:
            continue

        for subj, _, obj in edge_counts:
            for node in (subj, obj):
                if _categorize_node(node, metadata) == "unknown":
                    unknown_nodes.add(node)

        output_path = MEMORY_STATE_OUTPUT_DIR / f"memory_state_{idx:03d}.pdf"
        _draw_memory_state_graph(edge_counts, metadata, idx, output_path)

    if unknown_nodes:
        formatted = ", ".join(sorted(unknown_nodes))
        print(
            "Warning: some nodes were not categorized and were rendered as 'unknown':"
            f" {formatted}"
        )


def main() -> None:
    args = parse_args()
    states_path = args.states_path
    states_data = _load_states_yaml(states_path)

    if states_data is None:
        print("Failed to load states data; no memory-state graphs were rendered.")
        return

    records = _extract_state_records(states_data)
    if args.limit < len(records):
        states_data = records[: args.limit]

    render_memory_state_graphs(states_data, states_path)


if __name__ == "__main__":
    main()
