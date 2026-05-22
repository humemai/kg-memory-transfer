import os
from collections import defaultdict
from glob import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

# Create directories if they don't exist
os.makedirs("./data", exist_ok=True)
os.makedirs("./figures", exist_ok=True)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


ROOM_NAME_ALIASES = {
    "large-02": "train",
    "large-02-q": "test",
}


def room_display_name(room: str) -> str:
    return ROOM_NAME_ALIASES.get(room, room)


def room_slug(room: str) -> str:
    return room_display_name(room).replace("-", "_")


def compute_duration_hours(results_path: str) -> float:
    result_file = Path(results_path)
    start_time = None

    for parent in result_file.parents:
        candidate = parent.name.split("__")[0]
        try:
            start_time = datetime.strptime(candidate, TIMESTAMP_FORMAT)
            break
        except ValueError:
            continue

    if start_time is None:
        return float("nan")

    try:
        end_time = datetime.fromtimestamp(result_file.stat().st_mtime)
    except FileNotFoundError:
        return float("nan")

    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds < 0:
        return float("nan")

    return duration_seconds / 3600.0


# Collect results grouped by config (excluding seed)
results_by_config = defaultdict(list)

for path in tqdm(
    glob("./training-results-dqn/*/results.yaml")
    + glob("./training-results-dqn/*/*/results.yaml")
):
    with open(path, "r", encoding="utf-8") as f:
        results = yaml.safe_load(f)

    duration_hours = compute_duration_hours(path)

    test_mean = results["test_score"]["mean"]
    test_std = results["test_score"]["std"]

    # Find best validation score and its std
    val_scores = results["validation_score"]
    if val_scores:
        best_val = max(val_scores, key=lambda x: x["mean"])
        val_mean = best_val["mean"]
        val_std = best_val["std"]
    else:
        val_mean = float("nan")
        val_std = float("nan")

    with open(path.replace("results.yaml", "train.yaml"), "r", encoding="utf-8") as f:
        hp = yaml.safe_load(f)

    # Extract environment size from env_config
    room_size = hp["env_config"]["room_size"]

    # Extract parameters from kwargs section for DQN
    kwargs = hp.get("kwargs", {})
    architecture_type = kwargs["architecture_type"]
    max_memory = hp["max_long_term_memory_size"]
    forget_policy = hp.get("forget_policy", None)
    remember_policy = hp.get("remember_policy", "all")
    separate_networks = kwargs.get("separate_networks", False)

    qa_policy = hp.get("qa_policy", None)
    explore_policy = hp.get("explore_policy", None)
    gamma = kwargs.get("gamma")

    seed = hp["seed"]

    # Extract network configuration parameters from kwargs
    if architecture_type == "stare":
        params = kwargs.get("stare_params", {})
    elif architecture_type == "gcn":
        params = kwargs.get("gcn_params", {})
    elif architecture_type == "rgcn":
        params = kwargs.get("rgcn_params", {})
    elif architecture_type == "transformer":
        params = kwargs.get("transformer_params", {})
    else:
        params = {}

    embedding_dim = params.get("embedding_dim", kwargs.get("embedding_dim", None))
    num_layers = params.get("num_layers", kwargs.get("num_layers", None))

    if embedding_dim == 16:
        network_size = "small"
    elif embedding_dim == 32:
        network_size = "big"
    else:
        network_size = "custom"

    config_key = (
        room_size,
        architecture_type,
        max_memory,
        forget_policy,
        remember_policy,
        qa_policy,
        explore_policy,
        separate_networks,
        network_size,
        gamma,
    )
    results_by_config[config_key].append(
        (test_mean, test_std, val_mean, val_std, duration_hours)
    )

# Build a DataFrame from the aggregated results
records = []
for config, score_tuples in sorted(results_by_config.items()):
    (
        room_size,
        architecture_type,
        max_memory,
        forget_policy,
        remember_policy,
        qa_policy,
        explore_policy,
        separate_networks,
        network_size,
        gamma,
    ) = config

    if len(score_tuples) == 1:
        n = 1
        test_mean, test_std, val_mean, val_std, duration_hours = score_tuples[0]
        avg_duration_hours = (
            duration_hours if np.isfinite(duration_hours) else float("nan")
        )
    else:
        # If there are multiple results, average them
        n = len(score_tuples)
        test_mean = np.mean([t[0] for t in score_tuples])
        test_std = np.std([t[0] for t in score_tuples])
        val_mean = np.mean([t[2] for t in score_tuples])
        val_std = np.std([t[2] for t in score_tuples])
        durations = [t[4] for t in score_tuples if np.isfinite(t[4])]
        avg_duration_hours = float(np.mean(durations)) if durations else float("nan")

    records.append(
        {
            "room_size": room_size,
            "architecture_type": architecture_type,
            "max_memory": max_memory,
            "forget_policy": forget_policy,
            "remember_policy": remember_policy,
            "qa_policy": qa_policy,
            "explore_policy": explore_policy,
            "separate_networks": separate_networks,
            "network_size": network_size,
            "gamma": gamma,
            "test_mean": test_mean,
            "test_std": test_std,
            "val_mean": val_mean,
            "val_std": val_std,
            "n": n,
            "avg_duration_hours": avg_duration_hours,
        }
    )

df = pd.DataFrame(records)
pd.set_option("display.precision", 4)

# Restrict aggregation to the learned transfer variants reported in the RLC paper.
df = df[
    (df["architecture_type"].isin(["gcn", "rgcn", "stare"]))
    & (df["max_memory"] == 128)
    & (df["forget_policy"] == "lru")
    & (df["remember_policy"].isin(["rl", "rl_no_context", "rl_global", "rl_global_no_context"]))
    & (df["qa_policy"] == "mru")
    & (df["explore_policy"] == "mru")
    & (df["separate_networks"] == False)
    & (df["network_size"] == "small")
    & (df["gamma"] == 0.95)
].copy()

# Define room sizes to process (environment sizes)
room_filter = ["large-02", "large-02-q"]

# Initialize combined results storage
all_room_data = {}
combined_markdown_content = "# DQN Results\n\n"

# Process each room size
for room_size in room_filter:
    room_label = room_display_name(room_size)
    room_key = room_slug(room_size)
    print(f"\n{'='*100}")
    print(
        "PROCESSING ENVIRONMENT SIZE: "
        f"{room_size} (alias: {room_label})"
    )
    print(f"{'='*100}")

    # Filter for the specific room size
    df_filtered = df[df["room_size"] == room_size].drop(columns="room_size")

    if df_filtered.empty:
        print(f"No data found for room size: {room_size}")
        continue
    else:
        # Store data for this room size
        all_room_data[room_size] = df_filtered

        # Create separate DataFrames for each memory size
        memory_sizes = sorted(df_filtered["max_memory"].unique())
        dataframes_by_memory = {}

        print(
            f"Creating {len(memory_sizes)} DataFrames for memory sizes: {memory_sizes}"
        )
        print("=" * 80)

        for memory_size in memory_sizes:
            # Filter data for this specific memory size
            memory_df = df_filtered[df_filtered["max_memory"] == memory_size].drop(
                columns="max_memory"
            )
            # Sort by test_mean in descending order
            memory_df = memory_df.sort_values(
                by="test_mean", ascending=False
            ).reset_index(drop=True)

            # Store in dictionary
            dataframes_by_memory[memory_size] = memory_df

            # Display the DataFrame
            print(f"\n=== DataFrame for Memory Size: {memory_size} ===")
            print()
            print(memory_df)
            print("-" * 80)

        # Summary statistics
        print("\n=== SUMMARY ===")
        print(f"Total number of memory sizes: {len(memory_sizes)}")
        print(f"Memory sizes analyzed: {memory_sizes}")

        # Find best configuration across all memory sizes for this environment size
        best_overall = df_filtered.loc[df_filtered["test_mean"].idxmax()]
        print("\nBest overall configuration:")
        print(f"  Architecture: {best_overall['architecture_type']}")
        print(f"  Network Size: {best_overall['network_size']}")
        print(f"  Max Memory: {best_overall['max_memory']}")
        print(f"  Forget Policy: {best_overall['forget_policy']}")
        print(f"  Remember Policy: {best_overall['remember_policy']}")
        print(f"  QA Policy: {best_overall['qa_policy']}")
        print(f"  Explore Policy: {best_overall['explore_policy']}")
        print(f"  Separate Networks: {best_overall['separate_networks']}")
        print(f"  Gamma: {best_overall['gamma']}")
        print(
            "  Test Score: "
            f"{best_overall['test_mean']:.3f} ± {best_overall['test_std']:.3f}"
        )
        print(
            "  Val Score: "
            f"{best_overall['val_mean']:.3f} ± {best_overall['val_std']:.3f}"
        )
        if pd.notna(best_overall.get("avg_duration_hours")):
            print(
                "  Avg Duration (h): "
                f"{best_overall['avg_duration_hours']:.3f}"
            )

        # Export results for this room size
        print(f"\n=== EXPORTING FILES FOR {room_label} ===")

        # Export global results (all data for the environment size, sorted by test_mean)
        global_section = df_filtered.sort_values(
            by="test_mean", ascending=False
        ).reset_index(drop=True)

        # Save JSON file for this room size
        json_filename = f"./data/results_dqn_{room_key}.json"
        global_section.to_json(json_filename, orient="records", indent=2)
        print(f"Global results exported to {json_filename}")

        # Add to combined markdown content
        combined_markdown_content += f"## Room: {room_label}\n\n"
        combined_markdown_content += f"Total configurations: {len(global_section)}\n"
        combined_markdown_content += f"Memory sizes: {memory_sizes}\n\n"

        # Add best overall configuration info
        if not df_filtered.empty:
            best = df_filtered.loc[df_filtered["test_mean"].idxmax()]
            combined_markdown_content += "**Best overall configuration:**\n"
            combined_markdown_content += (
                f"- Architecture: {best['architecture_type']}\n"
            )
            combined_markdown_content += f"- Network Size: {best['network_size']}\n"
            combined_markdown_content += f"- Max Memory: {best['max_memory']}\n"
            combined_markdown_content += f"- Forget Policy: {best['forget_policy']}\n"
            combined_markdown_content += (
                f"- Remember Policy: {best['remember_policy']}\n"
            )
            combined_markdown_content += f"- QA Policy: {best['qa_policy']}\n"
            combined_markdown_content += f"- Explore Policy: {best['explore_policy']}\n"
            combined_markdown_content += (
                f"- Separate Networks: {best['separate_networks']}\n"
            )
            combined_markdown_content += f"- Gamma: {best['gamma']}\n"
            combined_markdown_content += (
                f"- Test Score: {best['test_mean']:.3f} ± {best['test_std']:.3f}\n"
            )
            combined_markdown_content += (
                f"- Val Score: {best['val_mean']:.3f} ± {best['val_std']:.3f}\n\n"
            )
            if pd.notna(best.get("avg_duration_hours")):
                combined_markdown_content += (
                    f"- Avg Duration (h): {best['avg_duration_hours']:.3f}\n\n"
                )

        # Add overall table
        combined_markdown_content += "### Overall Results (All Memory Sizes)\n\n"
        combined_markdown_content += global_section.to_markdown(
            index=False, floatfmt=".3f"
        )
        combined_markdown_content += "\n\n"

        # Add separate table for each memory size
        for memory_size in memory_sizes:
            memory_df = df_filtered[df_filtered["max_memory"] == memory_size].drop(
                columns="max_memory"
            )
            memory_df = memory_df.sort_values(
                by="test_mean", ascending=False
            ).reset_index(drop=True)

            combined_markdown_content += f"#### Memory Size {memory_size}\n\n"
            combined_markdown_content += f"Total configurations: {len(memory_df)}\n\n"
            combined_markdown_content += memory_df.to_markdown(
                index=False, floatfmt=".3f"
            )
            combined_markdown_content += "\n\n"

# Export combined results
print(f"\n{'='*100}")
print("EXPORTING COMBINED RESULTS")
print(f"{'='*100}")

# Save combined markdown file
markdown_filename = "./data/results_dqn.md"
with open(markdown_filename, "w", encoding="utf-8") as f:
    f.write(combined_markdown_content)
print(f"Combined markdown table exported to {markdown_filename}")

print(f"\n{'='*100}")
print("ALL DQN PROCESSING COMPLETE")
print(f"{'='*100}")
