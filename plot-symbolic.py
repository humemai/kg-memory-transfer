import os
from collections import defaultdict
from glob import glob
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

for path in tqdm(glob("./training-results-symbolic/*/results.yaml")):
    with open(path, "r", encoding="utf-8") as f:
        results = yaml.safe_load(f)

    duration_hours = compute_duration_hours(path)

    test_mean = results["test_score"]["mean"]
    test_std = results["test_score"]["std"]
    num_episodes = results["num_episodes"]

    with open(path.replace("results.yaml", "train.yaml"), "r", encoding="utf-8") as f:
        hp = yaml.safe_load(f)

    room_size = hp["env_config"]["room_size"]
    qa_policy = hp["qa_policy"]
    explore_policy = hp["explore_policy"]
    forget_policy = hp.get("forget_policy", None)
    if forget_policy.lower() == "random":
        continue
    remember_policy = hp.get("remember_policy", "all")
    memory_size = hp.get("max_long_term_memory_size", None)
    seed = hp["seed"]

    config_key = (
        room_size,
        qa_policy,
        explore_policy,
        forget_policy,
        remember_policy,
        memory_size,
    )
    results_by_config[config_key].append(
        (test_mean, test_std, num_episodes, duration_hours)
    )

# Build a DataFrame from the aggregated results
records = []
for config, score_episode_pairs in sorted(results_by_config.items()):
    (
        room_size,
        qa_policy,
        explore_policy,
        forget_policy,
        remember_policy,
        memory_size,
    ) = config

    scores = [pair[0] for pair in score_episode_pairs]
    stds = [pair[1] for pair in score_episode_pairs]
    total_episodes = sum(pair[2] for pair in score_episode_pairs)
    durations = [pair[3] for pair in score_episode_pairs if np.isfinite(pair[3])]

    # Use weighted average for std when multiple results exist
    if len(scores) == 1:
        n = 1
        test_mean = scores[0]
        test_std = stds[0]
        avg_duration_hours = durations[0] if durations else float("nan")
    else:
        n = len(scores)
        test_mean = np.mean(scores)
        test_std = np.std(scores)
        avg_duration_hours = float(np.mean(durations)) if durations else float("nan")

    records.append(
        {
            "room_size": room_size,
            "test_mean": test_mean,
            "test_std": test_std,
            "qa_policy": qa_policy,
            "explore_policy": explore_policy,
            "forget_policy": forget_policy,
            "remember_policy": remember_policy,
            "memory_size": memory_size,
            "n_seeds": n,
            "avg_duration_hours": avg_duration_hours,
        }
    )

df = pd.DataFrame(records)
pd.set_option("display.precision", 4)

# Restrict aggregation to the symbolic transfer baselines reported in the RLC paper.
df = df[
    (df["memory_size"] == 128)
    & (df["qa_policy"] == "mru")
    & (df["explore_policy"] == "mru")
    & (df["forget_policy"] == "lru")
    & (df["remember_policy"].isin(["all", "novel", "random_0.5"]))
].copy()

# Define all room sizes to process
room_filter = ["large-02", "large-02-q"]

# Ensure room_filter is a list for consistent processing
if isinstance(room_filter, str):
    room_filter = [room_filter]

# Process each room size
all_room_data = []
for current_room in room_filter:
    room_label = room_display_name(current_room)
    room_key = room_slug(current_room)
    print(f"\n{'='*100}")
    print("PROCESSING ROOM SIZE: " f"{current_room} (alias: {room_label})")
    print(f"{'='*100}")

    # Filter for the specific room size
    df_filtered = df[df["room_size"] == current_room].copy()

    if df_filtered.empty:
        print(f"No data found for room size: {current_room}")
        continue

    # Store data for combined processing later
    all_room_data.append((current_room, room_label, df_filtered.copy()))

    # Remove room_size column for processing
    df_filtered = df_filtered.drop(columns="room_size")
    # Create separate DataFrames for each memory size
    memory_sizes = sorted(df_filtered["memory_size"].unique())
    dataframes_by_memory = {}

    print(f"Creating {len(memory_sizes)} DataFrames for memory sizes: {memory_sizes}")
    print("=" * 80)

    for memory_size in memory_sizes:
        # Filter data for this specific memory size
        memory_df = df_filtered[df_filtered["memory_size"] == memory_size].drop(
            columns="memory_size"
        )
        # Sort by mean_score in descending order
        memory_df = memory_df.sort_values(by="test_mean", ascending=False).reset_index(
            drop=True
        )

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

    # Find best configuration across all memory sizes for this room
    best_overall = df_filtered.loc[df_filtered["test_mean"].idxmax()]
    print("\nBest overall configuration:")
    print(f"  Memory Size: {best_overall['memory_size']}")
    print(f"  QA Policy: {best_overall['qa_policy']}")
    print(f"  Explore Policy: {best_overall['explore_policy']}")
    print(f"  MM Forget Policy: {best_overall['forget_policy']}")
    print(f"  MM Remember Policy: {best_overall['remember_policy']}")
    print(f"  Score: {best_overall['test_mean']:.3f} ± {best_overall['test_std']:.3f}")
    if pd.notna(best_overall.get("avg_duration_hours")):
        print("  Avg Duration (h): " f"{best_overall['avg_duration_hours']:.3f}")

    # Export individual room results to JSON only
    print(f"\n=== EXPORTING FILES ({room_label}) ===")

    # Export global results (all data for the room size, sorted by mean_score)
    global_section = df_filtered.sort_values(
        by="test_mean", ascending=False
    ).reset_index(drop=True)

    # Save JSON file with room name in filename for individual processing
    room_name = room_key
    global_filename = f"./data/results_symbolic_{room_name}.json"
    global_section.to_json(global_filename, orient="records", indent=2)
    print(f"Individual room results exported to {global_filename}")

# Create combined results and single markdown file across all rooms
if all_room_data:
    print(f"\n{'='*100}")
    print("CREATING COMBINED RESULTS FILE")
    print(f"{'='*100}")

    # Combine all room data
    combined_df = pd.concat(
        [room_data[2] for room_data in all_room_data], ignore_index=True
    )

    if "room_size" in combined_df.columns:
        combined_df.loc[:, "room_size"] = combined_df["room_size"].map(
            room_display_name
        )

    # Save combined JSON
    combined_filename = "./data/results_symbolic.json"
    combined_df.to_json(combined_filename, orient="records", indent=2)
    print(f"Combined results exported to {combined_filename}")

    # Create single markdown file with sections for each room
    markdown_filename = "./data/results_symbolic.md"
    markdown_content = "# Results\n\n"
    markdown_content += f"Room sizes: {[room[1] for room in all_room_data]}\n\n"

    for room_key, room_label, room_df in all_room_data:
        room_df_clean = room_df.drop(columns="room_size")
        memory_sizes = sorted(room_df["memory_size"].unique())

        markdown_content += f"## {room_label}\n\n"
        markdown_content += f"Total configurations: {len(room_df_clean)}\n"
        markdown_content += f"Memory sizes: {memory_sizes}\n\n"

        # Best overall for this room
        best_in_room = room_df_clean.loc[room_df_clean["test_mean"].idxmax()]
        markdown_content += "### Best Overall Configuration\n\n"
        markdown_content += f"Memory Size: {best_in_room['memory_size']}\n\n"
        markdown_content += f"QA Policy: {best_in_room['qa_policy']}\n\n"
        markdown_content += f"Explore Policy: {best_in_room['explore_policy']}\n\n"
        markdown_content += f"MM Forget Policy: {best_in_room['forget_policy']}\n\n"
        markdown_content += f"MM Remember Policy: {best_in_room['remember_policy']}\n\n"
        markdown_content += (
            "Score: "
            f"{best_in_room['test_mean']:.3f} ± {best_in_room['test_std']:.3f}\n\n"
        )
        if pd.notna(best_in_room.get("avg_duration_hours")):
            markdown_content += (
                f"Avg Duration (h): {best_in_room['avg_duration_hours']:.3f}\n\n"
            )

        # Overall results table
        markdown_content += "### Overall Results (All Memory Sizes)\n\n"
        room_df_sorted = room_df_clean.sort_values(by="test_mean", ascending=False)
        markdown_content += room_df_sorted.to_markdown(index=False, floatfmt=".3f")
        markdown_content += "\n\n"

        # Results by memory size
        for memory_size in memory_sizes:
            memory_df = room_df_clean[room_df_clean["memory_size"] == memory_size].drop(
                columns="memory_size"
            )
            memory_df = memory_df.sort_values(
                by="test_mean", ascending=False
            ).reset_index(drop=True)

            markdown_content += f"### Memory Size {memory_size}\n\n"
            markdown_content += f"Total configurations: {len(memory_df)}\n\n"
            markdown_content += memory_df.to_markdown(index=False, floatfmt=".3f")
            markdown_content += "\n\n"

    with open(markdown_filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Single markdown file exported to {markdown_filename}")

# # === PLOTTING CONFIGURATION ===
# include_shading = True  # Toggle shading for std deviation
# log_xaxis = True
# log_yaxis = False
# line_width = 2.5

# qa_policies = ["mra", "mru", "mfu"]
# explore_policies = ["mra", "mru", "mfu"]
# mm_policies = ["fifo", "lru", "lfu"]

# # Generate plots for all room sizes
# print(f"\n{'='*100}")
# print("GENERATING PLOTS")
# print(f"{'='*100}")

# # Create plots for each room size individually
# for room_key, room_label, room_data in all_room_data:
#     print(f"\nGenerating plot for {room_label}...")

#     room_name_clean = room_slug(room_key)
#     file_path = f"./data/results_symbolic_{room_name_clean}.json"

#     # Load and prepare data
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"File not found: {file_path}, skipping...")
#         continue

#     df_plot = pd.DataFrame(data)
#     df_plot = df_plot.sort_values(
#         by=["qa_policy", "explore_policy", "forget_policy", "memory_size"]
#     )

#     # Create 3×3 subplot grid (3 QA policies × 3 exploration policies)
#     fig, axes = plt.subplots(3, 3, figsize=(24, 24), sharex=True, sharey=True)

#     for i, qa_policy in enumerate(qa_policies):
#         for j, explore_policy in enumerate(explore_policies):
#             ax = axes[i, j]
#             for forget_policy in mm_policies:
#                 sub_df = df_plot[
#                     (df_plot["qa_policy"] == qa_policy)
#                     & (df_plot["explore_policy"] == explore_policy)
#                     & (df_plot["forget_policy"] == forget_policy)
#                 ]
#                 if len(sub_df) > 0:  # Check if data exists for this combination
#                     x = sub_df["memory_size"]
#                     y = sub_df["test_mean"]
#                     yerr = sub_df["test_std"]

#                     ax.plot(x, y, label=forget_policy.upper(), linewidth=line_width)
#                     if include_shading:
#                         ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

#             # Format only the first row and the first column for titles
#             if i == 0:
#                 ax.set_title(
#                     f"BFS explore with {explore_policy.replace('_', ' ')}", fontsize=22
#                 )
#             if j == 0:
#                 ax.text(
#                     -0.2,
#                     0.5,
#                     f"QA: {qa_policy.replace('_', ' ').upper()}",
#                     rotation=90,
#                     transform=ax.transAxes,
#                     fontsize=22,
#                     verticalalignment="center",
#                 )

#             # Add legend to each subplot
#             ax.legend(
#                 title="Long-Term Memory Management",
#                 fontsize=18,
#                 title_fontsize=18,
#                 loc="best",
#             )
#             ax.grid(True, which="both", linestyle="--", linewidth=0.5)
#             ax.tick_params(axis="both", which="major", labelsize=18)

#             if log_xaxis:
#                 ax.set_xscale("log")
#                 ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
#                 ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

#             if log_yaxis:
#                 ax.set_yscale("log")
#                 ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=None))
#                 ax.yaxis.set_minor_locator(ticker.NullLocator())
#                 ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

#     # Shared axis labels
#     fig.text(
#         0.5,
#         0.01,
#         "Long-Term Memory Capacity" + (" (log scale)" if log_xaxis else ""),
#         ha="center",
#         fontsize=24,
#     )
#     fig.text(
#         0.01,
#         0.5,
#         "Mean Score" + (" (log scale)" if log_yaxis else ""),
#         va="center",
#         rotation="vertical",
#         fontsize=24,
#     )
#     fig.suptitle(
#         f"Mean Score vs. Long-Term Memory Capacity - {room_label}",
#         fontsize=28,
#         y=0.99,
#     )

#     plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])

#     # Save combined plots
#     pdf_path = f"./figures/agent_test_performance_symbolic_{room_name_clean}.pdf"
#     # png_path = f"./figures/agent_test_performance_symbolic_{room_name_clean}.png"

#     plt.savefig(pdf_path, bbox_inches="tight")
#     # plt.savefig(png_path, bbox_inches="tight")
#     plt.close()  # Close the figure to free memory

#     print(f"Plot saved: {pdf_path}")
#     # print(f"Plot saved: {png_path}")

#     # Generate individual plots for each QA policy × exploration policy combination
#     print(f"Generating individual plots for {room_label}...")

#     for i, qa_policy in enumerate(qa_policies):
#         for j, explore_policy in enumerate(explore_policies):
#             # Create individual plot
#             fig, ax = plt.subplots(1, 1, figsize=(10, 8))

#             for forget_policy in mm_policies:
#                 sub_df = df_plot[
#                     (df_plot["qa_policy"] == qa_policy)
#                     & (df_plot["explore_policy"] == explore_policy)
#                     & (df_plot["forget_policy"] == forget_policy)
#                 ]
#                 if len(sub_df) > 0:  # Check if data exists for this combination
#                     x = sub_df["memory_size"]
#                     y = sub_df["test_mean"]
#                     yerr = sub_df["test_std"]

#                     ax.plot(x, y, label=forget_policy.upper(), linewidth=line_width)
#                     if include_shading:
#                         ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

#             # Configure individual plot
#             title = (
#                 f"QA: {qa_policy.replace('_', ' ').upper()}, "
#                 f"Explore: {explore_policy.replace('_', ' ').upper()}"
#             )
#             ax.set_title(title, fontsize=20)
#             ax.set_xlabel(
#                 "Long-Term Memory Capacity" + (" (log scale)" if log_xaxis else ""),
#                 fontsize=18,
#             )
#             ax.set_ylabel(
#                 "Mean Score" + (" (log scale)" if log_yaxis else ""), fontsize=18
#             )

#             ax.legend(
#                 title="Long-Term Memory Management",
#                 fontsize=16,
#                 title_fontsize=16,
#                 loc="best",
#             )
#             ax.grid(True, which="both", linestyle="--", linewidth=0.5)
#             ax.tick_params(axis="both", which="major", labelsize=16)

#             if log_xaxis:
#                 ax.set_xscale("log")
#                 ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
#                 ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

#             if log_yaxis:
#                 ax.set_yscale("log")
#                 ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=None))
#                 ax.yaxis.set_minor_locator(ticker.NullLocator())
#                 ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

#             plt.tight_layout()

#             Save individual plots
#             qa_clean = qa_policy.replace("_", "-")
#             explore_clean = explore_policy.replace("_", "-")
#             individual_pdf_path = (
#                 f"./figures/agent_test_performance_symbolic_"
#                 f"{room_name_clean}_qa={qa_clean}_explore={explore_clean}.pdf"
#             )
#             individual_png_path = (
#                 f"./figures/agent_test_performance_symbolic_"
#                 f"{room_name_clean}_qa={qa_clean}_explore={explore_clean}.png"
#             )

#             plt.savefig(individual_pdf_path, bbox_inches="tight")
#             plt.savefig(individual_png_path, bbox_inches="tight")
#             plt.close()  # Close the figure to free memory

#             print(f"Individual plot saved: {individual_pdf_path}")
#             print(f"Individual plot saved: {individual_png_path}")

print(f"\n{'='*100}")
print("ALL PROCESSING COMPLETE")
print(f"{'='*100}")
