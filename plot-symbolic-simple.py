import os
import json
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

# Create directories if they don't exist (mirror plot-symbolic.py)
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


def collect_results(root: str = "./training-results-symbolic-simple/") -> pd.DataFrame:
    """Collect results across seeds, grouped by room size and memory size.

    For the SIMPLE variant, policies are fixed (uniform QA, random explore/forget),
    but we still record them if present for completeness.
    """
    results_by_config = defaultdict(list)

    for res_path in tqdm(glob(os.path.join(root, "*/results.yaml"))):
        train_path = os.path.join(os.path.dirname(res_path), "train.yaml")
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                results = yaml.safe_load(f)
            with open(train_path, "r", encoding="utf-8") as f:
                hp = yaml.safe_load(f)
        except FileNotFoundError:
            continue

        if not results or not hp:
            continue

        test_mean = results.get("test_score", {}).get("mean")
        test_std = results.get("test_score", {}).get("std")
        num_episodes = results.get("num_episodes")

        room_size = (hp or {}).get("env_config", {}).get("room_size")
        memory_size = (hp or {}).get("max_long_term_memory_size")
        seed = (hp or {}).get("seed")

        # Optional (may be absent in SIMPLE runs)
        qa_policy = (hp or {}).get("qa_policy", "uniform")
        explore_policy = (hp or {}).get("explore_policy", "random")
        forget_policy = (hp or {}).get("forget_policy", "random")
        remember_policy = (hp or {}).get("remember_policy", "all")

        # Guard against missing critical fields
        if (
            room_size is None
            or memory_size is None
            or test_mean is None
            or test_std is None
        ):
            continue

        key = (room_size, memory_size)
        duration_hours = compute_duration_hours(res_path)
        results_by_config[key].append(
            (
                test_mean,
                test_std,
                num_episodes,
                qa_policy,
                explore_policy,
                forget_policy,
                remember_policy,
                seed,
                duration_hours,
            )
        )

    # Aggregate across seeds per (room, memory)
    records = []
    for (room_size, memory_size), entries in sorted(results_by_config.items()):
        scores = [e[0] for e in entries]
        stds = [e[1] for e in entries]
        qa_vals = {e[3] for e in entries}
        explore_vals = {e[4] for e in entries}
        forget_vals = {e[5] for e in entries}
        remember_vals = {e[6] for e in entries}
        durations = [e[8] for e in entries if np.isfinite(e[8])]

        n = len(scores)
        agg_mean = float(np.mean(scores)) if n > 0 else None
        # As in plot-symbolic.py, use std of means across seeds
        agg_std = float(np.std(scores)) if n > 1 else float(stds[0]) if n == 1 else None

        def _one_or_join(values):
            return (
                ",".join(sorted(values))
                if len(values) > 1
                else next(iter(values), None)
            )

        avg_duration_hours = float(np.mean(durations)) if durations else float("nan")

        records.append(
            {
                "room_size": room_size,
                "memory_size": memory_size,
                "test_mean": agg_mean,
                "test_std": agg_std,
                "qa_policy": _one_or_join(qa_vals),
                "explore_policy": _one_or_join(explore_vals),
                "forget_policy": _one_or_join(forget_vals),
                "remember_policy": _one_or_join(remember_vals),
                "n_seeds": n,
                "avg_duration_hours": avg_duration_hours,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df[df["memory_size"] == 128].copy()
    return df


def main():
    # Collect and aggregate
    df = collect_results()

    if df.empty:
        print("No results found under ./training-results-symbolic-simple/")
        return

    pd.set_option("display.precision", 4)

    # Define room filters to process
    room_filter = ["large-02", "large-02-q"]

    # Ensure room_filter is a list for consistent processing
    if isinstance(room_filter, str):
        room_filter = [room_filter]

    all_room_data = []

    for current_room in room_filter:
        room_label = room_display_name(current_room)
        room_key = room_slug(current_room)
        print(f"\n{'='*100}")
        print(
            "PROCESSING ROOM SIZE: "
            f"{current_room} (alias: {room_label})"
        )
        print(f"{'='*100}")

        df_filtered = df[df["room_size"] == current_room].copy()
        if df_filtered.empty:
            print(f"No data for room size: {current_room}")
            continue

        # Store data for combined processing later
        all_room_data.append((current_room, room_label, df_filtered.copy()))

        # Sort by performance
        global_section = df_filtered.sort_values(
            by=["test_mean", "memory_size"], ascending=[False, True]
        ).reset_index(drop=True)

        memory_sizes = sorted(df_filtered["memory_size"].unique())
        print(
            f"Creating {len(memory_sizes)} DataFrames for memory sizes: {memory_sizes}"
        )
        print("=" * 80)

        for memory_size in memory_sizes:
            mem_df = (
                df_filtered[df_filtered["memory_size"] == memory_size]
                .drop(columns=["room_size"])
                .sort_values(by="test_mean", ascending=False)
                .reset_index(drop=True)
            )
            print(f"\n=== DataFrame for Memory Size: {memory_size} ===\n")
            print(mem_df)
            print("-" * 80)

        # Best overall
        best_overall = df_filtered.loc[df_filtered["test_mean"].idxmax()]
        print("\nBest overall configuration:")
        print(f"  Memory Size: {best_overall['memory_size']}")
        if "qa_policy" in best_overall:
            print(f"  QA Policy: {best_overall['qa_policy']}")
        if "explore_policy" in best_overall:
            print(f"  Explore Policy: {best_overall['explore_policy']}")
        if "forget_policy" in best_overall:
            print(f"  Forget Policy: {best_overall['forget_policy']}")
        if "remember_policy" in best_overall:
            print(f"  Remember Policy: {best_overall['remember_policy']}")
        print(
            f"  Score: {best_overall['test_mean']:.3f} ± {best_overall['test_std']:.3f}"
        )
        avg_duration_hours = best_overall.get("avg_duration_hours")
        if pd.notna(avg_duration_hours):
            print(f"  Avg Duration (hrs): {avg_duration_hours:.2f}")

        # === EXPORTS ===
        json_path = f"./data/results_symbolic_simple_{room_key}.json"

        export_section = global_section.copy()
        if "room_size" in export_section.columns:
            export_section.loc[:, "room_size"] = room_label

        export_section.to_json(json_path, orient="records", indent=2)
        print(f"Individual room results exported to {json_path}")

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
        combined_filename = "./data/results_symbolic_simple.json"
        combined_df.to_json(combined_filename, orient="records", indent=2)
        print(f"Combined results exported to {combined_filename}")

        # Create single markdown file with sections for each room
        markdown_filename = "./data/results_symbolic_simple.md"
        markdown_content = "# Results (Simple)\n\n"
        markdown_content += (
            f"Room sizes: {[room[1] for room in all_room_data]}\n\n"
        )

        for room_key, room_label, room_df in all_room_data:
            room_df_clean = room_df.drop(columns="room_size")
            memory_sizes = sorted(room_df["memory_size"].unique())

            markdown_content += f"## {room_label}\n\n"
            markdown_content += f"Room size: {room_label}\n\n"
            markdown_content += f"Total configurations: {len(room_df_clean)}\n"
            markdown_content += f"Memory sizes: {memory_sizes}\n\n"

            # Best overall for this room
            best_in_room = room_df_clean.loc[room_df_clean["test_mean"].idxmax()]
            markdown_content += "### Best Overall Configuration\n\n"
            markdown_content += f"Memory Size: {best_in_room['memory_size']}\n\n"
            if "qa_policy" in best_in_room and pd.notna(best_in_room["qa_policy"]):
                markdown_content += f"QA Policy: {best_in_room['qa_policy']}\n\n"
            if "explore_policy" in best_in_room and pd.notna(
                best_in_room["explore_policy"]
            ):
                markdown_content += (
                    f"Explore Policy: {best_in_room['explore_policy']}\n\n"
                )
            if "forget_policy" in best_in_room and pd.notna(
                best_in_room["forget_policy"]
            ):
                markdown_content += (
                    f"Forget Policy: {best_in_room['forget_policy']}\n\n"
                )
            if "remember_policy" in best_in_room and pd.notna(
                best_in_room["remember_policy"]
            ):
                markdown_content += (
                    f"Remember Policy: {best_in_room['remember_policy']}\n\n"
                )
            markdown_content += (
                f"Score: {best_in_room['test_mean']:.3f} ± "
                f"{best_in_room['test_std']:.3f}\n"
            )
            avg_duration = best_in_room.get("avg_duration_hours")
            if pd.notna(avg_duration):
                markdown_content += (
                    f"Avg Duration (hrs): {avg_duration:.2f}\n"
                )
            markdown_content += "\n"

            # Overall results table
            markdown_content += "### Overall Results (All Memory Sizes)\n\n"
            room_df_sorted = room_df_clean.sort_values(by="test_mean", ascending=False)
            markdown_content += room_df_sorted.to_markdown(index=False, floatfmt=".3f")
            markdown_content += "\n\n"

            # Results by memory size
            for memory_size in memory_sizes:
                mem_df = (
                    room_df_clean[room_df_clean["memory_size"] == memory_size]
                    .drop(columns=["memory_size"])
                    .sort_values(by="test_mean", ascending=False)
                    .reset_index(drop=True)
                )
                markdown_content += f"### Memory Size {memory_size}\n\n"
                markdown_content += f"Total configurations: {len(mem_df)}\n\n"
                markdown_content += mem_df.to_markdown(index=False, floatfmt=".3f")
                markdown_content += "\n\n"

        with open(markdown_filename, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Single markdown file exported to {markdown_filename}")

        # # === PLOTTING ===
        # include_shading = True
        # log_xaxis = True
        # log_yaxis = False
        # line_width = 2.5

        # print(f"\n{'='*100}")
        # print("GENERATING PLOTS")
        # print(f"{'='*100}")

        # # Create plots for each room size individually
        # for room_key, room_label, room_data in all_room_data:
        #     print(f"\nGenerating plot for {room_label}...")

        #     room_name_clean = room_slug(room_key)
        #     json_path = f"./data/results_symbolic_simple_{room_name_clean}.json"

        #     try:
        #         with open(json_path, "r", encoding="utf-8") as f:
        #             data = json.load(f)
        #     except FileNotFoundError:
        #         print(f"File not found: {json_path}, skipping...")
        #         continue

        #     df_plot = pd.DataFrame(data).sort_values(by=["memory_size"])

        #     _, ax = plt.subplots(1, 1, figsize=(10, 8))

        #     x = df_plot["memory_size"].to_numpy()
        #     y = df_plot["test_mean"].to_numpy()
        #     yerr = df_plot["test_std"].to_numpy()

        #     ax.plot(x, y, label=room_label, linewidth=line_width)
        #     if include_shading:
        #         ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

        #     ax.set_title(
        #         f"Simple (uniform QA, random explore/forget) - {room_label}",
        #         fontsize=20,
        #     )
        #     ax.set_xlabel(
        #         "Long-Term Memory Capacity" + (" (log scale)" if log_xaxis else ""),
        #         fontsize=18,
        #     )
        #     ax.set_ylabel(
        #         "Mean Score" + (" (log scale)" if log_yaxis else ""), fontsize=18
        #     )
        #     ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        #     ax.legend(fontsize=14, loc="best")

        #     if log_xaxis:
        #         ax.set_xscale("log")
        #         ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        #         ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        #     if log_yaxis:
        #         ax.set_yscale("log")
        #         ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=None))
        #         ax.yaxis.set_minor_locator(ticker.NullLocator())
        #         ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        #     plt.tight_layout()

            # pdf = (
            #     "./figures/agent_test_performance_symbolic_simple_"
            #     f"{room_name_clean}.pdf"
            # )
            # png = (
            #     "./figures/agent_test_performance_symbolic_simple_"
            #     f"{room_name_clean}.png"
            # )
            # plt.savefig(pdf, bbox_inches="tight")
            # plt.savefig(png, bbox_inches="tight")
            # plt.close()
            # print(f"Plot saved: {pdf}")
            # print(f"Plot saved: {png}")

        # Create combined plot showing all room sizes
        # print("\nGenerating combined plot...")

        # _, ax = plt.subplots(1, 1, figsize=(12, 8))

        # for room_key, room_label, room_data in all_room_data:
        #     room_name_clean = room_slug(room_key)
        #     json_path = f"./data/results_symbolic_simple_{room_name_clean}.json"

        #     try:
        #         with open(json_path, "r", encoding="utf-8") as f:
        #             data = json.load(f)
        #     except FileNotFoundError:
        #         continue

        #     df_plot = pd.DataFrame(data).sort_values(by=["memory_size"])

        #     x = df_plot["memory_size"].to_numpy()
        #     y = df_plot["test_mean"].to_numpy()
        #     yerr = df_plot["test_std"].to_numpy()

        #     ax.plot(x, y, label=room_label, linewidth=line_width)
        #     if include_shading:
        #         ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

        # ax.set_title(
        #     "Simple (uniform QA, random explore/forget) - All Rooms", fontsize=20
        # )
        # ax.set_xlabel(
        #     "Long-Term Memory Capacity" + (" (log scale)" if log_xaxis else ""),
        #     fontsize=18,
        # )
        # ax.set_ylabel("Mean Score" + (" (log scale)" if log_yaxis else ""), fontsize=18)
        # ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        # ax.legend(fontsize=14, loc="best")

        # if log_xaxis:
        #     ax.set_xscale("log")
        #     ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        #     ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        # if log_yaxis:
        #     ax.set_yscale("log")
        #     ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=None))
        #     ax.yaxis.set_minor_locator(ticker.NullLocator())
        #     ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        # plt.tight_layout()

        # combined_pdf = "./figures/agent_test_performance_symbolic_simple_combined.pdf"
        # combined_png = "./figures/agent_test_performance_symbolic_simple_combined.png"
        # plt.savefig(combined_pdf, bbox_inches="tight")
        # plt.savefig(combined_png, bbox_inches="tight")
        # plt.close()
        # print(f"Combined plot saved: {combined_pdf}")
        # print(f"Combined plot saved: {combined_png}")


if __name__ == "__main__":
    main()
