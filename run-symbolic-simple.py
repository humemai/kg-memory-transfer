import argparse
import itertools
import logging
import multiprocessing
import random

from agent import LongTermSimpleAgent

# Disable logging
logging.getLogger().setLevel(logging.CRITICAL)


def run_long_term_simple_experiment(params):
    seed, room_size, max_memory = params
    print(f"Seed: {seed}, Room size: {room_size}, Max memory: {max_memory}")

    agent = LongTermSimpleAgent(
        env_config={
            "terminates_at": 99,
            "room_size": room_size,
        },
        max_long_term_memory_size=max_memory,
        num_samples_for_results=5,
        default_root_dir="./training-results-symbolic-simple/",
        save_results=True,
        seed=seed,
    )
    agent.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run symbolic simple long-term experiments"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (<=1 runs sequentially)",
    )
    args = parser.parse_args()

    seeds = [0, 5, 10, 15, 20]
    room_sizes = ["large-02", "large-02-q"]
    max_memories = [128]

    all_combinations = list(itertools.product(seeds, room_sizes, max_memories))
    random.shuffle(all_combinations)

    print(f"Total combinations: {len(all_combinations)}")

    if args.workers <= 1:
        print("Running sequentially (workers=1)...")
        for combo in all_combinations:
            run_long_term_simple_experiment(combo)
    else:
        num_processes = args.workers
        print(f"Using {num_processes} processes for parallel execution.")
        with multiprocessing.Pool(num_processes) as pool:
            pool.map(run_long_term_simple_experiment, all_combinations)
