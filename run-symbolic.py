import argparse
import itertools
import logging
import multiprocessing
import random

from rdflib import Namespace

from agent import LongTermAgent

ns = Namespace("https://humem.ai/ontology#")

# Disable logging
logging.getLogger().setLevel(logging.CRITICAL)


def run_long_term_experiment(params):
    (
        seed,
        room_size,
        qa_policy,
        explore_policy,
        forget_policy,
        remember_policy,
        max_memory,
    ) = params
    print(
        (
            f"Seed: {seed}, Room size: {room_size}, QA: {qa_policy}, "
            f"Explore: {explore_policy}, Forget: {forget_policy}, "
            f"Remember: {remember_policy}, Max memory: {max_memory}"
        )
    )

    agent = LongTermAgent(
        env_config={
            "terminates_at": 99,
            "room_size": room_size,
        },
        qa_policy=qa_policy,
        explore_policy=explore_policy,
        forget_policy=forget_policy,
        remember_policy=remember_policy,
        max_long_term_memory_size=max_memory,
        num_samples_for_results=5,
        default_root_dir="./training-results-symbolic/",
        save_results=True,
        seed=seed,
    )
    agent.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run symbolic long-term experiments")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (<=1 runs sequentially)",
    )
    args = parser.parse_args()

    seeds = [0, 5, 10, 15, 20]
    room_sizes = ["large-02", "large-02-q"]
    # qa_policies = ["intersection_vote"]
    # explore_policies = ["intersection_vote"]
    # forget_policies = ["intersection_vote"]

    qa_policies = ["mru"]
    explore_policies = ["mru"]
    forget_policies = ["lru"]

    remember_policies = ["all", "novel", "random_0.5"]
    max_memories = [128]

    all_combinations = list(
        itertools.product(
            seeds,
            room_sizes,
            qa_policies,
            explore_policies,
            forget_policies,
            remember_policies,
            max_memories,
        )
    )

    random.shuffle(all_combinations)

    print(f"Total combinations: {len(all_combinations)}")

    if args.workers <= 1:
        print("Running sequentially (workers=1)...")
        for combo in all_combinations:
            run_long_term_experiment(combo)
    else:
        num_processes = args.workers
        print(f"Using {num_processes} processes for parallel execution.")
        with multiprocessing.Pool(num_processes) as pool:
            pool.map(run_long_term_experiment, all_combinations)
