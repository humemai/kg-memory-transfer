import logging
import multiprocessing
import random
from pathlib import Path
import argparse

import matplotlib
import yaml
from agent import DQNAgent

matplotlib.use("Agg")
logger = logging.getLogger()
logger.disabled = True
logging.disable(logging.CRITICAL)

# Define network size configurations
network_configs = {
    "gcn": {
        "small": {
            "embedding_dim": 16,
            "num_layers": 2,
            "num_heads": None,
            "num_bases": None,
            "mlp_hidden_layers": 1,
        },
        "big": {
            "embedding_dim": 32,
            "num_layers": 4,
            "num_heads": None,
            "num_bases": None,
            "mlp_hidden_layers": 1,
        },
    },
    "rgcn": {
        "small": {
            "embedding_dim": 16,
            "num_layers": 2,
            "num_heads": None,
            "num_bases": 20,
            "mlp_hidden_layers": 1,
        },
        "big": {
            "embedding_dim": 32,
            "num_layers": 4,
            "num_heads": None,
            "num_bases": 30,
            "mlp_hidden_layers": 1,
        },
    },
    "stare": {
        "small": {
            "embedding_dim": 16,
            "num_layers": 2,
            "num_heads": None,
            "num_bases": None,
            "mlp_hidden_layers": 1,
        },
        "big": {
            "embedding_dim": 32,
            "num_layers": 4,
            "num_heads": None,
            "num_bases": None,
            "mlp_hidden_layers": 1,
        },
    },
    "transformer": {
        "small": {
            "embedding_dim": 16,
            "num_layers": 2,
            "num_heads": 2,
            "num_bases": None,
            "mlp_hidden_layers": 1,
        },
        "big": {
            "embedding_dim": 32,
            "num_layers": 4,
            "num_heads": 4,
            "num_bases": None,
            "mlp_hidden_layers": 1,
        },
    },
}
default_root_dir = "training-results-dqn"


def run_dqn_experiment(exp_params):
    (
        room_size,
        seed,
        architecture_type,
        max_memory,
        forget_policy,
        remember_policy,
        qa_policy,
        explore_policy,
        separate_networks,
        embedding_dim,
        num_layers,
        num_heads,
        num_bases,
        mlp_hidden_layers,
        gamma,
    ) = exp_params

    batch_size = 32
    terminates_at = 99
    num_episodes = 200  # should be between 100 and 500
    num_iterations = (terminates_at + 1) * num_episodes
    target_update_interval = 50  # 50 to 200 is common
    epsilon_decay_until = num_iterations // 2  # 50% of iterations
    warm_start = num_iterations // 10  # 10 percent of the iterations

    print(
        f"room_size: {room_size}, "
        f"seed: {seed}, Architecture: {architecture_type}, "
        f"Max memory: {max_memory}, Forget: {forget_policy}, "
        f"Remember: {remember_policy}, QA: {qa_policy}, "
        f"Explore: {explore_policy}, Separate networks: {separate_networks}, "
        f"Embedding dim: {embedding_dim}, Num layers: {num_layers}, "
        f"Num heads: {num_heads}, Num bases: {num_bases}, "
        f"MLP hidden layers: {mlp_hidden_layers}, "
        f"Gamma: {gamma}"
    )

    # Define architecture-specific parameters
    stare_params = {
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "gcn_drop": 0.0,
        "triple_qual_weight": 0.8,
        "silu_between_layers": True,
        "dropout_between_layers": False,
    }

    gcn_params = {
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "gcn_drop": 0.0,
        "silu_between_layers": True,
        "dropout_between_layers": False,
    }

    rgcn_params = {
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "gcn_drop": 0.0,
        "num_bases": num_bases if num_bases else 30,
        "silu_between_layers": True,
        "dropout_between_layers": False,
    }

    transformer_params = {
        "embedding_dim": embedding_dim,
        "dim_feedforward": embedding_dim * 4,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": 0.0,
    }

    mlp_params = {"num_hidden_layers": mlp_hidden_layers, "dueling_dqn": True}

    agent = DQNAgent(
        env_config={
            "terminates_at": terminates_at,
            "room_size": room_size,
        },
        qa_policy=qa_policy,
        explore_policy=explore_policy,
        forget_policy=forget_policy,
        remember_policy=remember_policy,
        max_long_term_memory_size=max_memory,
        num_samples_for_results={"val": 5, "test": 5},  # to account for stochasticity
        save_results=True,
        default_root_dir=default_root_dir,
        num_iterations=num_iterations,
        replay_buffer_size=num_iterations,
        batch_size=batch_size,
        warm_start=warm_start,
        target_update_interval=target_update_interval,
        epsilon_decay_until=epsilon_decay_until,
        max_epsilon=1.0,
        min_epsilon=0.01,
        gamma=gamma,
        learning_rate=1e-4,
        architecture_type=architecture_type,
        stare_params=stare_params,
        gcn_params=gcn_params,
        rgcn_params=rgcn_params,
        transformer_params=transformer_params,
        mlp_params=mlp_params,
        validation_interval=1,
        plotting_interval=20,
        seed=seed,
        device="cpu",
        ddqn=True,
        use_gradient_clipping=True,
        gradient_clip_value=10.0,
        separate_networks=separate_networks,
    )

    agent.train()


def extract_experiment_params(train_yaml_path):
    """Extract experiment parameters from train.yaml file."""
    try:
        with open(train_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Extract all the parameters that define a unique experiment
        architecture_type = data["kwargs"]["architecture_type"]
        arch_params = data["kwargs"][f"{architecture_type}_params"]

        # Extract num_heads for transformer, num_bases for rgcn
        num_heads = arch_params.get("num_heads", None)
        num_bases = arch_params.get("num_bases", None)

        return {
            "room_size": data["env_config"]["room_size"],
            "seed": data["seed"],
            "architecture_type": architecture_type,
            "max_memory": data["max_long_term_memory_size"],
            "forget_policy": data["forget_policy"],
            "remember_policy": data["remember_policy"],
            "qa_policy": data["qa_policy"],
            "explore_policy": data["explore_policy"],
            "separate_networks": data["kwargs"]["separate_networks"],
            "embedding_dim": arch_params["embedding_dim"],
            "num_layers": arch_params["num_layers"],
            "num_heads": num_heads,
            "num_bases": num_bases,
            "mlp_hidden_layers": data["kwargs"]["mlp_params"]["num_hidden_layers"],
            "gamma": data["kwargs"]["gamma"],
        }
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"Error extracting parameters from {train_yaml_path}: {e}")
        return None


def is_experiment_completed(exp_params, results_root):
    """Check if an experiment with the given parameters is already completed."""
    (
        room_size,
        seed,
        architecture_type,
        max_memory,
        forget_policy,
        remember_policy,
        qa_policy,
        explore_policy,
        separate_networks,
        embedding_dim,
        num_layers,
        _num_heads,
        _num_bases,
        mlp_hidden_layers,
        gamma,
    ) = exp_params

    # Check all subdirectories in the results directory
    results_dir = Path(results_root)
    if not results_dir.exists():
        return False

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        train_yaml_path = subdir / "train.yaml"
        results_yaml_path = subdir / "results.yaml"

        # Check if experiment is completed (has results.yaml)
        if not results_yaml_path.exists():
            continue

        # Extract parameters from existing experiment
        existing_params = extract_experiment_params(train_yaml_path)
        if existing_params is None:
            continue

        # Compare all parameters (excluding num_heads)
        if (
            existing_params["room_size"] == room_size
            and existing_params["seed"] == seed
            and existing_params["architecture_type"] == architecture_type
            and existing_params["max_memory"] == max_memory
            and existing_params["forget_policy"] == forget_policy
            and existing_params["remember_policy"] == remember_policy
            and existing_params["qa_policy"] == qa_policy
            and existing_params["explore_policy"] == explore_policy
            and existing_params["separate_networks"] == separate_networks
            and existing_params["embedding_dim"] == embedding_dim
            and existing_params["num_layers"] == num_layers
            and existing_params["mlp_hidden_layers"] == mlp_hidden_layers
            and existing_params["gamma"] == gamma
        ):
            return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Run DQN experiments")
    # default to cpu_count to avoid hard-coding; user can override via --workers
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="large-02",
        help="Environment room size (e.g., large-02, large-02-q)",
    )
    args = parser.parse_args()

    num_processes = args.workers
    room_sizes = [args.env]
    seeds = [0, 5, 10, 15, 20]
    architecture_types = ["stare", "gcn", "rgcn"]

    max_memories = [128]
    policy_combinations = [
        # (forget, remember, qa, explore, separate_networks)
        ("lru", "rl", "mru", "mru", False),
        ("lru", "rl_global", "mru", "mru", False),
        ("lru", "rl_no_context", "mru", "mru", False),
        ("lru", "rl_global_no_context", "mru", "mru", False),
    ]

    network_sizes = ["small"]
    gamma_values = [0.95]

    all_combinations = []
    num_skipped = 0
    for room_size in room_sizes:
        for seed in seeds:
            for arch_type in architecture_types:
                for memory in max_memories:
                    for policy_combo in policy_combinations:
                        for network_size in network_sizes:
                            for gamma in gamma_values:
                                config = network_configs[arch_type][network_size]

                                (
                                    forget_policy,
                                    remember_policy,
                                    qa_policy,
                                    explore_policy,
                                    separate_networks,
                                ) = policy_combo

                                combo_params = (
                                    room_size,
                                    seed,
                                    arch_type,
                                    memory,
                                    forget_policy,
                                    remember_policy,
                                    qa_policy,
                                    explore_policy,
                                    separate_networks,
                                    config["embedding_dim"],
                                    config["num_layers"],
                                    config["num_heads"],
                                    config["num_bases"],
                                    config["mlp_hidden_layers"],
                                    gamma,
                                )

                                # Only add if not already completed
                                if not is_experiment_completed(
                                    combo_params, default_root_dir
                                ):
                                    all_combinations.append(combo_params)
                                else:
                                    print(
                                        "Skipping already completed experiment: "
                                        f"{combo_params}"
                                    )
                                    num_skipped += 1

    random.shuffle(all_combinations)

    print(f"Total combinations to run: {len(all_combinations)}")
    print(f"Total combinations skipped: {num_skipped}")
    print(f"Running experiments with {num_processes} processes")

    with multiprocessing.Pool(num_processes) as pool:
        pool.map(run_dqn_experiment, all_combinations)


if __name__ == "__main__":
    main()
