import logging
import multiprocessing
import random
from pathlib import Path
import argparse

import matplotlib
import yaml
from agent import SimpleDQNAgent

matplotlib.use("Agg")
logger = logging.getLogger()
logger.disabled = True
logging.disable(logging.CRITICAL)

# Define network size configurations
network_configs = {
    "lstm": {
        "small": {
            "embedding_dim": 16,
            "num_layers": 2,
            "num_heads": None,
            "mlp_hidden_layers": 1,
        },
        "big": {
            "embedding_dim": 32,
            "num_layers": 4,
            "num_heads": None,
            "mlp_hidden_layers": 1,
        },
    },
    "transformer": {
        "small": {
            "embedding_dim": 16,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_hidden_layers": 1,
        },
        "big": {
            "embedding_dim": 32,
            "num_layers": 4,
            "num_heads": 4,
            "mlp_hidden_layers": 1,
        },
    },
}
default_root_dir = "training-results-simple-dqn"


def run_simple_dqn_experiment(run_params):
    (
        p_room_size,
        p_seed_value,
        p_architecture_type,
        p_embedding_dim,
        p_num_layers,
        p_num_heads,
        p_mlp_hidden_layers,
        p_gamma_value,
        p_max_memory,
    ) = run_params

    batch_size = 32
    terminates_at = 99
    num_episodes = 200  # should be between 100 and 500
    num_iterations = (terminates_at + 1) * num_episodes
    target_update_interval = 50  # 50 to 200 is common
    epsilon_decay_until = num_iterations // 2  # 50% of iterations
    warm_start = num_iterations // 10  # 10 percent of the iterations

    print(
        f"room_size: {p_room_size}, seed: {p_seed_value}, arch: {p_architecture_type}, "
        f"E: {p_embedding_dim}, L: {p_num_layers}, H: {p_num_heads}, "
        f"MLP: {p_mlp_hidden_layers}, gamma: {p_gamma_value}, max_mem: {p_max_memory}"
    )

    transformer_params = {
        "embedding_dim": p_embedding_dim,
        "dim_feedforward": p_embedding_dim * 4,
        "num_layers": p_num_layers,
        "num_heads": p_num_heads,
        "dropout": 0.0,
    }

    mlp_params = {"num_hidden_layers": p_mlp_hidden_layers, "dueling_dqn": False}

    agent = SimpleDQNAgent(
        env_config={
            "terminates_at": terminates_at,
            "room_size": p_room_size,
        },
        num_samples_for_results={"val": 1, "test": 1},  # to account for determinism
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
        gamma=p_gamma_value,
        learning_rate=1e-4,
        architecture_type=p_architecture_type,
        transformer_params=transformer_params,
        lstm_params={"embedding_dim": p_embedding_dim, "num_layers": p_num_layers},
        mlp_params=mlp_params,
        validation_interval=1,
        plotting_interval=20,
        seed=p_seed_value,
        device="cpu",
        ddqn=True,
        use_gradient_clipping=True,
        gradient_clip_value=10.0,
        max_long_term_memory_size=p_max_memory,
    )

    agent.train()


def extract_experiment_params_simple(train_yaml_path):
    """Extract experiment parameters from train.yaml file for simple DQN."""
    try:
        with open(train_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Extract all the parameters that define a unique experiment
        params = {
            "room_size": data["env_config"]["room_size"],
            "seed": data["seed"],
            "architecture_type": data["architecture_type"],
            "embedding_dim": data[f"{data['architecture_type']}_params"][
                "embedding_dim"
            ],
            "num_layers": data[f"{data['architecture_type']}_params"]["num_layers"],
            "num_heads": data.get("transformer_params", {}).get("num_heads"),
            "mlp_hidden_layers": data["mlp_params"]["num_hidden_layers"],
            "gamma": data["gamma"],
            "max_long_term_memory_size": data.get("max_long_term_memory_size", 100),
        }
        return params
    except (FileNotFoundError, KeyError, yaml.YAMLError):
        return None


def is_experiment_completed_simple(check_params, results_root):
    """Check if an experiment with the given parameters is already completed."""
    (
        c_room_size,
        c_seed,
        c_architecture_type,
        c_embedding_dim,
        c_num_layers,
        c_num_heads,
        c_mlp_hidden_layers,
        c_gamma,
        c_max_memory,
    ) = check_params

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
        existing_params = extract_experiment_params_simple(train_yaml_path)
        if existing_params is None:
            continue

        # Compare all parameters
        if (
            existing_params["room_size"] == c_room_size
            and existing_params["seed"] == c_seed
            and existing_params["architecture_type"] == c_architecture_type
            and existing_params["embedding_dim"] == c_embedding_dim
            and existing_params["num_layers"] == c_num_layers
            and existing_params["num_heads"] == c_num_heads
            and existing_params["mlp_hidden_layers"] == c_mlp_hidden_layers
            and existing_params["gamma"] == c_gamma
            and existing_params["max_long_term_memory_size"] == c_max_memory
        ):
            return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Simple DQN experiments")
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
    architecture_types = ["lstm", "transformer"]
    network_sizes = ["small"]
    gamma_values = [0.95]
    memory_sizes = [128]

    all_combinations = []

    for room_size in room_sizes:
        for seed_value in seeds:
            for arch_type in architecture_types:
                for network_size in network_sizes:
                    for gamma_value in gamma_values:
                        for mem_size in memory_sizes:
                            config = network_configs[arch_type][network_size]

                            exp_params = (
                                room_size,
                                seed_value,
                                arch_type,
                                config["embedding_dim"],
                                config["num_layers"],
                                config["num_heads"],
                                config["mlp_hidden_layers"],
                                gamma_value,
                                mem_size,
                            )

                            # Only add if not already completed
                            if not is_experiment_completed_simple(
                                exp_params, default_root_dir
                            ):
                                all_combinations.append(exp_params)
                            else:
                                print(
                                    "Skipping already completed experiment: "
                                    f"{exp_params}"
                                )

    random.shuffle(all_combinations)

    print(f"Total combinations to run: {len(all_combinations)}")
    print(f"Running experiments with {num_processes} processes")

    with multiprocessing.Pool(num_processes) as pool:
        pool.map(run_simple_dqn_experiment, all_combinations)
