import argparse
import logging
import os
import multiprocessing as mp
from pathlib import Path

import matplotlib
import yaml

from agent import DQNAgent


matplotlib.use("Agg")
logger = logging.getLogger()
logger.disabled = True
logging.disable(logging.CRITICAL)


def find_best_checkpoint(run_dir: Path) -> str | None:
    """Pick a checkpoint to load from a training run directory.

    Prefers files named with "val-score=...pt"; if multiple, pick the highest score.
    Falls back to the most recent .pt if no val-score files exist.
    For separate_networks=True runs, a single .pt file may contain all policies.
    """
    candidates = list(run_dir.glob("*val-score=*.pt"))
    if candidates:
        def score_of(p: Path) -> float:
            try:
                s = p.name.split("val-score=")[-1].split(".pt")[0]
                return float(s)
            except (IndexError, ValueError):
                return float("-inf")

        best = max(candidates, key=score_of)
        return str(best)

    any_pts = list(run_dir.glob("*.pt"))
    if any_pts:
        latest = max(any_pts, key=lambda p: p.stat().st_mtime)
        return str(latest)

    return None


def build_agent_from_train_yaml(
    train_yaml: Path, test_root_dir: str, room_size: str
) -> tuple[DQNAgent, str | None, bool]:
    """Construct a DQNAgent for testing using a training run's params.

    room_size controls the test environment (default provided by CLI: 'large-02-q').
    Returns the agent, the checkpoint path (if found), and whether separate_networks.
    """
    with open(train_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    run_dir = train_yaml.parent
    checkpoint = find_best_checkpoint(run_dir)

    # Top-level env
    env_cfg = data.get("env_config", {})
    terminates_at = env_cfg.get("terminates_at", 99)

    # Result sampling and seed
    num_samples_for_results = data.get(
        "num_samples_for_results", {"val": 1, "test": 1}
    )
    seed = int(data.get("seed", 0))

    architecture_type = data.get("kwargs", {}).get("architecture_type") or data.get(
        "architecture_type", "stare"
    )
    # Arch params are nested under kwargs in DQNAgent's saved train.yaml
    stare_params = data.get("kwargs", {}).get(
        "stare_params", data.get("stare_params", None)
    )
    gcn_params = data.get("kwargs", {}).get(
        "gcn_params", data.get("gcn_params", None)
    )
    rgcn_params = data.get("kwargs", {}).get(
        "rgcn_params", data.get("rgcn_params", None)
    )
    transformer_params = data.get("kwargs", {}).get(
        "transformer_params", data.get("transformer_params", None)
    )
    mlp_params = data.get("kwargs", {}).get("mlp_params", data.get("mlp_params", None))

    separate_networks = bool(data.get("kwargs", {}).get("separate_networks", False))

    # Test-only safe hyperparameters to satisfy constructor assertion
    # (we are not training here): ensure batch_size <= warm_start <= replay_buffer_size
    test_batch_size = 1
    test_warm_start = 1
    test_replay_buffer_size = 1

    # Write test outputs under a stable per-training-run folder and env suffix,
    # then timestamped. Example:
    # training-results-dqn/<run_name>__test__large-02-q/<timestamp>/...
    test_output_base = os.path.join(
        test_root_dir, f"{run_dir.name}__test__{room_size}"
    )

    agent = DQNAgent(
        env_config={
            "terminates_at": terminates_at,
            "room_size": room_size,
        },
        qa_policy=data.get("qa_policy", "mra"),
        explore_policy=data.get("explore_policy", "mra"),
        forget_policy=data.get("forget_policy", "lru"),
        remember_policy=data.get("remember_policy", "all"),
        max_long_term_memory_size=int(data.get("max_long_term_memory_size", 100)),
        num_samples_for_results=num_samples_for_results,
        save_results=True,
        default_root_dir=test_output_base,
        num_iterations=0,
        replay_buffer_size=test_replay_buffer_size,
        warm_start=test_warm_start,
        batch_size=test_batch_size,
        target_update_interval=int(data.get("target_update_interval", 50)),
        epsilon_decay_until=float(data.get("epsilon_decay_until", 1.0)),
        max_epsilon=float(data.get("max_epsilon", 1.0)),
        min_epsilon=float(data.get("min_epsilon", 0.01)),
        gamma=float(data.get("kwargs", {}).get("gamma", data.get("gamma", 0.99))),
        learning_rate=float(data.get("learning_rate", 1e-4)),
        architecture_type=architecture_type,
        stare_params=stare_params,
        gcn_params=gcn_params,
        rgcn_params=rgcn_params,
        transformer_params=transformer_params,
        mlp_params=mlp_params,
        validation_interval=int(data.get("validation_interval", 1)),
        plotting_interval=int(data.get("plotting_interval", 20)),
        seed=seed,
        device=str(data.get("device", "cpu")),
        ddqn=bool(data.get("ddqn", True)),
        use_gradient_clipping=bool(data.get("use_gradient_clipping", True)),
        gradient_clip_value=float(data.get("gradient_clip_value", 10.0)),
        separate_networks=separate_networks,
    )

    return agent, checkpoint, separate_networks


def discover_training_runs(results_root: str) -> list[Path]:
    root = Path(results_root)
    if not root.exists():
        return []
    runs = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        if (sub / "results.yaml").exists() and (sub / "train.yaml").exists():
            runs.append(sub)
    return runs


def test_already_completed(run_dir: Path, test_root_dir: str, room_size: str) -> bool:
    """Check if this training run has already been tested.

    We consider it done if there exists any timestamped subdir under
    <test_root_dir>/<run_name>__test__<env>/ with a results.yaml.
    """
    base = Path(test_root_dir) / f"{run_dir.name}__test__{room_size}"
    if not base.exists():
        return False
    for sub in base.iterdir():
        if sub.is_dir() and (sub / "results.yaml").exists():
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Run DQN tests on large-02-q")
    parser.add_argument(
        "--results-root",
        type=str,
        default="training-results-dqn",
        help="Root directory containing completed training runs",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="large-02-q",
        help="Environment room size for testing (e.g., 'large-02-q')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel processes to use (default: 1)",
    )
    args = parser.parse_args()

    default_root_dir = args.results_root
    runs = discover_training_runs(default_root_dir)
    if not runs:
        print(f"No completed training runs found under {default_root_dir}")
        return

    # Filter out ones already completed (best-effort; worker re-checks too)
    pending_runs = [
        r for r in sorted(runs)
        if not test_already_completed(r, default_root_dir, args.env)
    ]
    if not pending_runs:
        print("All runs already have test results. Nothing to do.")
        return

    print(
        f"Found {len(runs)} training runs, {len(pending_runs)} pending for testing."
    )

    if args.workers <= 1:
        print("Running sequentially (workers=1)...")
        for run_dir in pending_runs:
            name, ok, msg = _worker_run_test(run_dir, default_root_dir, args.env)
            print(f"[{name}] {'OK' if ok else 'SKIP/ERR'} - {msg}")
        return

    print(f"Running in parallel with {args.workers} workers...")
    # Use spawn to be safe with libraries
    ctx = mp.get_context("spawn")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(
                _worker_run_test, run_dir, default_root_dir, args.env
            ): run_dir.name
            for run_dir in pending_runs
        }
        for fut in as_completed(futures):
            try:
                name, ok, msg = fut.result()
            except (RuntimeError, OSError, ValueError) as e:
                name = futures[fut]
                ok = False
                msg = f"unhandled exception: {e}"
            print(f"[{name}] {'OK' if ok else 'SKIP/ERR'} - {msg}")


def _worker_run_test(
    run_dir: Path | str, test_root_dir: str, room_size: str
) -> tuple[str, bool, str]:
    """Worker that builds the agent from a run dir and executes test.

    Returns (run_name, ok, message).
    """
    try:
        run_dir = Path(run_dir)
        run_name = run_dir.name

        if test_already_completed(run_dir, test_root_dir, room_size):
            return run_name, False, "already completed"

        train_yaml = run_dir / "train.yaml"
        try:
            agent, checkpoint, _ = build_agent_from_train_yaml(
                train_yaml, test_root_dir, room_size
            )
        except (yaml.YAMLError, OSError, ValueError) as e:
            return run_name, False, f"failed to build agent: {e}"

        if not checkpoint or not os.path.exists(checkpoint):
            return run_name, False, "no checkpoint .pt found"

        # DQNAgent.test() asserts that len(val_file_names) == 1 before loading.
        agent.val_file_names = [checkpoint]
        agent.test(checkpoint=checkpoint)
        return run_name, True, f"tested with {os.path.basename(checkpoint)}"
    except (RuntimeError, OSError, ValueError) as e:
        return Path(run_dir).name if run_dir else "<unknown>", False, str(e)


if __name__ == "__main__":
    main()
