"""
This module defines the base Agent class for interacting with the RoomEnv environment.

The Agent class provides a foundation for implementing intelligent agents that can
explore a room, answer questions about the environment, and manage their memory.
It includes methods for creating a training directory, saving results, running test
episodes, and generating action pairs.
"""

import os
import shutil
from copy import deepcopy
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np

from .utils import set_all_seeds, write_yaml


class Agent:
    """
    Base class for agents interacting with the RoomEnv environment.

    This class provides a foundation for implementing intelligent agents that can
    explore a room, answer questions about the environment, and manage their memory.
    It includes methods for creating a training directory, saving results, running test
    episodes, and generating action pairs.
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v3",
        env_config: dict[str, Any] = {
            "terminates_at": 99,
            "room_size": "dev",
        },
        num_samples_for_results: int = 1,
        default_root_dir: str = "./training-results/",
        save_results: bool = True,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Agent with the given environment and configuration.

        Args:
            env_str: The name of the environment to use.
            env_config: A dictionary containing the environment configuration.
            num_samples_for_results: The number of test episodes to run.
            default_root_dir: The root directory to store the results.
            save_results: Whether to save the results to disk.
            seed: Random seed for reproducibility.
            **kwargs: Additional keyword arguments to be added to the configuration.
        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]

        # Unpack kwargs into the top-level dictionary
        kwargs_dict = params_to_save.pop("kwargs", {})
        params_to_save.update(kwargs_dict)

        self.env_str = env_str
        self.env_config = env_config
        self.num_samples_for_results = num_samples_for_results
        self.save_results = save_results
        self.seed = seed
        self.default_root_dir = os.path.join(default_root_dir, str(datetime.now()))

        self.env: gym.Env = gym.make(self.env_str, **self.env_config)

        if self.save_results:
            self._create_directory(params_to_save)

    def _create_directory(self, params_to_save: dict[str, Any]) -> None:
        """Create the directory to store the results.

        Args:
            params_to_save: Dictionary of parameters to save in the train.yaml file.

        Returns:
            None
        """
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def remove_results_from_disk(self) -> None:
        """Remove the results directory from disk.

        Deletes the entire results directory and all its contents.

        Args:
            None

        Returns:
            None
        """
        shutil.rmtree(self.default_root_dir)

    def answer_question(self, question: list[str]) -> str:
        """Answers a question about the environment.

        Args:
            question: A list containing the question components.

        Returns:
            str: The answer to the question.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def explore(self) -> str:
        """Explores the environment.

        Returns:
            str: The exploration action to take.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def manage_memory(self) -> None:
        """Manages the agent's memory.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def test(self) -> dict[str, Any]:
        """Test the agent on multiple episodes and collect performance metrics.

        Runs the agent through multiple test episodes and tracks scores and episode
        lengths. Calculates statistics and saves results to disk.

        Args:
            None

        Returns:
            dict: Results dictionary containing test score and episode length
                statistics.
        """
        self.scores: list[float] = []

        for episode in range(self.num_samples_for_results):
            set_all_seeds(self.seed + episode)
            score, steps = self._run_test_episode()
            self.scores.append(score)
            print(
                f"Episode {episode+1}/{self.num_samples_for_results} "
                f"completed. Score: {score:.3f}"
            )

        # Calculate and save comprehensive metrics
        results: dict[str, Any] = {
            "test_score": {
                "mean": round(float(np.mean(self.scores)), 3),
                "std": round(float(np.std(self.scores)), 3),
                "min": round(float(np.min(self.scores)), 3),
                "max": round(float(np.max(self.scores)), 3),
            },
            "num_episodes": len(self.scores),
        }

        num_main_triples = self.get_num_main_triples()

        if num_main_triples is not None:
            results["num_main_triples"] = num_main_triples

        print(f"Results: {results}")

        if self.save_results:
            write_yaml(results, os.path.join(self.default_root_dir, "results.yaml"))
        return results

    def _generate_action_pair(self, observations: dict[str, Any]) -> tuple[str, str]:
        """Generate action pair from observations.

        Creates an action pair consisting of an answer to the question and an exploration
        action.

        Args:
            observations: Dictionary containing environment observations, including
                'questions' and information about the room.

        Returns:
            tuple: (answer, explore_action) - Answer string and exploration
                direction.
        """
        answer: str = self.answer_question(observations["question"])
        explore_action: str = self.explore()
        return (answer, explore_action)

    def _run_test_episode(self) -> tuple[float, int]:
        """Runs a single test episode.

        Returns:
            tuple[float, int]: The score and number of steps for the episode.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def get_num_main_triples(self) -> int:
        """Get the number of main triples in the agent's long-term memory.

        Returns:
            int: The number of main triples.

        """
        return None
