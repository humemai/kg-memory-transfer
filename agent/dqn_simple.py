import os
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from .agent import Agent
from .nn import (
    LSTMSequenceNet,
    TransformerSequenceNet,
)
from .utils import (
    ReplayBuffer,
    plot_results_simple,
    save_final_results_simple,
    set_all_seeds,
    target_hard_update,
    update_epsilon,
    update_model_simple,
    write_yaml,
    select_action_simple_masked,
)


class SimpleDQNAgent(Agent):
    """
    SimpleDQNAgent is a pure neural approach to the room environment problem.
    Unlike DQNAgent which uses neuro-symbolic policies, this agent learns to directly
    map observations to combinatorial actions (QA answer + exploration direction).

    Action space: 49 rooms × 5 explore directions = 245 total actions
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v3",
        env_config: dict[str, Any] | None = None,
        num_samples_for_results: dict | None = None,
        save_results: bool = True,
        default_root_dir: str = "./training-results-simple/",
        num_iterations: int = 10000,
        replay_buffer_size: int = 1000,
        warm_start: int = 32,
        batch_size: int = 32,
        target_update_interval: int = 10,
        epsilon_decay_until: float = 10000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
        architecture_type: str = "lstm",  # "lstm" or "transformer" (sequence)
        transformer_params: dict | None = None,
        lstm_params: dict | None = None,
        mlp_params: dict | None = None,
        validation_interval: int = 5,
        plotting_interval: int = 20,
        seed: int = 0,
        device: str = "cpu",
        ddqn: bool = True,
        use_gradient_clipping: bool = True,
        gradient_clip_value: float = 1.0,
        max_long_term_memory_size: int = 100,
    ) -> None:
        """Initialize the SimpleDQNAgent.

        Args:
            env_str: The name of the environment to use.
            env_config: A dictionary containing the environment configuration.
            num_samples_for_results: A dictionary containing the number of samples for
                validation and testing.
            save_results: Whether to save the results to disk.
            default_root_dir: The root directory to store the results.
            num_iterations: The number of training iterations.
            replay_buffer_size: The size of the replay buffer.
            warm_start: The number of warm start samples before training.
            batch_size: The batch size for training.
            target_update_interval: The interval for updating the target network.
            epsilon_decay_until: The number of iterations until epsilon decays to
                min_epsilon.
            max_epsilon: The maximum value of epsilon for exploration.
            min_epsilon: The minimum value of epsilon for exploration.
            gamma: The discount factor for future rewards.
            learning_rate: The learning rate for the optimizer.
            architecture_type: The type of architecture to use: "lstm" or
                "transformer" (sequence)
            transformer_params: Parameters specific to Transformer architecture
            mlp_params: Parameters for the MLP heads
            validation_interval: The interval for validation during training.
            plotting_interval: The interval for plotting the results.
            seed: The seed for the training environment.
            device: The device to use for training (e.g., "cpu" or "cuda").
            ddqn: Whether to use Double DQN.
            use_gradient_clipping: Whether to use gradient clipping during training.
            gradient_clip_value: The maximum norm for gradient clipping.
        """
        # Resolve mutable defaults
        if env_config is None:
            env_config = {"terminates_at": 99, "room_size": "dev"}
        if num_samples_for_results is None:
            num_samples_for_results = {"val": 1, "test": 1}
        if transformer_params is None:
            transformer_params = {
                "embedding_dim": 64,
                "num_layers": 2,
                "num_heads": 8,
                "dropout": 0.0,
            }
        if lstm_params is None:
            lstm_params = {"embedding_dim": 64, "num_layers": 2}
        if mlp_params is None:
            mlp_params = {"num_hidden_layers": 2, "dueling_dqn": True}

        params_to_save = deepcopy(locals())
        del params_to_save["self"]
        del params_to_save["__class__"]

        super().__init__(**params_to_save)

        # Validate architecture type
        valid_architectures = ["lstm", "transformer"]
        if architecture_type.lower() not in valid_architectures:
            raise ValueError(
                f"architecture_type must be one of {valid_architectures}, got "
                f"{architecture_type}"
            )

        self.architecture_type = architecture_type.lower()
        self.transformer_params = transformer_params
        self.lstm_params = lstm_params
        self.mlp_params = mlp_params

        self.num_iterations = num_iterations
        self.replay_buffer_size = replay_buffer_size
        self.warm_start = warm_start
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.epsilon_decay_until = epsilon_decay_until
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.validation_interval = validation_interval
        self.plotting_interval = plotting_interval
        self.device = device
        self.ddqn = ddqn
        self.use_gradient_clipping = use_gradient_clipping
        self.gradient_clip_value = gradient_clip_value
        self.val_file_names = []
        self.max_long_term_memory_size = max_long_term_memory_size
        # Initialize attributes commonly set later to satisfy linters and clarity
        self.observations = None
        self.memory = []  # sequence of (h,r,t) triples
        self.replay_buffer = None
        self.epsilons = []
        self.training_loss = []
        self.scores = {"train": [], "val": [], "test": None}
        self.iteration_idx = 0

        assert self.batch_size <= self.warm_start <= self.replay_buffer_size

        # Define action spaces
        self.room_names = self.env.unwrapped.room_names  # 49 rooms
        self.explore_actions = ["north", "south", "east", "west", "stay"]  # 5 actions
        self.total_actions = len(self.room_names) * len(
            self.explore_actions
        )  # 245 actions

        # Track observed rooms for action masking
        self.observed_rooms = set()

        # Create action mappings
        self.action_to_pair = {}
        self.pair_to_action = {}
        action_idx = 0
        for room in self.room_names:
            for explore_action in self.explore_actions:
                self.action_to_pair[action_idx] = (room, explore_action)
                self.pair_to_action[(room, explore_action)] = action_idx
                action_idx += 1

        # Build vocabularies for sequence models (simple, no time qualifiers)
        entities = list(
            set(self.env.unwrapped.entities + self.room_names + ["user", "?"])
        )
        relations = list(self.env.unwrapped.relations)

        # Create neural networks for the Simple Agent
        if self.architecture_type == "lstm":
            emb = self.lstm_params.get("embedding_dim", 64)
            layers = self.lstm_params.get("num_layers", 2)
            self.dqn = LSTMSequenceNet(
                entities=entities,
                relations=relations,
                embedding_dim=emb,
                num_layers=layers,
                action_dim=self.total_actions,
                mlp_hidden_layers=self.mlp_params.get("num_hidden_layers", 1),
                device=self.device,
            )
            self.dqn_target = LSTMSequenceNet(
                entities=entities,
                relations=relations,
                embedding_dim=emb,
                num_layers=layers,
                action_dim=self.total_actions,
                mlp_hidden_layers=self.mlp_params.get("num_hidden_layers", 1),
                device=self.device,
            )
        elif self.architecture_type == "transformer":
            emb = self.transformer_params.get("embedding_dim", 64)
            layers = self.transformer_params.get("num_layers", 2)
            heads = self.transformer_params.get("num_heads", 8)
            dropout = self.transformer_params.get("dropout", 0.0)
            self.dqn = TransformerSequenceNet(
                entities=entities,
                relations=relations,
                embedding_dim=emb,
                num_layers=layers,
                num_heads=heads,
                action_dim=self.total_actions,
                mlp_hidden_layers=self.mlp_params.get("num_hidden_layers", 1),
                dropout=dropout,
                device=self.device,
            )
            self.dqn_target = TransformerSequenceNet(
                entities=entities,
                relations=relations,
                embedding_dim=emb,
                num_layers=layers,
                num_heads=heads,
                action_dim=self.total_actions,
                mlp_hidden_layers=self.mlp_params.get("num_hidden_layers", 1),
                dropout=dropout,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture_type}")
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # Disable gradients for target network
        for param in self.dqn_target.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = optim.Adam(list(self.dqn.parameters()), lr=self.learning_rate)

        self.q_values = {
            "train": [],
            "val": [],
            "test": [],
        }

        # Add episode-level storage
        self.episode_q_values = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.episode_actions = {
            "train": [],
            "val": [],
            "test": [],
        }

        self._save_number_of_parameters()
        self.init_memory_systems()

    def _save_number_of_parameters(self) -> None:
        """Save the number of parameters in the model."""
        total_params = sum(p.numel() for p in self.dqn.parameters())
        params_dict = {
            "total": total_params,
            "architecture": self.architecture_type,
        }
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_dict, os.path.join(self.default_root_dir, "num_params.yaml"))

    def init_memory_systems(self) -> None:
        """Initialize the agent's simple memory (list of triples)."""
        self.current_step = 0
        self.memory = []
        self.observed_rooms = set()

    def update_observed_rooms(self) -> None:
        """Update the set of observed rooms based on current observations."""
        for obs in self.observations["room"]:
            # Check if the object (third element) of the triple is a room name
            if len(obs) >= 3 and obs[2] in self.room_names:
                self.observed_rooms.add(obs[2])

    def get_valid_actions(self) -> list[int]:
        """Get list of valid action indices based on observed rooms."""
        valid_actions = []
        for action_idx, (room, _) in self.action_to_pair.items():
            if room in self.observed_rooms:
                valid_actions.append(action_idx)
        return valid_actions

    def encode_all_observations(self) -> None:
        """Append current observations to simple memory and enforce capacity."""
        # Append observed triples
        for obs in self.observations["room"]:
            if len(obs) >= 3:
                self.memory.append([str(obs[0]), str(obs[1]), str(obs[2])])
        # Enforce capacity (fifo)
        if (
            self.max_long_term_memory_size is not None
            and self.max_long_term_memory_size >= 0
        ):
            excess = len(self.memory) - self.max_long_term_memory_size
            if excess > 0:
                self.memory = self.memory[excess:]
        # Update observed rooms
        self.update_observed_rooms()

    def reset(self) -> None:
        """Reset the agent's environment and memory systems."""
        self.init_memory_systems()
        self.observations, _info = self.env.reset()
        self.encode_all_observations()

    def step(self, greedy: bool) -> tuple[int, np.ndarray, int, bool]:
        """Step of the algorithm.

        Args:
            greedy: whether to use greedy policy

        Returns:
            action, q_values, reward, done
        """
        # Get current state
        memory_list = self.memory
        question = self.observations["question"]
        valid_actions = self.get_valid_actions()

        # Select action using neural network with action masking
        action, q_values = select_action_simple_masked(
            state=memory_list,
            question=question,
            valid_actions=valid_actions,
            greedy=greedy,
            dqn=self.dqn,
            epsilon=self.epsilon,
        )

        # Convert action to room answer and explore direction
        room_answer, explore_direction = self.action_to_pair[action.item()]

        # Take step in environment
        (
            self.observations,
            reward,
            done,
            truncated,
            _info,
        ) = self.env.step((room_answer, explore_direction))
        done = done or truncated

        # Encode new observations into simple memory
        self.current_step += 1
        self.encode_all_observations()

        return action, q_values, reward, done

    def fill_replay_buffer(self) -> None:
        """Fill the replay buffer until warm start size."""
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.batch_size)
        done = True

        while len(self.replay_buffer) < self.warm_start:
            if done:
                self.reset()
                done = False
            else:
                state = deepcopy(self.memory)
                question = deepcopy(self.observations["question"])
                action, _q_values, reward, done = self.step(greedy=False)
                next_state = deepcopy(self.memory)

                self.replay_buffer.store(
                    state={"state": state, "question": question},
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

    def train(self) -> None:
        """Train the agent."""
        self.fill_replay_buffer()

        self.epsilons = []
        self.training_loss = []
        self.scores = {"train": [], "val": [], "test": None}

        self.dqn.train()
        done = True
        score = 0
        self.iteration_idx = 0

        # Episode-level storage for current episode
        current_episode_q_values = []
        current_episode_actions = []

        while True:
            if done:
                self.reset()
                done = False
            else:
                state = deepcopy(self.memory)
                question = deepcopy(self.observations["question"])
                action, q_values, reward, done = self.step(greedy=False)
                score += reward
                next_state = deepcopy(self.memory)

                self.replay_buffer.store(
                    state={"state": state, "question": question},
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                self.q_values["train"].append(q_values)
                current_episode_q_values.append(q_values)
                current_episode_actions.append(action)
                self.iteration_idx += 1

            if done:
                self.scores["train"].append(score)
                # Save episode-level data
                self.episode_q_values["train"].append(current_episode_q_values)
                self.episode_actions["train"].append(current_episode_actions)
                # Reset for next episode
                current_episode_q_values = []
                current_episode_actions = []
                score = 0

                if (
                    self.iteration_idx
                    % (
                        self.validation_interval
                        * (self.env_config["terminates_at"] + 1)
                    )
                    == 0
                ):
                    with torch.no_grad():
                        self.validate()

            else:
                loss = update_model_simple(
                    replay_buffer=self.replay_buffer,
                    optimizer=self.optimizer,
                    device=self.device,
                    dqn=self.dqn,
                    dqn_target=self.dqn_target,
                    ddqn=self.ddqn,
                    gamma=self.gamma,
                    use_gradient_clipping=self.use_gradient_clipping,
                    gradient_clip_value=self.gradient_clip_value,
                )

                self.training_loss.append(loss)

                # Linearly decay epsilon
                self.epsilon = update_epsilon(
                    self.epsilon,
                    self.max_epsilon,
                    self.min_epsilon,
                    self.epsilon_decay_until,
                )
                self.epsilons.append(self.epsilon)

                # Update target network
                if self.iteration_idx % self.target_update_interval == 0:
                    target_hard_update(dqn=self.dqn, dqn_target=self.dqn_target)

                # Plotting & show training results
                if (
                    self.iteration_idx == self.num_iterations
                    or self.iteration_idx % self.plotting_interval == 0
                ):
                    self.plot_results(save_fig=True)

                if self.iteration_idx >= self.num_iterations:
                    break

        with torch.no_grad():
            self.test()

        self.env.close()

    def validate_test_middle(self, val_or_test: str) -> tuple[list, list, list, list]:
        """A function shared by validation and test in the middle."""
        scores_local = []
        states_local = []
        q_values_local = []
        actions_local = []

        for idx in range(self.num_samples_for_results[val_or_test]):
            set_all_seeds(self.seed + idx)
            done = True
            score = 0
            episode_q_values = []
            episode_actions = []

            while True:
                if done:
                    self.reset()
                    done = False
                else:
                    state = deepcopy(self.memory)
                    question = deepcopy(self.observations["question"])
                    action, q_values, reward, done = self.step(greedy=True)
                    score += reward

                    episode_q_values.append(q_values)
                    episode_actions.append(action)

                    if idx == self.num_samples_for_results[val_or_test] - 1:
                        states_local.append({"state": state, "question": question})
                        q_values_local.append(q_values)
                        actions_local.append(action)
                        self.q_values[val_or_test].append(q_values)

                if done:
                    break

            scores_local.append(score)
            # Save episode-level data for all samples
            self.episode_q_values[val_or_test].append(episode_q_values)
            self.episode_actions[val_or_test].append(episode_actions)

        return scores_local, states_local, q_values_local, actions_local

    def validate(self) -> None:
        """Validate the agent."""
        self.dqn.eval()

        scores_temp, states, q_values, actions = self.validate_test_middle("val")

        num_episodes = self.iteration_idx // (self.env_config["terminates_at"] + 1) - 1
        mean_score = round(np.mean(scores_temp).item(), 3)

        # Save detailed validation data
        self._save_detailed_episode_data(
            states, q_values, actions, f"val_episode={num_episodes}"
        )

        # Save validation checkpoint
        filename = os.path.join(
            self.default_root_dir,
            f"episode={num_episodes}_val-score={mean_score:.3f}.pt",
        )
        torch.save(self.dqn.state_dict(), filename)
        self.val_file_names.append(filename)

        # Update validation scores
        for _ in range(self.validation_interval):
            self.scores["val"].append(scores_temp)

        # Keep only the best validation checkpoint
        scores_to_compare = []
        for fn in self.val_file_names:
            score = float(fn.split("val-score=")[-1].split(".pt")[0])
            scores_to_compare.append(score)

        from .utils import list_duplicates_of

        indexes = list_duplicates_of(scores_to_compare, max(scores_to_compare))
        file_to_keep = self.val_file_names[indexes[-1]]

        # Remove non-best checkpoints (avoid mutating while iterating)
        for fn in list(self.val_file_names):
            if fn != file_to_keep:
                os.remove(fn)
                self.val_file_names.remove(fn)

        self.env.close()
        self.dqn.train()

    def test(self, checkpoint: str | None = None) -> None:
        """Test the agent."""
        self.dqn.eval()
        self.env = gym.make(self.env_str, **self.env_config)

        # Load checkpoint if provided or if a validation checkpoint exists;
        # otherwise, evaluate current weights
        if checkpoint is not None:
            self.dqn.load_state_dict(torch.load(checkpoint))
        elif len(self.val_file_names) >= 1:
            # Use the latest/best remaining validation checkpoint
            self.dqn.load_state_dict(torch.load(self.val_file_names[0]))

        scores, states, q_values, actions = self.validate_test_middle("test")
        self.scores["test"] = scores

        # Save detailed test data
        self._save_detailed_episode_data(states, q_values, actions, "test")

        save_final_results_simple(
            self.scores,
            self.training_loss,
            self.default_root_dir,
            self.q_values,
            self,
        )

        self.plot_results(save_fig=True)
        self.env.close()
        self.dqn.train()

    def _save_detailed_episode_data(
        self, states: list, q_values: list, actions: list, filename_prefix: str
    ) -> None:
        """Save detailed episode data including states, Q-values, and actions."""
        detailed_data = []

        for state, q_val, action in zip(states, q_values, actions):
            # Convert action from tensor to int and then to room/explore pair
            action_int = action.item() if hasattr(action, "item") else action
            room_answer, explore_direction = self.action_to_pair[action_int]

            # Format Q-values - SimpleDQN has single Q-value output for all actions
            q_values_simple = q_val.tolist() if hasattr(q_val, "tolist") else q_val

            step_data = {
                "state": state["state"],
                "question": state["question"],
                "q_values_simple": q_values_simple,
                "action_simple": action_int,
                "room_answer": room_answer,
                "explore_direction": explore_direction,
            }
            detailed_data.append(step_data)

        # Save to YAML file
        filename = f"states_q_values_actions_{filename_prefix}.yaml"
        filepath = os.path.join(self.default_root_dir, filename)
        write_yaml(detailed_data, filepath)

    def plot_results(self, save_fig: bool = False) -> None:
        """Plot things for Simple DQN training."""
        plot_results_simple(
            self.scores,
            self.training_loss,
            self.epsilons,
            self.q_values,
            self.iteration_idx,
            self.num_iterations,
            self.env.unwrapped.total_maximum_episode_rewards,
            self.default_root_dir,
            save_fig,
        )

    # Required abstract methods from Agent
    def answer_question(self, question: list[str]) -> str:
        """This method is not used in SimpleDQNAgent as actions are chosen directly."""
        raise NotImplementedError("SimpleDQNAgent uses direct neural action selection")

    def explore(self) -> str:
        """This method is not used in SimpleDQNAgent as actions are chosen directly."""
        raise NotImplementedError("SimpleDQNAgent uses direct neural action selection")

    def manage_memory(self) -> None:
        """This method is not used; memory is managed automatically."""
        raise NotImplementedError("SimpleDQNAgent manages memory automatically")
