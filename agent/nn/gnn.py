"""A lot copied from https://github.com/migalkin/StarE"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv

from .attention import AttentionAggregator
from .mlp import MLP
from .stare_conv import StarEConvLayer
from .transformer import TransformerMemoryNet
from .utils import process_graph


class GNN(torch.nn.Module):
    """Graph Neural Network model. This model is used to compute the Q-values for the
    memory management (remember and forget) policies. This model has N layers of
    GCNConv, RGCN, or StarEConv layers and two MLPs for the two memory management
    polices, respectively.

    Now supports StarE, vanilla GCN, RGCN, and Transformer architectures.
    """

    def __init__(
        self,
        entities: list[str],
        relations: list[str],
        architecture_type: str = "stare",  # "stare", "gcn", "rgcn", or "transformer"
        stare_params: dict = {
            "embedding_dim": 64,
            "num_layers": 2,
            "gcn_drop": 0.1,
            "triple_qual_weight": 0.8,
            "silu_between_layers": True,
            "dropout_between_layers": True,
        },
        gcn_params: dict = {
            "embedding_dim": 64,
            "num_layers": 2,
            "gcn_drop": 0.1,
            "silu_between_layers": True,
            "dropout_between_layers": True,
        },
        rgcn_params: dict = {
            "embedding_dim": 64,
            "num_layers": 2,
            "gcn_drop": 0.1,
            "num_bases": 30,
            "silu_between_layers": True,
            "dropout_between_layers": True,
        },
        transformer_params: dict = {
            "embedding_dim": 64,
            "num_layers": 2,
            "dim_feedforward": 256,
            "num_heads": 8,
            "dropout": 0.1,
        },
        mlp_params: dict = {"num_hidden_layers": 2, "dueling_dqn": True},
        device: str = "cpu",
        forget_needs_rl: bool = True,
        remember_needs_rl: bool = True,
        qa_needs_rl: bool = False,
        explore_needs_rl: bool = False,
        separate_network_type: str = None,
        rotational_for_relation: bool = True,
        num_actions: dict = None,
    ) -> None:
        """Initialize the GNN/Transformer model.

        Args:
            entities: List of entities
            relations: List of relations
            architecture_type: Type of architecture: "stare", "gcn", "rgcn", or "transformer"
            stare_params: Parameters for StarE architecture
            gcn_params: Parameters for vanilla GCN architecture
            rgcn_params: Parameters for RGCN architecture
            transformer_params: Parameters for Transformer architecture
            mlp_params: Parameters for the MLP heads
            device: The device to use. Default is "cpu".
            forget_needs_rl: Whether forget policy needs RL components
            remember_needs_rl: Whether remember policy needs RL components
            qa_needs_rl: Whether QA policy needs RL components
            explore_needs_rl: Whether explore policy needs RL components
            separate_network_type: If not None, this network is specialized for one policy only.
                                  Valid values: "forget", "remember", "qa", "explore", or None for shared.
            rotational_for_relation: Whether to use rotational embeddings for relations
            num_actions: Dictionary mapping policy names to number of actions.
        """
        super(GNN, self).__init__()

        # Validate architecture type
        valid_architectures = ["stare", "gcn", "rgcn", "transformer"]
        if architecture_type.lower() not in valid_architectures:
            raise ValueError(
                f"architecture_type must be one of {valid_architectures}, got {architecture_type}"
            )

        # Validate separate_network_type
        valid_separate_types = [
            None,
            "forget",
            "remember",
            "qa",
            "explore",
            "simple",
            "combinatorial",
        ]
        if separate_network_type not in valid_separate_types:
            raise ValueError(
                f"separate_network_type must be one of {valid_separate_types}, got {separate_network_type}"
            )

        self.entities = entities
        self.relations = relations
        self.architecture_type = architecture_type.lower()
        self.stare_params = stare_params
        self.gcn_params = gcn_params
        self.rgcn_params = rgcn_params
        self.transformer_params = transformer_params
        self.mlp_params = mlp_params
        self.device = device
        self.forget_needs_rl = forget_needs_rl
        self.remember_needs_rl = remember_needs_rl
        self.qa_needs_rl = qa_needs_rl
        self.explore_needs_rl = explore_needs_rl
        self.separate_network_type = separate_network_type
        self.rotational_for_relation = rotational_for_relation
        self.num_actions = num_actions

        # If this is a separate network, validate that only the corresponding policy needs RL
        if self.separate_network_type is not None:
            expected_rl_flags = {
                "forget_needs_rl": separate_network_type == "forget",
                "remember_needs_rl": separate_network_type == "remember",
                "qa_needs_rl": separate_network_type == "qa",
                "explore_needs_rl": separate_network_type == "explore",
            }

            actual_rl_flags = {
                "forget_needs_rl": forget_needs_rl,
                "remember_needs_rl": remember_needs_rl,
                "qa_needs_rl": qa_needs_rl,
                "explore_needs_rl": explore_needs_rl,
            }

            if actual_rl_flags != expected_rl_flags:
                raise ValueError(
                    f"For separate_network_type='{separate_network_type}', only "
                    f"{separate_network_type}_needs_rl should be True. "
                    f"Expected: {expected_rl_flags}, Got: {actual_rl_flags}"
                )

        # Get architecture-specific parameters
        if self.architecture_type == "stare":
            arch_params = self.stare_params
            self.gcn_type = "stare"
        elif self.architecture_type == "gcn":
            arch_params = self.gcn_params
            self.gcn_type = "gcn"
        elif self.architecture_type == "rgcn":
            arch_params = self.rgcn_params
            self.gcn_type = "rgcn"
        elif self.architecture_type == "transformer":
            arch_params = self.transformer_params
        else:
            raise ValueError(f"Unsupported architecture type: {self.architecture_type}")

        self.embedding_dim = arch_params["embedding_dim"]

        # If using transformer architecture, create transformer model and return
        if self.architecture_type == "transformer":
            self.transformer_model = TransformerMemoryNet(
                entities=entities,
                relations=relations,
                embedding_dim=self.transformer_params["embedding_dim"],
                dim_feedforward=self.transformer_params["dim_feedforward"],
                num_transformer_layers=self.transformer_params["num_layers"],
                num_heads=self.transformer_params["num_heads"],
                dropout=self.transformer_params["dropout"],
                mlp_params=mlp_params,
                device=device,
                forget_needs_rl=forget_needs_rl,
                remember_needs_rl=remember_needs_rl,
                qa_needs_rl=qa_needs_rl,
                explore_needs_rl=explore_needs_rl,
                separate_network_type=separate_network_type,
                num_actions=num_actions,
            )
            # Move to device
            self.to(self.device)
            return

        # Continue with GNN initialization for StarE/GCN architectures
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relation_to_idx = {
            relation: idx for idx, relation in enumerate(self.relations)
        }

        self.entity_embeddings = torch.nn.Parameter(
            torch.Tensor(len(self.entities), self.embedding_dim)
        ).to(self.device)
        torch.nn.init.xavier_normal_(self.entity_embeddings)

        if self.rotational_for_relation:
            # init relation embeddings with phase values
            phases = (
                2 * np.pi * torch.rand(len(self.relations), self.embedding_dim // 2)
            )
            self.relation_embeddings = torch.nn.Parameter(
                torch.cat(
                    [
                        torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
                        torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1),
                    ],
                    dim=0,
                )
            )
        else:
            self.relation_embeddings = torch.nn.Parameter(
                torch.Tensor(len(self.relations), self.embedding_dim)
            ).to(self.device)
            torch.nn.init.xavier_normal_(self.relation_embeddings)

        self.silu_between_gcn_layers = arch_params.get("silu_between_layers")
        self.dropout_between_gcn_layers = arch_params.get("dropout_between_layers")
        self.silu = torch.nn.SiLU()
        self.drop = torch.nn.Dropout(arch_params.get("gcn_drop"))

        if self.gcn_type == "stare":
            self.gcn_layers = torch.nn.ModuleList(
                [
                    StarEConvLayer(
                        in_channels=self.embedding_dim,
                        out_channels=self.embedding_dim,
                        num_rels=len(relations),
                        gcn_drop=arch_params.get("gcn_drop"),
                        triple_qual_weight=arch_params.get("triple_qual_weight"),
                        device=device,
                    )
                    for _ in range(arch_params["num_layers"])
                ]
            ).to(self.device)

        elif self.gcn_type == "gcn":
            self.gcn_layers = torch.nn.ModuleList(
                [
                    GCNConv(
                        self.embedding_dim,
                        self.embedding_dim,
                        improved=False,
                        add_self_loops=False,
                        normalize=False,
                    )
                    for _ in range(arch_params["num_layers"])
                ]
            ).to(self.device)

            for layer in self.gcn_layers:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

        elif self.gcn_type == "rgcn":
            # RGCN uses relation-specific transformations
            num_relations = len(relations)
            num_bases = arch_params.get("num_bases", min(num_relations, 30))

            self.gcn_layers = torch.nn.ModuleList(
                [
                    RGCNConv(
                        self.embedding_dim,
                        self.embedding_dim,
                        num_relations=num_relations,
                        num_bases=num_bases,
                    )
                    for _ in range(arch_params["num_layers"])
                ]
            ).to(self.device)

            for layer in self.gcn_layers:
                # Initialize RGCN layer parameters
                if hasattr(layer, "weight"):
                    torch.nn.init.xavier_normal_(layer.weight)
                if hasattr(layer, "root"):
                    torch.nn.init.xavier_normal_(layer.root)
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.data.zero_()

        else:
            raise ValueError(f"{self.gcn_type} is not a valid GNN type.")

        # Policy-specific components based on network type
        if separate_network_type == "forget":
            # Only create forget-related components
            self.attention_aggregator_forget = AttentionAggregator(
                embedding_dim=self.embedding_dim,
                device=device,
            )
            self.mlp_forget = MLP(
                n_actions=self.num_actions["forget"],
                input_size=self.embedding_dim,
                hidden_size=self.embedding_dim,
                device=device,
                **mlp_params,
            )
        elif separate_network_type == "remember":
            # Only create remember-related components
            self.mlp_remember = MLP(
                n_actions=self.num_actions["remember"],
                input_size=self.embedding_dim * 2,
                hidden_size=self.embedding_dim,
                device=device,
                **mlp_params,
            )
        elif separate_network_type == "qa":
            # Only create QA-related components with question MLP
            self.attention_aggregator_qa = AttentionAggregator(
                embedding_dim=self.embedding_dim,
                device=device,
            )
            self.question_mlp = MLP(
                n_actions=self.embedding_dim,  # Output dimension
                input_size=self.embedding_dim
                * 3,  # question [subject, relation, object]
                hidden_size=self.embedding_dim,
                device=device,
                num_hidden_layers=1,
                dueling_dqn=False,
            )
            self.mlp_qa = MLP(
                n_actions=self.num_actions["qa"],
                input_size=self.embedding_dim
                + self.embedding_dim,  # memory + compressed question
                hidden_size=self.embedding_dim,
                device=device,
                **mlp_params,
            )
        elif separate_network_type == "explore":
            # Only create explore-related components
            self.attention_aggregator_explore = AttentionAggregator(
                embedding_dim=self.embedding_dim,
                device=device,
            )
            self.mlp_explore = MLP(
                n_actions=self.num_actions["explore"],
                input_size=self.embedding_dim,
                hidden_size=self.embedding_dim,
                device=device,
                **mlp_params,
            )
        elif separate_network_type == "simple":
            # Only create simple DQN components
            self.attention_aggregator_simple = AttentionAggregator(
                embedding_dim=self.embedding_dim,
                device=device,
            )
            self.question_mlp = MLP(
                n_actions=self.embedding_dim,  # Output dimension
                input_size=self.embedding_dim * 3,  # question [s, r, o]
                hidden_size=self.embedding_dim,
                device=device,
                num_hidden_layers=1,
                dueling_dqn=False,
            )
            self.mlp_simple = MLP(
                n_actions=self.num_actions["simple"],
                input_size=self.embedding_dim + self.embedding_dim,  # memory + question
                hidden_size=self.embedding_dim,
                device=device,
                **mlp_params,
            )
        elif separate_network_type == "combinatorial":
            # Only create combinatorial components
            self.attention_aggregator_combinatorial = AttentionAggregator(
                embedding_dim=self.embedding_dim,
                device=device,
            )
            self.mlp_combinatorial = MLP(
                n_actions=self.num_actions["combinatorial"],
                input_size=self.embedding_dim,
                hidden_size=self.embedding_dim,
                device=device,
                **mlp_params,
            )
        elif separate_network_type is None:
            # Shared network: conditionally create components based on needs
            # Check if this is a combinatorial action space
            is_combinatorial = "combinatorial" in self.num_actions
            
            if is_combinatorial:
                # For combinatorial action space, create a single MLP that outputs 27 actions
                self.attention_aggregator_combinatorial = AttentionAggregator(
                    embedding_dim=self.embedding_dim,
                    device=device,
                )
                self.mlp_combinatorial = MLP(
                    n_actions=self.num_actions["combinatorial"],
                    input_size=self.embedding_dim,
                    hidden_size=self.embedding_dim,
                    device=device,
                    **mlp_params,
                )
            else:
                # Independent action spaces - create separate MLPs for each policy
                if forget_needs_rl:
                    self.attention_aggregator_forget = AttentionAggregator(
                        embedding_dim=self.embedding_dim,
                        device=device,
                    )
                    self.mlp_forget = MLP(
                        n_actions=self.num_actions["forget"],
                        input_size=self.embedding_dim,
                        hidden_size=self.embedding_dim,
                        device=device,
                        **mlp_params,
                    )

                if qa_needs_rl:
                    self.attention_aggregator_qa = AttentionAggregator(
                        embedding_dim=self.embedding_dim,
                        device=device,
                    )
                    self.question_mlp = MLP(
                        n_actions=self.embedding_dim,  # Output dimension
                        input_size=self.embedding_dim * 3,  # question [s, r, o]
                        hidden_size=self.embedding_dim,
                        device=device,
                        num_hidden_layers=1,
                        dueling_dqn=False,
                    )
                    self.mlp_qa = MLP(
                        n_actions=self.num_actions["qa"],
                        input_size=self.embedding_dim
                        + self.embedding_dim,  # memory + compressed question
                        hidden_size=self.embedding_dim,
                        device=device,
                        **mlp_params,
                    )

                if explore_needs_rl:
                    self.attention_aggregator_explore = AttentionAggregator(
                        embedding_dim=self.embedding_dim,
                        device=device,
                    )
                    self.mlp_explore = MLP(
                        n_actions=self.num_actions["explore"],
                        input_size=self.embedding_dim,
                        hidden_size=self.embedding_dim,
                        device=device,
                        **mlp_params,
                    )

            # Remember MLP is always created if needed (used by both combinatorial and independent)
            if remember_needs_rl:
                self.mlp_remember = MLP(
                    n_actions=self.num_actions["remember"],
                    input_size=self.embedding_dim * 2,
                    hidden_size=self.embedding_dim,
                    device=device,
                    **mlp_params,
                )
        else:
            raise ValueError(
                f"Invalid separate_network_type: {separate_network_type}. "
                "Must be one of 'forget', 'remember', 'qa', 'explore', or None."
            )

        # Move the entire model to the specified device
        self.to(self.device)

    def process_batch(
        self, data: np.ndarray, filter_to_short_term_only: bool = False
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        r"""Process the data batch.

        Args:
            data: The input data as a batch. This is the same as what the `forward`
                method receives. We will make them in to a batched version of the
                entity embeddings, relation embeddings, edge index, edge type, and
                qualifiers. StarE needs all of them, while vanilla-GCN only needs the
                entity embeddings and edge index.

        Returns:
            edge_idx: The shape is [2, num_quadruples]
            edge_type: The shape is [num_quadruples]
            quals: The shape is [3, number of qualifier key-value pairs]

            edge_idx_inv: The shape is [2, num_quadruples]
            edge_type_inv: The shape is [num_quadruples]
            quals_inv: The shape is [3, number of qualifier key-value pairs]

            short_memory_idx: The shape is [number of short-term memories]
                the idx indexes `edge_idx` and `edge_type`

            num_short_memories: The number of short-term memories in each sample

            num_entities_per_sample: The number of entities in each sample

        """
        entity_embeddings_batch = []
        relation_embeddings_batch = []
        edge_idx_batch = []
        edge_type_batch = []
        quals_batch = []

        entity_embeddings_batch_inv = []
        relation_embeddings_batch_inv = []
        edge_idx_inv_batch = []
        edge_type_inv_batch = []
        quals_inv_batch = []

        short_memory_idx_batch = []

        entity_offset_batch = [0]
        relation_offset_batch = [0]
        edge_offset_batch = [0]

        num_entities_per_sample = []

        for idx, sample in enumerate(data):
            (
                entities,
                relations,
                edge_idx,
                edge_type,
                quals,
                edge_idx_inv,
                edge_type_inv,
                quals_inv,
                short_memory_idx,
            ) = process_graph(
                sample, device=self.device, filter_to_short_term_only=filter_to_short_term_only
            )

            num_entities_per_sample.append(len(entities))

            for entity in entities:
                entity_embeddings_batch.append(
                    self.entity_embeddings[self.entity_to_idx[entity]]
                )
            for relation in relations:
                relation_embeddings_batch.append(
                    self.relation_embeddings[self.relation_to_idx[relation]]
                )

            # For RGCN, we need to remap local relation indices to global ones
            if self.gcn_type == "rgcn":
                # Create a mapping from local to global relation indices
                local_to_global = torch.tensor(
                    [self.relation_to_idx[rel] for rel in relations],
                    dtype=edge_type.dtype,
                    device=self.device,
                )
                edge_type = local_to_global[edge_type]
                edge_type_inv = local_to_global[edge_type_inv]

            edge_idx_batch.append(edge_idx)
            edge_type_batch.append(edge_type)
            quals_batch.append(quals)

            for entity in entities:
                entity_embeddings_batch_inv.append(
                    self.entity_embeddings[self.entity_to_idx[entity]]
                )
            for relation in relations:
                relation_embeddings_batch_inv.append(
                    self.relation_embeddings[self.relation_to_idx[relation]]
                )
            edge_idx_inv_batch.append(edge_idx_inv)
            edge_type_inv_batch.append(edge_type_inv)
            quals_inv_batch.append(quals_inv)

            short_memory_idx_batch.append(short_memory_idx)

            if idx < len(data) - 1:
                entity_offset_batch.append(len(entities) + entity_offset_batch[-1])
                relation_offset_batch.append(len(relations) + relation_offset_batch[-1])
                edge_offset_batch.append(edge_idx.size(1) + edge_offset_batch[-1])

        entity_embeddings_batch = torch.stack(entity_embeddings_batch, dim=0)
        entity_embeddings_batch_inv = torch.stack(entity_embeddings_batch_inv, dim=0)
        entity_embeddings = torch.cat(
            [entity_embeddings_batch, entity_embeddings_batch_inv], dim=0
        )

        relation_embeddings_batch = torch.stack(relation_embeddings_batch, dim=0)
        relation_embeddings_batch_inv = torch.stack(
            relation_embeddings_batch_inv, dim=0
        )
        relation_embeddings = torch.cat(
            [relation_embeddings_batch, relation_embeddings_batch_inv], dim=0
        )

        edge_idx_batch = [a + b for a, b in zip(edge_idx_batch, entity_offset_batch)]
        edge_idx_batch = torch.cat(edge_idx_batch, dim=1).to(self.device)

        edge_idx_batch_inv = torch.cat(
            [
                a + b + entity_embeddings_batch.shape[0]
                for a, b in zip(edge_idx_inv_batch, entity_offset_batch)
            ],
            dim=1,
        ).to(self.device)
        edge_idx = torch.cat([edge_idx_batch, edge_idx_batch_inv], dim=1)

        # For RGCN, edge types are already global indices, no offset needed
        if self.gcn_type == "rgcn":
            edge_type_batch = torch.cat(edge_type_batch, dim=0).to(self.device)
            edge_type_batch_inv = torch.cat(edge_type_inv_batch, dim=0).to(self.device)
        else:
            # For StarE and GCN, apply relation offsets for batching
            edge_type_batch = [
                a + b for a, b in zip(edge_type_batch, relation_offset_batch)
            ]
            edge_type_batch = torch.cat(edge_type_batch, dim=0).to(self.device)

            edge_type_batch_inv = torch.cat(
                [
                    a + b + relation_embeddings_batch.shape[0]
                    for a, b in zip(edge_type_inv_batch, relation_offset_batch)
                ],
                dim=0,
            ).to(self.device)

        edge_type = torch.cat([edge_type_batch, edge_type_batch_inv], dim=0)

        quals_batch = torch.cat(
            [
                a + torch.tensor([c, b, d], device=self.device).reshape(-1, 1)
                for a, b, c, d in zip(
                    quals_batch,
                    entity_offset_batch,
                    relation_offset_batch,
                    edge_offset_batch,
                )
            ],
            dim=1,
        ).to(self.device)
        quals = quals_batch.repeat(1, 2)

        num_short_memories = torch.tensor(
            [len(short_memory_idx) for short_memory_idx in short_memory_idx_batch],
            device=self.device,
        )
        short_memory_idx = torch.cat(
            [a + b for a, b in zip(short_memory_idx_batch, edge_offset_batch)], dim=0
        ).to(self.device)

        return (
            entity_embeddings,
            relation_embeddings,
            edge_idx,
            edge_type,
            quals,
            short_memory_idx,
            num_short_memories,
            torch.tensor(num_entities_per_sample, device=self.device),
        )

    def encode_question(self, question: tuple[str, str, str]) -> torch.Tensor:
        """Encode a question as a compressed embedding.

        Args:
            question: Tuple of (subject, predicate, object) representing the question

        Returns:
            Question embedding: (embedding_dim,)
        """
        subject, predicate, object_placeholder = question

        # Get embeddings for each component
        subject_emb = self.entity_embeddings[self.entity_to_idx[subject]]
        predicate_emb = self.relation_embeddings[self.relation_to_idx[predicate]]
        object_emb = self.entity_embeddings[self.entity_to_idx[object_placeholder]]

        # Concatenate to form question embedding
        question_concat = torch.cat([subject_emb, predicate_emb, object_emb], dim=0)

        # Compress using question MLP
        question_emb = self.question_mlp(question_concat.unsqueeze(0)).squeeze(0)

        return question_emb

    def forward(
        self, data: np.ndarray, policy_type: str, question: tuple[str, str, str] = None
    ) -> list[torch.Tensor]:
        """Forward pass of the GNN/Transformer model.

        Args:
            data: The input data as a batch.
            policy_type: The type of policy to compute Q-values for.
            question: Optional question tuple (subject, predicate, object) for QA policy.

        Returns:
            Q-values for the specified policy:
        """
        # If using transformer, delegate to transformer model
        if self.architecture_type == "transformer":
            return self.transformer_model(data, policy_type, question)

        # Validate policy type based on network configuration
        valid_policies = [
            "remember",
            "remember_global",
            "remember_no_context",
            "remember_global_no_context",
            "forget",
            "qa",
            "explore",
            "simple",
            "combinatorial",
        ]
        if policy_type not in valid_policies:
            raise ValueError(
                f"policy_type must be one of {valid_policies}, got {policy_type}"
            )

        # For separate networks, ensure we're only being asked for the correct policy
        if self.separate_network_type is not None:
            # For remember networks, accept all remember variants
            if self.separate_network_type == "remember":
                if not policy_type.startswith("remember"):
                    raise ValueError(
                        f"This network is specialized for 'remember' policies, "
                        f"but was asked to compute '{policy_type}' policy"
                    )
            elif policy_type != self.separate_network_type:
                raise ValueError(
                    f"This network is specialized for '{self.separate_network_type}' policy only, "
                    f"but was asked to compute '{policy_type}' policy"
                )

        # For shared networks, ensure the requested policy components exist
        if self.separate_network_type is None:
            if policy_type.startswith("remember") and not hasattr(self, "mlp_remember"):
                raise ValueError(
                    "Remember policy components not available in this shared network"
                )
            if policy_type == "forget" and (
                not hasattr(self, "mlp_forget")
                or not hasattr(self, "attention_aggregator_forget")
            ):
                raise ValueError(
                    "Forget policy components (MLP and attention aggregator) not available in this shared network"
                )
            if policy_type == "qa" and (
                not hasattr(self, "mlp_qa")
                or not hasattr(self, "attention_aggregator_qa")
            ):
                raise ValueError(
                    "QA policy components (MLP and attention aggregator) not available in this shared network"
                )
            if policy_type == "explore" and (
                not hasattr(self, "mlp_explore")
                or not hasattr(self, "attention_aggregator_explore")
            ):
                raise ValueError(
                    "Explore policy components (MLP and attention aggregator) not available in this shared network"
                )

        # Determine if we need to filter to short-term only (no context ablation)
        filter_to_short_term = "no_context" in policy_type
        
        (
            entity_embeddings,
            relation_embeddings,
            edge_idx,
            edge_type,
            quals,
            short_memory_idx,
            num_short_memories,
            num_entities_per_sample,
        ) = self.process_batch(data, filter_to_short_term_only=filter_to_short_term)

        for layer_ in self.gcn_layers:
            if "stare" in self.gcn_type:
                entity_embeddings, relation_embeddings = layer_(
                    entity_embeddings=entity_embeddings,
                    relation_embeddings=relation_embeddings,
                    edge_idx=edge_idx,
                    edge_type=edge_type,
                    quals=quals,
                )
            elif self.gcn_type == "rgcn":
                # RGCN requires edge_type for relation-specific transformations
                entity_embeddings = layer_(entity_embeddings, edge_idx, edge_type)
            elif "gcn" in self.gcn_type or "vanilla" in self.gcn_type:
                entity_embeddings = layer_(entity_embeddings, edge_idx)
            else:
                raise ValueError(f"{self.gcn_type} is not a valid GNN type.")

            if self.dropout_between_gcn_layers:
                entity_embeddings = self.drop(entity_embeddings)
            if self.silu_between_gcn_layers:
                entity_embeddings = F.silu(entity_embeddings)

        # Handle remember policy with 2x2 ablation design:
        # - global_pooling: False=per-item, True=mean pooling
        # - no_context: False=full working memory, True=short-term only
        if policy_type in [
            "remember",
            "remember_global",
            "remember_no_context",
            "remember_global_no_context",
        ]:
            
            # Determine flags from policy type
            global_pooling = "global" in policy_type
            # Note: no_context filtering already applied in process_batch
            
            assert num_short_memories.sum() == short_memory_idx.size(0)
            
            # Extract embeddings for short-term memory items
            triple = []
            for idx in short_memory_idx:
                # Concatenate head and tail entity embeddings
                # Entity embeddings contain relation info for StarE/RGCN
                triple_ = torch.cat(
                    [
                        entity_embeddings[edge_idx[0, idx]],
                        entity_embeddings[edge_idx[1, idx]],
                    ],
                    dim=0,
                )
                triple.append(triple_)

            triple = torch.stack(triple, dim=0)  # (total_stm, 2*embed_dim)
            
            if global_pooling:
                # Global pooling: mean over all STM items -> single Q-value
                pooled = triple.mean(dim=0, keepdim=True)  # (1, 2*embed_dim)
                q_global = self.mlp_remember(pooled)  # (1, 2)
                
                # Replicate same Q-value for all items in each batch sample
                q_remember_batch = [
                    q_global.expand(num, -1)
                    for num in num_short_memories
                ]
            else:
                # Per-item: compute independent Q-value for each STM item
                q_remember = self.mlp_remember(triple)  # (total_stm, 2)
                
                # Split into batch format
                q_remember_batch = [
                    q_remember[start : start + num]
                    for start, num in zip(
                        num_short_memories.cumsum(0).roll(1), num_short_memories
                    )
                ]
                q_remember_batch[0] = q_remember[: num_short_memories[0]]

            return q_remember_batch

        # Handle forget policy with attention aggregation
        elif policy_type == "forget":
            # Get the first half of entity embeddings (original, not duplicated)
            entity_embeddings_first_half = entity_embeddings[
                : entity_embeddings.size(0) // 2
            ]

            # Create padded batch for efficient processing
            max_num_entities = max(num_entities_per_sample)
            batch_size = len(num_entities_per_sample)

            # Initialize padded tensor
            padded_embeddings = torch.zeros(
                batch_size, max_num_entities, self.embedding_dim, device=self.device
            )

            # Create attention mask
            attention_mask = torch.zeros(
                batch_size, max_num_entities, dtype=torch.bool, device=self.device
            )

            # Fill padded tensor and mask
            start_idx = 0
            for i, num_entities in enumerate(num_entities_per_sample):
                padded_embeddings[i, :num_entities] = entity_embeddings_first_half[
                    start_idx:start_idx + num_entities
                ]
                attention_mask[i, :num_entities] = True
                start_idx += num_entities

            # Check if combinatorial action space
            if hasattr(self, "mlp_combinatorial"):
                # Single batched attention aggregation
                aggregated_embeddings = self.attention_aggregator_combinatorial(
                    padded_embeddings, attention_mask
                )  # (batch_size, embedding_dim)
                # Single batched MLP forward pass
                q_forget = self.mlp_combinatorial(aggregated_embeddings)  # (batch_size, 27)
            else:
                # Single batched attention aggregation
                aggregated_embeddings = self.attention_aggregator_forget(
                    padded_embeddings, attention_mask
                )  # (batch_size, embedding_dim)
                # Single batched MLP forward pass
                q_forget = self.mlp_forget(aggregated_embeddings)  # (batch_size, 3)

            # Convert back to list format for consistency
            q_forget_batch = [q_forget[i:i + 1] for i in range(batch_size)]

            return q_forget_batch

        # Handle QA policy with question conditioning
        elif policy_type == "qa":
            # Get the first half of entity embeddings (original, not duplicated)
            entity_embeddings_first_half = entity_embeddings[
                : entity_embeddings.size(0) // 2
            ]

            # Create padded batch for efficient processing
            max_num_entities = max(num_entities_per_sample)
            batch_size = len(num_entities_per_sample)

            # Initialize padded tensor
            padded_embeddings = torch.zeros(
                batch_size, max_num_entities, self.embedding_dim, device=self.device
            )

            # Create attention mask
            attention_mask = torch.zeros(
                batch_size, max_num_entities, dtype=torch.bool, device=self.device
            )

            # Fill padded tensor and mask
            start_idx = 0
            for i, num_entities in enumerate(num_entities_per_sample):
                padded_embeddings[i, :num_entities] = entity_embeddings_first_half[
                    start_idx:start_idx + num_entities
                ]
                attention_mask[i, :num_entities] = True
                start_idx += num_entities

            # Check if combinatorial action space
            if hasattr(self, "mlp_combinatorial"):
                # Single batched attention aggregation
                aggregated_embeddings = self.attention_aggregator_combinatorial(
                    padded_embeddings, attention_mask
                )  # (batch_size, embedding_dim)
                # Single batched MLP forward pass (question ignored for combinatorial)
                q_qa = self.mlp_combinatorial(aggregated_embeddings)  # (batch_size, 27)
            else:
                # Encode the question
                question_emb = self.encode_question(question)  # (embedding_dim,)

                # Single batched attention aggregation
                aggregated_embeddings = self.attention_aggregator_qa(
                    padded_embeddings, attention_mask
                )  # (batch_size, embedding_dim)

                # Expand question embedding for batch
                question_emb_batch = question_emb.unsqueeze(0).expand(
                    batch_size, -1
                )  # (batch_size, embedding_dim)

                # Concatenate memory and question embeddings
                combined_input = torch.cat(
                    [aggregated_embeddings, question_emb_batch], dim=1
                )

                # Single batched MLP forward pass
                q_qa = self.mlp_qa(combined_input)  # (batch_size, 3)

            # Convert back to list format for consistency
            q_qa_batch = [q_qa[i:i + 1] for i in range(batch_size)]

            return q_qa_batch

        # Handle explore policy with attention aggregation
        elif policy_type == "explore":
            # Get the first half of entity embeddings (original, not duplicated)
            entity_embeddings_first_half = entity_embeddings[
                : entity_embeddings.size(0) // 2
            ]

            # Create padded batch for efficient processing
            max_num_entities = max(num_entities_per_sample)
            batch_size = len(num_entities_per_sample)

            # Initialize padded tensor
            padded_embeddings = torch.zeros(
                batch_size, max_num_entities, self.embedding_dim, device=self.device
            )

            # Create attention mask
            attention_mask = torch.zeros(
                batch_size, max_num_entities, dtype=torch.bool, device=self.device
            )

            # Fill padded tensor and mask
            start_idx = 0
            for i, num_entities in enumerate(num_entities_per_sample):
                padded_embeddings[i, :num_entities] = entity_embeddings_first_half[
                    start_idx:start_idx + num_entities
                ]
                attention_mask[i, :num_entities] = True
                start_idx += num_entities

            # Check if combinatorial action space
            if hasattr(self, "mlp_combinatorial"):
                # Single batched attention aggregation
                aggregated_embeddings = self.attention_aggregator_combinatorial(
                    padded_embeddings, attention_mask
                )  # (batch_size, embedding_dim)
                # Single batched MLP forward pass
                q_explore = self.mlp_combinatorial(aggregated_embeddings)  # (batch_size, 27)
            else:
                # Single batched attention aggregation
                aggregated_embeddings = self.attention_aggregator_explore(
                    padded_embeddings, attention_mask
                )  # (batch_size, embedding_dim)
                # Single batched MLP forward pass
                q_explore = self.mlp_explore(aggregated_embeddings)  # (batch_size, 3)

            # Convert back to list format for consistency
            q_explore_batch = [q_explore[i:i + 1] for i in range(batch_size)]

            return q_explore_batch

        # Handle simple policy with question conditioning
        elif policy_type == "simple":
            # Encode the question
            question_emb = self.encode_question(question)  # (embedding_dim,)

            # Get the first half of entity embeddings (original, not duplicated)
            entity_embeddings_first_half = entity_embeddings[
                : entity_embeddings.size(0) // 2
            ]

            # Create padded batch for efficient processing
            max_num_entities = max(num_entities_per_sample)
            batch_size = len(num_entities_per_sample)

            # Initialize padded tensor
            padded_embeddings = torch.zeros(
                batch_size, max_num_entities, self.embedding_dim, device=self.device
            )

            # Create attention mask
            attention_mask = torch.zeros(
                batch_size, max_num_entities, dtype=torch.bool, device=self.device
            )

            # Fill padded tensor and mask
            start_idx = 0
            for i, num_entities in enumerate(num_entities_per_sample):
                padded_embeddings[i, :num_entities] = entity_embeddings_first_half[
                    start_idx : start_idx + num_entities
                ]
                attention_mask[i, :num_entities] = True
                start_idx += num_entities

            # Single batched attention aggregation
            aggregated_embeddings = self.attention_aggregator_simple(
                padded_embeddings, attention_mask
            )  # (batch_size, embedding_dim)

            # Expand question embedding for batch
            question_emb_batch = question_emb.unsqueeze(0).expand(
                batch_size, -1
            )  # (batch_size, embedding_dim)

            # Concatenate memory and question embeddings
            combined_input = torch.cat(
                [aggregated_embeddings, question_emb_batch], dim=1
            )

            # Single batched MLP forward pass
            q_simple = self.mlp_simple(combined_input)  # (batch_size, n_actions)

            # Convert back to list format for consistency
            q_simple_batch = [q_simple[i : i + 1] for i in range(batch_size)]

            return q_simple_batch

        # Handle combinatorial policy
        elif policy_type == "combinatorial":
            # Get the first half of entity embeddings (original, not duplicated)
            entity_embeddings_first_half = entity_embeddings[
                : entity_embeddings.size(0) // 2
            ]

            # Create padded batch for efficient processing
            max_num_entities = max(num_entities_per_sample)
            batch_size = len(num_entities_per_sample)

            # Initialize padded tensor
            padded_embeddings = torch.zeros(
                batch_size, max_num_entities, self.embedding_dim, device=self.device
            )

            # Create attention mask
            attention_mask = torch.zeros(
                batch_size, max_num_entities, dtype=torch.bool, device=self.device
            )

            # Fill padded tensor and mask
            start_idx = 0
            for i, num_entities in enumerate(num_entities_per_sample):
                padded_embeddings[i, :num_entities] = entity_embeddings_first_half[
                    start_idx : start_idx + num_entities
                ]
                attention_mask[i, :num_entities] = True
                start_idx += num_entities

            if not hasattr(self, "mlp_combinatorial"):
                raise ValueError("Combinatorial policy components not available")

            # Single batched attention aggregation
            aggregated_embeddings = self.attention_aggregator_combinatorial(
                padded_embeddings, attention_mask
            )  # (batch_size, embedding_dim)

            # Single batched MLP forward pass
            q_combinatorial = self.mlp_combinatorial(
                aggregated_embeddings
            )  # (batch_size, 27)

            # Convert back to list format for consistency
            q_combinatorial_batch = [
                q_combinatorial[i : i + 1] for i in range(batch_size)
            ]

            return q_combinatorial_batch

        else:
            raise ValueError(
                f"{policy_type} is not a valid policy type. Use 'remember', 'forget', 'qa', 'explore', or 'simple'."
            )
