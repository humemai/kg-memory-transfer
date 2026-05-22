"""Sequence-based function approximators for the Simple Neural Agent.

Tokenization:
- Map entities and relations to embedding tables.
- For each memory entry (head, rel, tail), we form a token by
    concatenating embeddings of head, rel, and tail, then apply a linear
    projection that reduces (3*E) -> E.
- The question is embedded from (subject, predicate) via a small projection to
    the same E dimension and used to compute attention pooling over the sequence.

Encoders:
- LSTMSequenceNet: BiLSTM (configurable layers), attention-pool with question.
- TransformerSequenceNet: TransformerEncoder with sinusoidal positions,
    attention-pool with question.

Notes on Transformer FFN sizing:
- Inside the Transformer, d_model is the dimensionality of each token vector
    that the encoder sees. In this design we first concatenate the three
    embeddings (head, relation, tail) to size 3E and then PROJECT them back to E
    via a linear layer before the encoder. Sinusoidal positional encodings are
    ADDED (not concatenated), so d_model == embedding_dim (E).
- With this standard setup, using dim_feedforward ≈ 4 × d_model (i.e., 4 × E)
    is appropriate and matches common practice. If you instead fed the
    concatenated 3E tokens directly into the Transformer (no projection), you
    would set d_model = 3E and typically dim_feedforward around 4 × 3E.
    Values between 2× and 8× can be reasonable depending on compute and model
    size; 4× is a widely used default.

Both encoders share the same QHead MLP producing action_dim outputs.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import math
import numpy as np
import torch
from torch import nn

# no functional used


def _default_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class SimpleSeqTokenizer(nn.Module):
    """Tokenize memory triples + question to fixed-dim vectors.

    Inputs for a single memory item are expected as a list of length 3
    """

    def __init__(
        self,
        entities: list[str],
        relations: list[str],
        embedding_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = _default_device(device)
        self.embedding_dim = embedding_dim

        # Build vocabularies
        self.entity2idx = {str(e): i for i, e in enumerate(sorted(set(entities)))}
        self.relation2idx = {str(r): i for i, r in enumerate(sorted(set(relations)))}
        self.entity_emb = nn.Embedding(len(self.entity2idx), embedding_dim)
        self.relation_emb = nn.Embedding(len(self.relation2idx), embedding_dim)
        # Project concatenated triple (head, rel, tail) to E
        self.token_proj = nn.Linear(embedding_dim * 3, embedding_dim)

        # Question projection: (subj, pred) -> E
        self.q_proj = nn.Linear(embedding_dim * 2, embedding_dim)

        self._init_parameters()

        self.to(self.device)

    def _init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _e_idx(self, s: Any) -> int:
        key = str(s)
        return self.entity2idx.get(key, 0)

    def _r_idx(self, s: Any) -> int:
        key = str(s)
        return self.relation2idx.get(key, 0)

    def tokenize_state(self, memory_state: Iterable) -> torch.Tensor:
        """Convert a memory list into a [seq_len, E] tensor on device.

        If sequence is empty, return one zero token so the model remains stable.
        """
        tokens: list[torch.Tensor] = []
        for item in memory_state:
            # Robustly skip empty/malformed items; handle numpy arrays
            if item is None:
                continue
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if not isinstance(item, (list, tuple)):
                continue
            if len(item) < 3:
                continue
            h, r, t = item[:3]

            h_idx = torch.tensor(self._e_idx(h), device=self.device)
            r_idx = torch.tensor(self._r_idx(r), device=self.device)
            t_idx = torch.tensor(self._e_idx(t), device=self.device)

            h_e = self.entity_emb(h_idx)
            r_e = self.relation_emb(r_idx)
            t_e = self.entity_emb(t_idx)

            feat = torch.cat([h_e, r_e, t_e], dim=-1)
            tok = self.token_proj(feat)
            tokens.append(tok)

        if not tokens:
            # Return one zero token
            return torch.zeros(1, self.embedding_dim, device=self.device)
        return torch.stack(tokens, dim=0)

    def embed_question(self, question: Tuple[str, str, Any] | None) -> torch.Tensor:
        """Embed (subject, predicate) to an E-dim vector.

        If question is None, return a learned-neutral vector (zeros) so that
        the downstream code stays robust.
        """
        if not question:
            return torch.zeros(self.embedding_dim, device=self.device)

        subj, pred, *_ = question
        s_idx = torch.tensor(self._e_idx(subj), device=self.device)
        p_idx = torch.tensor(self._r_idx(pred), device=self.device)
        s_e = self.entity_emb(s_idx)
        p_e = self.relation_emb(p_idx)
        q = torch.cat([s_e, p_e], dim=-1)
        return self.q_proj(q)

    # Provide a forward to satisfy Module interface; treats input as one state
    def forward(self, memory_state: Iterable) -> torch.Tensor:  # type: ignore[override]
        return self.tokenize_state(memory_state)


class QHead(nn.Module):
    """Small MLP head to map pooled representation to action Q-values."""

    def __init__(self, input_dim: int, action_dim: int, hidden_layers: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(dim, dim), nn.SiLU()]
        layers += [nn.Linear(dim, action_dim)]
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DotAttentionPool(nn.Module):
    """Single-vector query attention pooling over a token sequence.

    Given sequence X=[L,E] and query q=[E], returns context c=[E].
    """

    def forward(self, seq: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2:
            raise ValueError("seq must be [L,E]")
        if query.ndim != 1:
            raise ValueError("query must be [E]")
        scores = torch.matmul(seq, query)  # [L]
        weights = torch.softmax(scores, dim=0)
        return torch.sum(weights.unsqueeze(-1) * seq, dim=0)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [L, E]
        L = x.size(0)
        return x + self.pe[:L]


class LSTMSequenceNet(nn.Module):
    """Unidirectional LSTM encoder with question-conditioned attention,
    for Simple DQN.
    """

    def __init__(
        self,
        entities: list[str],
        relations: list[str],
        embedding_dim: int,
        num_layers: int,
        action_dim: int,
        mlp_hidden_layers: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = _default_device(device)
        self.embedding_dim = embedding_dim
        self.tokenizer = SimpleSeqTokenizer(
            entities, relations, embedding_dim, self.device
        )

        hidden = embedding_dim
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.pool = DotAttentionPool()
        self.q_head = QHead(
            input_dim=hidden, action_dim=action_dim, hidden_layers=mlp_hidden_layers
        )
        self.to(self.device)

    def forward(
        self,
        states: np.ndarray | list,
        policy_type: str = "simple",
        question: tuple[str, str, Any] | None = None,
    ) -> list[torch.Tensor]:
        assert (
            policy_type == "simple"
        ), "LSTMSequenceNet supports only policy_type='simple'"
        # We expect states to be a numpy array of dtype=object with each element
        # a state list
        batch_list = states if isinstance(states, (list, tuple)) else list(states)
        outputs: list[torch.Tensor] = []
        for state in batch_list:
            tokens = self.tokenizer.tokenize_state(state)  # [L,E]
            q_vec = self.tokenizer.embed_question(question)  # [E]

            # LSTM expects [N,L,E] when batch_first=True; use N=1
            seq, _ = self.encoder(tokens.unsqueeze(0))  # [1,L,H]
            seq = seq.squeeze(0)  # [L,H]
            pooled = self.pool(seq, q_vec)  # [H]
            q_values = self.q_head(pooled).unsqueeze(0)  # [1, A]
            outputs.append(q_values)
        return outputs


class TransformerSequenceNet(nn.Module):
    """Transformer encoder with question-conditioned attention for Simple DQN."""

    def __init__(
        self,
        entities: list[str],
        relations: list[str],
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        action_dim: int,
        mlp_hidden_layers: int = 1,
        dropout: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = _default_device(device)
        self.embedding_dim = embedding_dim
        self.tokenizer = SimpleSeqTokenizer(
            entities, relations, embedding_dim, self.device
        )
        self.posenc = PositionalEncoding(embedding_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,  # Typically 4x embedding_dim
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = DotAttentionPool()
        self.q_head = QHead(
            input_dim=embedding_dim,
            action_dim=action_dim,
            hidden_layers=mlp_hidden_layers,
        )
        self.to(self.device)

    def forward(
        self,
        states: np.ndarray | list,
        policy_type: str = "simple",
        question: tuple[str, str, Any] | None = None,
    ) -> list[torch.Tensor]:
        assert (
            policy_type == "simple"
        ), "TransformerSequenceNet supports only policy_type='simple'"
        batch_list = states if isinstance(states, (list, tuple)) else list(states)
        outputs: list[torch.Tensor] = []
        for state in batch_list:
            tokens = self.tokenizer.tokenize_state(state)  # [L,E]
            q_vec = self.tokenizer.embed_question(question)  # [E]

            x = self.posenc(tokens)  # [L,E]
            # Transformer expects [N,L,E] when batch_first=True; use N=1
            x = x.unsqueeze(0)
            x = self.encoder(x)  # [1,L,E]
            x = x.squeeze(0)  # [L,E]
            pooled = self.pool(x, q_vec)  # [E]
            q_values = self.q_head(pooled).unsqueeze(0)  # [1,A]
            outputs.append(q_values)
        return outputs
