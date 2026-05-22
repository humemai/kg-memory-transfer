"""This module defines the LongTermAgent class which uses both short-term and
long-term memory to store and retrieve observations, along with QA, exploration, and
memory management (forget and remember) policies.
"""

from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Optional

from humemai_research.rdflib import Humemai
from rdflib import XSD, Literal, Namespace, URIRef

from .agent import Agent


class LongTermAgent(Agent):
    """
    An agent that manages both short-term and long-term memories via Humemai.
    Supports QA, exploration, and memory management (forget and remember) policies.
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v3",
        env_config: dict[str, Any] = {
            "terminates_at": 99,
            "room_size": "dev",
        },
        qa_policy: str = "mra",
        explore_policy: str = "mra",
        forget_policy: str = "lru",
        remember_policy: str = "all",
        max_long_term_memory_size: int = 100,
        num_samples_for_results: int = 1,
        save_results: bool = True,
        default_root_dir: str = "./training-results/",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a LongTermAgent with environment configuration, QA policy,
        exploration policy, and memory management parameters.
        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]
        del params_to_save["__class__"]
        super().__init__(**params_to_save)

        assert qa_policy.lower() in [
            "mra",
            "mru",
            "mfu",
            "rl",
            "random_meta",
            "memory_pressure",
            "intersection_vote",
            "borda",
            "rl_combinatorial",
        ], f"Invalid QA policy: {qa_policy}"
        self.qa_policy = qa_policy.lower()

        assert explore_policy.lower() in [
            "mra",
            "mru",
            "mfu",
            "rl",
            "random_meta",
            "memory_pressure",
            "intersection_vote",
            "borda",
            "rl_combinatorial",
        ], f"Invalid explore policy: {explore_policy}"
        self.explore_policy = explore_policy.lower()

        assert forget_policy.lower() in [
            "fifo",
            "lru",
            "lfu",
            "random",
            "rl",
            "random_meta",
            "memory_pressure",
            "intersection_vote",
            "borda",
            "rl_combinatorial",  # Combinatorial action space (3x3x3=27 actions)
        ], f"Invalid long-term memory management policy: {forget_policy}"
        self.forget_policy = forget_policy.lower()

        # Enforce consistency for rl_combinatorial: all three policies must match
        if self.forget_policy == "rl_combinatorial":
            assert self.qa_policy == "rl_combinatorial", (
                f"rl_combinatorial requires all policies to be rl_combinatorial. "
                f"Got forget_policy={self.forget_policy}, qa_policy={self.qa_policy}"
            )
            assert self.explore_policy == "rl_combinatorial", (
                f"rl_combinatorial requires all policies to be rl_combinatorial. "
                f"Got forget_policy={self.forget_policy}, explore_policy={self.explore_policy}"
            )
        if self.qa_policy == "rl_combinatorial":
            assert self.forget_policy == "rl_combinatorial", (
                f"rl_combinatorial requires all policies to be rl_combinatorial. "
                f"Got qa_policy={self.qa_policy}, forget_policy={self.forget_policy}"
            )
            assert self.explore_policy == "rl_combinatorial", (
                f"rl_combinatorial requires all policies to be rl_combinatorial. "
                f"Got qa_policy={self.qa_policy}, explore_policy={self.explore_policy}"
            )
        if self.explore_policy == "rl_combinatorial":
            assert self.forget_policy == "rl_combinatorial", (
                f"rl_combinatorial requires all policies to be rl_combinatorial. "
                f"Got explore_policy={self.explore_policy}, forget_policy={self.forget_policy}"
            )
            assert self.qa_policy == "rl_combinatorial", (
                f"rl_combinatorial requires all policies to be rl_combinatorial. "
                f"Got explore_policy={self.explore_policy}, qa_policy={self.qa_policy}"
            )

        # Enforce consistency for symbolic baseline policies
        symbolic_baselines = ["random_meta", "memory_pressure", "intersection_vote", "borda"]
        if self.forget_policy in symbolic_baselines:
            assert self.qa_policy == self.forget_policy, (
                f"Symbolic baseline '{self.forget_policy}' requires matching QA policy. "
                f"Got forget_policy={self.forget_policy}, qa_policy={self.qa_policy}"
            )
            assert self.explore_policy == self.forget_policy, (
                f"Symbolic baseline '{self.forget_policy}' requires matching explore policy. "
                f"Got forget_policy={self.forget_policy}, explore_policy={self.explore_policy}"
            )
        if self.qa_policy in symbolic_baselines:
            assert self.forget_policy == self.qa_policy, (
                f"Symbolic baseline '{self.qa_policy}' requires matching forget policy. "
                f"Got qa_policy={self.qa_policy}, forget_policy={self.forget_policy}"
            )
            assert self.explore_policy == self.qa_policy, (
                f"Symbolic baseline '{self.qa_policy}' requires matching explore policy. "
                f"Got qa_policy={self.qa_policy}, explore_policy={self.explore_policy}"
            )
        if self.explore_policy in symbolic_baselines:
            assert self.forget_policy == self.explore_policy, (
                f"Symbolic baseline '{self.explore_policy}' requires matching forget policy. "
                f"Got explore_policy={self.explore_policy}, forget_policy={self.forget_policy}"
            )
            assert self.qa_policy == self.explore_policy, (
                f"Symbolic baseline '{self.explore_policy}' requires matching QA policy. "
                f"Got explore_policy={self.explore_policy}, qa_policy={self.qa_policy}"
            )

        # Validate remember policy (allow random_X.X format)
        valid_policies = [
            "all",
            "novel",
            "rl",  # Per-item Q-values with full context (default)
            "rl_global",  # Global pooling with full context
            "rl_no_context",  # Per-item Q-values, short-term only
            "rl_global_no_context",  # Global pooling, short-term only
        ]
        is_random = remember_policy.lower().startswith("random_")
        if not is_random and remember_policy.lower() not in valid_policies:
            raise ValueError(
                f"Invalid long-term memory remember policy: {remember_policy}"
            )
        self.remember_policy = remember_policy.lower()

        self.max_long_term_memory_size = max_long_term_memory_size

        # Track current_step to sync with environment steps
        self.current_step = 0

        # HumemAI setup
        self.humemai_ns = Namespace("https://humem.ai/ontology#")
        self.base_date = datetime.fromisoformat("2025-01-01T00:00:00")
        self.humemai = Humemai()

    # -------------------------------------------------------------------------
    # Memory Management
    # -------------------------------------------------------------------------

    def manage_memory(self, observations: list[list[str]], step: int) -> None:
        """
        Manage memory by:
          1) Move short-term memories to episodic (based on remember policy).
          2) Enforce memory management until total meets the size limit.
          3) Clear short-term memory.
          4) Add new short-term observations.
        """
        if self.remember_policy.lower() == "all":
            self.humemai.move_all_short_term_to_episodic()
        elif self.remember_policy.lower() == "novel":
            self._remember_novel()
        elif self.remember_policy.lower().startswith("random_"):
            # Extract probability from policy name (e.g., "random_0.5")
            try:
                prob = float(self.remember_policy.split("_")[1])
                self._remember_random(prob)
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Invalid random policy format: {self.remember_policy}. "
                    f"Expected format: random_X.X (e.g., random_0.5)"
                ) from e
        else:
            # RL-based remembering will be handled in DQNAgent
            raise NotImplementedError(
                f"Memory management remember policy '{self.remember_policy}' not implemented."
            )

        # assert short-term is empty
        assert (
            self.humemai.get_short_term_memory_count() == 0
        ), "Short-term memory should be empty after moving to episodic."

        # 2) While we exceed memory limits, prune one statement at a time
        while (
            self.humemai.get_long_term_memory_count() > self.max_long_term_memory_size
        ):
            if self.forget_policy.lower() == "fifo":
                mem_id_to_delete = self._pick_fifo_victim()
            elif self.forget_policy.lower() == "lru":
                mem_id_to_delete = self._pick_lru_victim()
            elif self.forget_policy.lower() == "lfu":
                mem_id_to_delete = self._pick_lfu_victim()
            elif self.forget_policy.lower() == "random":
                mem_id_to_delete = self._pick_random_victim()
            elif self.forget_policy.lower() == "random_meta":
                mem_id_to_delete = self._pick_random_meta_victim()
            elif self.forget_policy.lower() == "memory_pressure":
                mem_id_to_delete = self._pick_memory_pressure_victim()
            elif self.forget_policy.lower() == "intersection_vote":
                mem_id_to_delete = self._pick_intersection_vote_victim()
            elif self.forget_policy.lower() == "borda":
                mem_id_to_delete = self._pick_borda_victim()
            else:
                raise NotImplementedError(
                    f"Memory management forget policy '{self.forget_policy}' not implemented."
                )

            if mem_id_to_delete is None:
                raise ValueError("No memory ID found for deletion.")
            self.humemai.delete_memory(Literal(mem_id_to_delete))

        # 3) Insert new observations in short-term
        self.current_time = self.base_date + timedelta(days=step)
        triples = [[URIRef(item) for item in obs] for obs in observations]
        qualifiers = {
            self.humemai_ns.current_time: Literal(
                self.current_time.isoformat(timespec="seconds"), datatype=XSD.dateTime
            )
        }
        self.humemai.add_short_term_memory(triples=triples, qualifiers=qualifiers)

    # ------------------------ fifo ------------------------

    def _pick_fifo_victim(self) -> Optional[int]:
        """Earliest humemai:time_added => evict (fifo)."""
        timestamps = self.find_unique_timestamps()
        if not timestamps:
            raise ValueError("No timestamps => cannot pick fifo victim.")
        oldest = min(timestamps)
        mem_ids = self.find_memory_ids_by_time_added(oldest)
        if not mem_ids:
            raise ValueError(
                "No memory found at earliest timestamp => fifo victim error."
            )
        return random.choice(mem_ids)

    def find_unique_timestamps(self) -> list[str]:
        """All distinct humemai:time_added in LTM."""
        query = """
            PREFIX humemai: <https://humem.ai/ontology#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT DISTINCT ?ta
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    humemai:time_added ?ta .
            }
        """
        results = self.humemai.graph.query(query)
        return [str(row.ta) for row in results]

    def find_memory_ids_by_time_added(self, t: str) -> list[int]:
        """MemIDs with time_added == t."""
        query = f"""
            PREFIX humemai: <https://humem.ai/ontology#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?memory_id
            WHERE {{
              ?stmt rdf:type rdf:Statement ;
                    humemai:time_added "{t}"^^xsd:dateTime ;
                    humemai:memory_id ?memory_id .
            }}
        """
        rows = list(self.humemai.graph.query(query))
        return [int(r.memory_id) for r in rows]

    # ------------------------ lru ------------------------

    def _pick_lru_victim(self) -> Optional[int]:
        """
        Select victim using lru policy:
        1. Find memories with oldest last_accessed value
        2. If multiple, fall back to lfu (lowest num_recalled)
        3. If still multiple, fall back to fifo (oldest time_added)
        4. If still multiple, choose one uniformly at random
        """
        query = """
            PREFIX humemai: <https://humem.ai/ontology#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?memory_id ?la ?rec ?ta
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    humemai:memory_id ?memory_id ;
                    humemai:last_accessed ?la ;
                    humemai:num_recalled ?rec ;
                    humemai:time_added ?ta .
            }
        """
        rows = list(self.humemai.graph.query(query))
        if not rows:
            raise ValueError("No statements => can't pick lru victim.")

        # Convert to list of tuples: (memory_id, last_accessed, num_recalled, time_added)
        candidates = [
            (int(row.memory_id), str(row.la), int(row.rec), str(row.ta)) for row in rows
        ]

        # Step 1: Filter by oldest last_accessed (lru)
        candidates.sort(key=lambda x: x[1])  # Sort by last_accessed
        oldest_la = candidates[0][1]
        lru_candidates = [c for c in candidates if c[1] == oldest_la]

        # If only one lru candidate, we're done
        if len(lru_candidates) == 1:
            return lru_candidates[0][0]  # Return memory ID

        # Step 2: If multiple lru candidates with same last_accessed, fall back to lfu
        # Sort by num_recalled
        lru_candidates.sort(key=lambda x: x[2])
        min_recall = lru_candidates[0][2]
        lfu_winners = [c for c in lru_candidates if c[2] == min_recall]

        # If only one lfu winner, we're done
        if len(lfu_winners) == 1:
            return lfu_winners[0][0]  # Return memory ID

        # Step 3: If still tied, fall back to fifo
        lfu_winners.sort(key=lambda x: x[3])  # Sort by time_added
        oldest_ta = lfu_winners[0][3]
        fifo_winners = [c for c in lfu_winners if c[3] == oldest_ta]

        # Step 4: If still multiple candidates, choose one uniformly at random
        return random.choice(fifo_winners)[0]  # Return memory ID of a random candidate

    # ------------------------ lfu ------------------------

    def _pick_lfu_victim(self) -> Optional[int]:
        """
        Select victim using lfu policy:
        1. Find memories with lowest num_recalled value
        2. If multiple, fall back to lru (oldest last_accessed)
        3. If still multiple, fall back to fifo (oldest time_added)
        4. If still multiple, choose one uniformly at random
        """
        query = """
            PREFIX humemai: <https://humem.ai/ontology#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?memory_id ?rec ?la ?ta
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    humemai:memory_id ?memory_id ;
                    humemai:num_recalled ?rec ;
                    humemai:last_accessed ?la ;
                    humemai:time_added ?ta .
            }
        """
        rows = list(self.humemai.graph.query(query))
        if not rows:
            raise ValueError("No statements => can't pick lfu victim.")

        # Convert to list of tuples: (memory_id, num_recalled, last_accessed, time_added)
        candidates = [
            (int(row.memory_id), int(row.rec), str(row.la), str(row.ta)) for row in rows
        ]

        # Step 1: Filter by lowest num_recalled (lfu)
        min_recall = min(c[1] for c in candidates)
        lfu_candidates = [c for c in candidates if c[1] == min_recall]

        # If only one lfu candidate, we're done
        if len(lfu_candidates) == 1:
            return lfu_candidates[0][0]  # Return memory ID

        # Step 2: If multiple lfu candidates with same recall count, fall back to lru
        # Sort by oldest last_accessed
        lfu_candidates.sort(key=lambda x: x[2])
        oldest_la = lfu_candidates[0][2]
        lru_winners = [c for c in lfu_candidates if c[2] == oldest_la]

        # If only one lru winner, we're done
        if len(lru_winners) == 1:
            return lru_winners[0][0]  # Return memory ID

        # Step 3: If still tied, fall back to fifo
        lru_winners.sort(key=lambda x: x[3])  # Sort by time_added
        oldest_ta = lru_winners[0][3]
        fifo_winners = [c for c in lru_winners if c[3] == oldest_ta]

        # Step 4: If still multiple candidates, choose one uniformly at random
        return random.choice(fifo_winners)[0]  # Return memory ID of a random candidate

    # ------------------------ random ------------------------

    def _pick_random_victim(self) -> Optional[int]:
        """Randomly evict anything in LTM."""
        query = """
            PREFIX humemai: <https://humem.ai/ontology#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?memory_id
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    humemai:memory_id ?memory_id ;
                    humemai:time_added ?ta .
            }
        """
        rows = list(self.humemai.graph.query(query))
        if not rows:
            raise ValueError("No statements => can't pick random victim.")
        return int(random.choice(rows).memory_id)

    # ------------------------ random_meta ------------------------

    def _pick_random_meta_victim(self) -> Optional[int]:
        """Random meta-policy: randomly select FIFO, LRU, or LFU at each call."""
        heuristic = random.choice(["fifo", "lru", "lfu"])
        if heuristic == "fifo":
            return self._pick_fifo_victim()
        elif heuristic == "lru":
            return self._pick_lru_victim()
        else:  # lfu
            return self._pick_lfu_victim()

    # ------------------------ memory_pressure ------------------------

    def _pick_memory_pressure_victim(self) -> Optional[int]:
        """Memory-pressure switching: FIFO if <50% full, LFU if >=50% full."""
        current_size = self.humemai.get_long_term_memory_count()
        threshold = 0.5 * self.max_long_term_memory_size
        
        if current_size < threshold:
            return self._pick_fifo_victim()
        else:
            return self._pick_lfu_victim()

    # ------------------------ intersection_vote ------------------------

    def _pick_intersection_vote_victim(self) -> Optional[int]:
        """Intersection voting: evict triple selected by >=2 heuristics."""
        fifo_victim = self._pick_fifo_victim()
        lru_victim = self._pick_lru_victim()
        lfu_victim = self._pick_lfu_victim()
        
        victims = [fifo_victim, lru_victim, lfu_victim]
        
        # Check for >=2 agreement
        for victim in set(victims):
            if victims.count(victim) >= 2:
                return victim
        
        # No agreement: fall back to LFU (most conservative)
        return lfu_victim

    # ------------------------ borda ------------------------

    def _pick_borda_victim(self) -> Optional[int]:
        """Rank aggregation (Borda count): sum ranks from all heuristics."""
        # Get all memory IDs with their qualifiers
        query = """
            PREFIX humemai: <https://humem.ai/ontology#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?memory_id ?time_added ?last_accessed ?num_recalled
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    humemai:memory_id ?memory_id ;
                    humemai:time_added ?time_added ;
                    humemai:last_accessed ?last_accessed ;
                    humemai:num_recalled ?num_recalled .
            }
        """
        rows = list(self.humemai.graph.query(query))
        if not rows:
            return None
        
        # Convert to list for ranking
        memories = [
            {
                "id": int(row.memory_id),
                "time_added": str(row.time_added),
                "last_accessed": str(row.last_accessed),
                "num_recalled": int(row.num_recalled),
            }
            for row in rows
        ]
        
        # Rank by each criterion (lower rank = more likely to evict)
        # FIFO: rank by time_added (ascending)
        sorted_by_fifo = sorted(memories, key=lambda x: x["time_added"])
        fifo_ranks = {m["id"]: idx for idx, m in enumerate(sorted_by_fifo)}
        
        # LRU: rank by last_accessed (ascending)
        sorted_by_lru = sorted(memories, key=lambda x: x["last_accessed"])
        lru_ranks = {m["id"]: idx for idx, m in enumerate(sorted_by_lru)}
        
        # LFU: rank by num_recalled (ascending)
        sorted_by_lfu = sorted(memories, key=lambda x: x["num_recalled"])
        lfu_ranks = {m["id"]: idx for idx, m in enumerate(sorted_by_lfu)}
        
        # Sum ranks for each memory
        total_ranks = {}
        for memory in memories:
            mem_id = memory["id"]
            total_ranks[mem_id] = (
                fifo_ranks[mem_id] + lru_ranks[mem_id] + lfu_ranks[mem_id]
            )
        
        # Return memory with lowest total rank (most agreed-upon victim)
        victim_id = min(total_ranks, key=total_ranks.get)
        return victim_id

    # -------------------------------------------------------------------------
    # Remember Policies
    # -------------------------------------------------------------------------

    def _remember_novel(self) -> None:
        """Remember only triples that are not already in long-term memory."""
        short_term_list = self.humemai.to_list()
        short_term = [mem for mem in short_term_list if "current_time" in mem[-1].keys()]
        
        # Get all existing long-term memory triples
        existing_triples = set()
        query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX humemai: <https://humem.ai/ontology#>
            SELECT ?s ?p ?o
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    rdf:subject ?s ;
                    rdf:predicate ?p ;
                    rdf:object ?o .
              FILTER NOT EXISTS { ?stmt humemai:current_time ?ct }
            }
        """
        for row in self.humemai.graph.query(query):
            existing_triples.add((str(row.s), str(row.p), str(row.o)))
        
        for memory in short_term:
            triple = memory[:-1]
            qualifiers = memory[-1]
            
            # Convert triple to comparable format
            triple_key = (str(triple[0]), str(triple[1]), str(triple[2]))
            
            # Only remember if this triple doesn't already exist
            if triple_key not in existing_triples:
                memory_id = Literal(qualifiers["memory_id"], datatype=XSD.integer)
                self.humemai.move_short_term_to_episodic(memory_id)
        
        # Clear any remaining short-term memories
        self.humemai.clear_short_term_memories()

    def _remember_random(self, probability: float) -> None:
        """Remember each triple with given probability.

        Args:
            probability: Probability of remembering each triple (0.0 to 1.0)
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Probability must be between 0.0 and 1.0, got {probability}"
            )

        short_term_list = self.humemai.to_list()
        short_term = [
            mem for mem in short_term_list if "current_time" in mem[-1].keys()
        ]

        for memory in short_term:
            qualifiers = memory[-1]

            # Remember with probability p
            if random.random() < probability:
                memory_id = Literal(qualifiers["memory_id"], datatype=XSD.integer)
                self.humemai.move_short_term_to_episodic(memory_id)

        # Clear any remaining short-term memories
        self.humemai.clear_short_term_memories()

    # -------------------------------------------------------------------------
    # QA Policy
    # -------------------------------------------------------------------------

    def answer_question(self, question: tuple[str, str]) -> str:
        """
        Answer question using self.qa_policy, then increment recall for that statement.

        Priority order:
        1. Always prioritize memories with current_time if available (without updating
           memory access)
        2. Otherwise apply specific policy:
           - mra: Choose memory with latest time_added
           - mru: Choose memory with latest last_accessed,
             breaking ties with mra
           - mfu: Choose memory with highest num_recalled,
             breaking ties with mra
           - random: Choose randomly among the other three policies
        """
        subj_str = f"<{question[0]}>"
        pred_str = f"<{question[1]}>"
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX humemai: <https://humem.ai/ontology#>
            SELECT ?stmt ?obj ?ta ?la ?rec ?ct
            WHERE {{
              ?stmt rdf:type rdf:Statement ;
                    rdf:subject {subj_str} ;
                    rdf:predicate {pred_str} ;
                    rdf:object ?obj .
              OPTIONAL {{ ?stmt humemai:current_time ?ct. }}
              OPTIONAL {{ ?stmt humemai:time_added ?ta. }}
              OPTIONAL {{ ?stmt humemai:last_accessed ?la. }}
              OPTIONAL {{ ?stmt humemai:num_recalled ?rec. }}
            }}
        """
        rows = list(self.humemai.graph.query(query))
        if not rows:
            return None

        # Step 1: Always prioritize memories with current_time (short-term memory)
        # Return directly without incrementing num_recalled or updating last_accessed
        current_time_rows = [r for r in rows if r.ct is not None]
        if current_time_rows:
            best_row = current_time_rows[0]  # Take any memory from current time
            return str(best_row.obj)

        # Step 2: Process long-term memories (with time_added property)
        valid_long_term_rows = [r for r in rows if r.ta is not None]
        if not valid_long_term_rows:
            return None  # No valid long-term memories found

        # For long-term memories, verify they have all required properties
        for row in valid_long_term_rows:
            assert row.la is not None, "last_accessed must exist for long-term memory"
            assert row.rec is not None, "num_recalled must exist for long-term memory"

        # Apply the selected policy
        return self._apply_qa_policy(self.qa_policy, valid_long_term_rows, question)

    def _apply_qa_policy(
        self, policy: str, rows: list, question: tuple[str, str]
    ) -> str:
        """Apply a specific QA policy to select the best row and update memory stats."""
        if policy.lower() == "mra":
            best_row, time_added = self._get_mra(rows)
        elif policy.lower() == "mru":
            best_row, time_added = self._get_mru(rows)
        elif policy.lower() == "mfu":
            best_row, time_added = self._get_mfu(rows)
        elif policy.lower() == "random_meta":
            best_row, time_added = self._get_random_meta_qa(rows)
        elif policy.lower() == "memory_pressure":
            best_row, time_added = self._get_memory_pressure_qa(rows)
        elif policy.lower() == "intersection_vote":
            best_row, time_added = self._get_intersection_vote_qa(rows)
        elif policy.lower() == "borda":
            best_row, time_added = self._get_borda_qa(rows)
        else:
            raise ValueError(f"Unknown QA policy: {policy}")

        obj_uri = URIRef(str(best_row.obj))

        # Update memory access stats for long-term memory
        self._update_qa_memory_access(
            subject=URIRef(question[0]),
            predicate=URIRef(question[1]),
            object_=obj_uri,
            time_added=time_added,
        )
        return str(best_row.obj)

    def _get_mra(self, rows: list) -> tuple:
        """Get row with latest time_added, breaking ties randomly."""
        # Sort by time_added, newest first
        rows.sort(key=lambda x: x.ta, reverse=True)
        newest_ta = rows[0].ta
        newest_rows = [r for r in rows if r.ta == newest_ta]

        # If multiple with same time_added, pick one randomly
        best_row = random.choice(newest_rows)
        time_added = best_row.ta
        return best_row, time_added

    def _get_mru(self, rows: list) -> tuple:
        """Get row with latest last_accessed, breaking ties with mra."""
        # Sort by last_accessed, newest first
        rows.sort(key=lambda x: x.la, reverse=True)
        newest_la = rows[0].la
        newest_rows = [r for r in rows if r.la == newest_la]

        if len(newest_rows) > 1:
            # Multiple rows with same last_accessed, break ties with mra
            best_row, _ = self._get_mra(newest_rows)
        else:
            best_row = newest_rows[0]

        time_added = best_row.ta
        return best_row, time_added

    def _get_mfu(self, rows: list) -> tuple:
        """Get row with highest num_recalled, breaking ties with mra."""
        # Sort by num_recalled, highest first
        rows.sort(key=lambda x: int(x.rec.toPython()), reverse=True)
        highest_rec = int(rows[0].rec.toPython())
        highest_rows = [r for r in rows if int(r.rec.toPython()) == highest_rec]

        if len(highest_rows) > 1:
            # Multiple rows with same num_recalled, break ties with mra
            best_row, _ = self._get_mra(highest_rows)
        else:
            best_row = highest_rows[0]

        # For updating memory access, use time_added
        time_literal = best_row.ta
        return best_row, time_literal

    def _get_random_meta_qa(self, rows: list) -> tuple:
        """Random meta-policy: randomly select MRA, MRU, or MFU."""
        heuristic = random.choice(["mra", "mru", "mfu"])
        if heuristic == "mra":
            return self._get_mra(rows)
        elif heuristic == "mru":
            return self._get_mru(rows)
        else:  # mfu
            return self._get_mfu(rows)

    def _get_memory_pressure_qa(self, rows: list) -> tuple:
        """Memory-pressure switching: MRA if <50% full, MFU if >=50% full."""
        current_size = self.humemai.get_long_term_memory_count()
        threshold = 0.5 * self.max_long_term_memory_size
        
        if current_size < threshold:
            return self._get_mra(rows)
        else:
            return self._get_mfu(rows)

    def _get_intersection_vote_qa(self, rows: list) -> tuple:
        """Intersection voting: select answer chosen by >=2 heuristics."""
        mra_row, mra_ta = self._get_mra(rows)
        mru_row, mru_ta = self._get_mru(rows)
        mfu_row, mfu_ta = self._get_mfu(rows)
        
        # Detect if rows have 'obj' (QA) or 'roomY' (explore)
        attr = 'obj' if hasattr(mra_row, 'obj') else 'roomY'
        
        # Get answers from each
        answers = [
            (str(getattr(mra_row, attr)), mra_row, mra_ta),
            (str(getattr(mru_row, attr)), mru_row, mru_ta),
            (str(getattr(mfu_row, attr)), mfu_row, mfu_ta),
        ]
        
        # Check for >=2 agreement
        answer_objs = [a[0] for a in answers]
        for answer, row, ta in answers:
            if answer_objs.count(answer) >= 2:
                return row, ta
        
        # No agreement: fall back to MRA
        return mra_row, mra_ta

    def _get_borda_qa(self, rows: list) -> tuple:
        """Rank aggregation (Borda count): sum ranks from all heuristics."""
        # Detect if rows have 'obj' (QA) or 'roomY' (explore)
        attr = 'obj' if hasattr(rows[0], 'obj') else 'roomY'
        
        # Rank by each criterion (higher rank = better for QA)
        # MRA: rank by time_added (descending)
        sorted_by_mra = sorted(rows, key=lambda x: x.ta, reverse=True)
        mra_ranks = {str(getattr(r, attr)): idx for idx, r in enumerate(sorted_by_mra)}
        
        # MRU: rank by last_accessed (descending)
        sorted_by_mru = sorted(rows, key=lambda x: x.la, reverse=True)
        mru_ranks = {str(getattr(r, attr)): idx for idx, r in enumerate(sorted_by_mru)}
        
        # MFU: rank by num_recalled (descending)
        sorted_by_mfu = sorted(
            rows, key=lambda x: int(x.rec.toPython()), reverse=True
        )
        mfu_ranks = {str(getattr(r, attr)): idx for idx, r in enumerate(sorted_by_mfu)}
        
        # Sum ranks for each answer (lower is better)
        total_ranks = {}
        row_map = {str(getattr(r, attr)): r for r in rows}
        for row in rows:
            answer = str(getattr(row, attr))
            total_ranks[answer] = (
                mra_ranks[answer] + mru_ranks[answer] + mfu_ranks[answer]
            )
        
        # Return answer with lowest total rank (highest consensus)
        best_answer = min(total_ranks, key=total_ranks.get)
        best_row = row_map[best_answer]
        return best_row, best_row.ta

    def _update_qa_memory_access(
        self,
        subject: URIRef,
        predicate: URIRef,
        object_: URIRef,
        time_added: Literal,
    ) -> None:
        """Increment num_recalled + update last_accessed for one triple in LTM."""
        self.humemai.increment_num_recalled(
            subject=subject,
            predicate=predicate,
            object_=object_,
            lower_time_added_bound=time_added,
            upper_time_added_bound=time_added,
        )
        new_time_lit = Literal(
            self.current_time.isoformat(timespec="seconds"), datatype=XSD.dateTime
        )
        self.humemai.update_last_accessed(
            subject=subject,
            predicate=predicate,
            object_=object_,
            new_time=new_time_lit,
            lower_time_added_bound=time_added,
            upper_time_added_bound=time_added,
        )

    def _filter_memories_by_policy(self, rows: list, policy: str) -> tuple:
        """Apply memory filtering policy to select the best row from candidates.

        Args:
            rows: List of memory rows with ta, la, rec attributes
            policy: One of "mra", "mru", "mfu", or new baseline policies

        Returns:
            Tuple of (best_row, time_added) or (None, None) if no rows
        """
        if not rows:
            return None, None

        if policy.lower() == "mra":
            return self._get_mra(rows)
        elif policy.lower() == "mru":
            return self._get_mru(rows)
        elif policy.lower() == "mfu":
            return self._get_mfu(rows)
        elif policy.lower() == "random_meta":
            return self._get_random_meta_qa(rows)
        elif policy.lower() == "memory_pressure":
            return self._get_memory_pressure_qa(rows)
        elif policy.lower() == "intersection_vote":
            return self._get_intersection_vote_qa(rows)
        elif policy.lower() == "borda":
            return self._get_borda_qa(rows)
        else:
            raise ValueError(f"Unknown memory filtering policy: {policy}")

    # -------------------------------------------------------------------------
    # Exploration
    # -------------------------------------------------------------------------

    def explore(self) -> str:
        """Explore with the specified self.explore_policy.

        Returns:
            str: Direction to explore: "north", "south", "east", "west", or "stay".
        """
        valid_policies = [
            "mra", "mru", "mfu", 
            "random_meta", "memory_pressure", "intersection_vote", "borda"
        ]
        if self.explore_policy.lower() in valid_policies:
            return self._explore_bfs_with_memory_filtering()
        else:
            raise ValueError(f"Invalid explore_policy: {self.explore_policy}")

    # -------------------------------------------------------------------------
    # BFS with Memory Filtering
    # -------------------------------------------------------------------------

    def _explore_bfs_with_memory_filtering(self) -> str:
        """BFS-based exploration using memory filtering to resolve conflicts."""
        current_room = self._get_agent_current_room()
        visited_rooms = self._get_visited_rooms()
        adjacency = self._build_room_adjacency_with_memory_filtering()

        path = self._bfs_find_new_room_with_path(current_room, adjacency, visited_rooms)
        if path:
            self._update_path_memory_access(path)
            return path[0][1]  # direction in the first edge

        # else fallback: try least-visited, then 2nd least, ... excluding current room
        non_current_count = len([r for r in visited_rooms if r != current_room])
        for rank in range(1, non_current_count + 1):
            candidate_room = self._pick_least_visited_room(
                visited_rooms, current_room, rank=rank
            )
            if not candidate_room:
                continue
            path2 = self._bfs_path_to_specific_room(
                current_room, candidate_room, adjacency
            )
            if path2:
                self._update_path_memory_access(path2)
                return path2[0][1]

        return random.choice(["north", "south", "east", "west", "stay"])

    def _build_room_adjacency_with_memory_filtering(
        self,
    ) -> dict[str, list[tuple[str, str]]]:
        """Build adjacency preferring short-term memory; fall back to LTM with policy.

        Rules:
        - If any short-term edge exists for (roomX, direction) (i.e., has current_time),
          pick among only those short-term candidates (latest current_time if >1).
        - Otherwise, use long-term edges and resolve conflicts via explore_policy.
        - Exclude edges pointing to a wall.
        """
        adjacency = {}

        # Query all room connections with both ST (current_time) and LTM metadata
        # (optional)
        query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX humemai: <https://humem.ai/ontology#>
            SELECT ?roomX ?dir ?roomY ?ct ?ta ?la ?rec
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    rdf:subject ?roomX ;
                    rdf:predicate ?dir ;
                    rdf:object ?roomY .
              OPTIONAL { ?stmt humemai:current_time ?ct. }
              OPTIONAL { ?stmt humemai:time_added ?ta. }
              OPTIONAL { ?stmt humemai:last_accessed ?la. }
              OPTIONAL { ?stmt humemai:num_recalled ?rec. }
              FILTER(?dir IN (<north>, <south>, <east>, <west>))
            }
        """
        rows = list(self.humemai.graph.query(query))

        # Group by (roomX, direction)
        connections: dict[tuple[str, str], list] = {}
        for r in rows:
            rx = str(r.roomX)
            d = r.dir.rsplit("/", 1)[-1]
            key = (rx, d)
            connections.setdefault(key, []).append(r)

        # Resolve conflicts per (roomX, direction)
        for (room_x, direction), candidate_rows in connections.items():
            # Prefer short-term rows (those having current_time)
            st_rows = [r for r in candidate_rows if getattr(r, "ct", None) is not None]
            chosen_row = None

            if st_rows:
                # If multiple ST candidates, pick the latest current_time
                st_rows.sort(key=lambda x: x.ct, reverse=True)
                chosen_row = st_rows[0]
            else:
                # Only LTM rows available; ensure they have time_added
                ltm_rows = [
                    r for r in candidate_rows if getattr(r, "ta", None) is not None
                ]
                if not ltm_rows:
                    continue
                chosen_row, _ = self._filter_memories_by_policy(
                    ltm_rows, self.explore_policy
                )

            if chosen_row is not None:
                ry = str(chosen_row.roomY)
                if not ry.endswith("wall"):
                    adjacency.setdefault(room_x, []).append((direction, ry))

        return adjacency

    def _bfs_find_new_room_with_path(
        self,
        start_room: str,
        adjacency: dict[str, list[tuple[str, str]]],
        visited_rooms: set[str],
    ) -> Optional[list[tuple[str, str, str]]]:
        """Find BFS path to first unvisited room."""
        queue = deque([start_room])
        parents = {start_room: None}
        while queue:
            current = queue.popleft()
            if current not in visited_rooms and current != start_room:
                return self._bfs_reconstruct_path(parents, start_room, current)
            for direction, nbr in adjacency.get(current, []):
                if nbr not in parents:
                    parents[nbr] = (current, direction)
                    queue.append(nbr)
        return None

    def _bfs_path_to_specific_room(
        self,
        start: str,
        goal: str,
        adjacency: dict[str, list[tuple[str, str]]],
    ) -> Optional[list[tuple[str, str, str]]]:
        """BFS from start to goal."""
        if start == goal:
            return []
        queue = deque([start])
        parents = {start: None}
        while queue:
            cur = queue.popleft()
            if cur == goal:
                return self._bfs_reconstruct_path(parents, start, goal)
            for d, nbr in adjacency.get(cur, []):
                if nbr not in parents:
                    parents[nbr] = (cur, d)
                    queue.append(nbr)
        return None

    def _bfs_reconstruct_path(
        self, parents: dict[str, Optional[tuple[str, str]]], start: str, goal: str
    ) -> list[tuple[str, str, str]]:
        """Reverse BFS parents to get edges: (roomX, direction, roomY)."""
        edges = []
        cur = goal
        while cur != start and cur in parents:
            link = parents[cur]
            if not link:
                break
            prev_room, direction = link
            edges.append((prev_room, direction, cur))
            cur = prev_room
        edges.reverse()
        return edges

    # -------------------------------------------------------------------------
    # BFS Helpers
    # -------------------------------------------------------------------------

    def _get_agent_current_room(self) -> str:
        """Agent's current room from short-term memory."""
        query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX humemai: <https://humem.ai/ontology#>
            SELECT ?room
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    rdf:subject <agent> ;
                    rdf:predicate <at_location> ;
                    rdf:object ?room ;
                    humemai:current_time ?time .
            }
            LIMIT 1
        """
        rows = list(self.humemai.graph.query(query))
        if not rows:
            raise ValueError("No short-term record of the agent's current room!")
        return str(rows[0].room)

    def _get_visited_rooms(self) -> set[str]:
        """
        Rooms in LTM: (agent, at_location, room, time_added=...).
        """
        visited = set()
        query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX humemai: <https://humem.ai/ontology#>
            SELECT ?room
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    rdf:subject <agent> ;
                    rdf:predicate <at_location> ;
                    rdf:object ?room ;
                    humemai:time_added ?ta .
            }
        """
        rows = list(self.humemai.graph.query(query))
        for r in rows:
            visited.add(str(r.room))
        return visited

    # -------------------------------------------------------------------------
    # "Least Visited" Helper
    # -------------------------------------------------------------------------

    def _pick_least_visited_room(
        self, visited_rooms: set[str], current_room: str, rank: int = 1
    ) -> Optional[str]:
        """Pick the Nth least-visited room among visited_rooms, excluding current.

        Args:
            visited_rooms: Set of rooms the agent has been in (from LTM).
            current_room: The agent's current room; always excluded.
            rank: 1 for least-visited, 2 for second least, etc.

        Returns:
            The room URI string for the requested rank, or None if unavailable.
        """
        if not visited_rooms:
            return None
        room_visits = self._get_room_visit_counts()
        candidates = [r for r in visited_rooms if r != current_room]
        if not candidates:
            return None
        # Sort by visit count asc; stable tie-breaker by room name for determinism
        candidates.sort(key=lambda r: (room_visits.get(r, 0), r))
        index = rank - 1
        if 0 <= index < len(candidates):
            return candidates[index]
        return None

    def _get_room_visit_counts(self) -> dict[str, int]:
        """Return number of times the agent was at a room in time_added-based LTM."""
        query = """
            PREFIX humemai: <https://humem.ai/ontology#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?room (COUNT(?ta) as ?vcount)
            WHERE {
              ?stmt rdf:type rdf:Statement ;
                    rdf:subject <agent> ;
                    rdf:predicate <at_location> ;
                    rdf:object ?room ;
                    humemai:time_added ?ta .
            }
            GROUP BY ?room
        """
        rows = list(self.humemai.graph.query(query))
        visits = {}
        for r in rows:
            room_uri = str(r.room)
            visits[room_uri] = int(r.vcount.toPython())
        return visits

    # -------------------------------------------------------------------------
    # BFS "Memory Access" increments
    # -------------------------------------------------------------------------

    def _update_path_memory_access(self, edges: list[tuple[str, str, str]]) -> None:
        """Increment recall + update last_accessed for each edge in the path."""
        for subj, pred, obj in edges:
            self._update_exploration_edge_memory(subj, pred, obj)

    def _update_exploration_edge_memory(
        self, subject: str, predicate: str, object_: str
    ) -> None:
        """
        Find the *latest* statement in LTM for (subject, predicate, object_).
        If it's only short-term, skip.
        """
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX humemai: <https://humem.ai/ontology#>
            SELECT ?stmt ?ct ?ta (COALESCE(?ct, ?ta) AS ?time_value)
            WHERE {{
              ?stmt rdf:type rdf:Statement ;
                    rdf:subject <{subject}> ;
                    rdf:predicate <{predicate}> ;
                    rdf:object <{object_}> .
              OPTIONAL {{ ?stmt humemai:current_time ?ct. }}
              OPTIONAL {{ ?stmt humemai:time_added ?ta. }}
            }}
            ORDER BY DESC(?time_value)
            LIMIT 1
        """

        rows = list(self.humemai.graph.query(query))
        if not rows:
            return

        row = rows[0]
        time_val = row.time_value
        ta_val = row.ta

        # If purely short-term => skip updates (we don't touch ST counters here)
        if ta_val is None:
            return
        if time_val is None:
            raise ValueError(
                f"No valid time found for ({subject}, {predicate}, {object_})."
            )

        # Update recall count and last_accessed for the chosen LTM edge
        self.humemai.increment_num_recalled(
            subject=URIRef(subject),
            predicate=URIRef(predicate),
            object_=URIRef(object_),
            lower_time_added_bound=time_val,
            upper_time_added_bound=time_val,
        )
        la_literal = Literal(
            self.current_time.isoformat(timespec="seconds"), datatype=XSD.dateTime
        )
        self.humemai.update_last_accessed(
            subject=URIRef(subject),
            predicate=URIRef(predicate),
            object_=URIRef(object_),
            new_time=la_literal,
            lower_time_added_bound=time_val,
            upper_time_added_bound=time_val,
        )

    # -------------------------------------------------------------------------
    # _run_test_episode
    # -------------------------------------------------------------------------

    def _run_test_episode(self) -> tuple[float, int]:
        """Test run an episode, returning total reward + steps."""

        score = 0.0
        self.current_step = 0

        self.humemai.reset()
        obs, info = self.env.reset()

        self.manage_memory(obs["room"], self.current_step)

        while True:
            action_pair = self._generate_action_pair(obs)
            obs, reward, done, truncated, info = self.env.step(action_pair)
            score += reward

            self.current_step += 1

            self.manage_memory(obs["room"], self.current_step)
            if done:
                break

        return score, self.current_step
