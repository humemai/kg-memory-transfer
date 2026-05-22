"""A simplified long-term agent using plain RDF triples (no RDF-star qualifiers).

- Maintains TWO graphs:
    - Short-Term (ST) graph: latest observations only (cleared each step).
    - Long-Term (LT) graph: accumulated knowledge with capacity + random eviction.

- QA: uniform among (s, p, ?) matches from LT.
- Explore: BFS to nearest unseen room using adjacency built from LT;
    if multiple neighbors per (room, dir), pick one at random; ignore walls.
- Forget: random eviction on LT to respect max memory size.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Optional, Sequence

from .agent import Agent
from rdflib import Graph, URIRef


class LongTermSimpleAgent(Agent):
    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v3",
        env_config: dict[str, Any] | None = None,
        max_long_term_memory_size: int = 100,
        num_samples_for_results: int = 1,
        save_results: bool = True,
        default_root_dir: str = "./training-results/",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        if env_config is None:
            env_config = {"terminates_at": 99, "room_size": "dev"}

        params_to_save = deepcopy(locals())
        del params_to_save["self"]
        del params_to_save["__class__"]
        super().__init__(**params_to_save)

        self.max_long_term_memory_size = max_long_term_memory_size
        # Two graphs for easier tracking/inspection
        self.st_graph: Graph = Graph()  # short-term: cleared every step
        self.lt_graph: Graph = Graph()  # long-term: capacity-limited
        self.directions = {"north", "south", "east", "west"}

    # Memory management: add all obs; randomly evict to capacity
    def manage_memory(self, observations: list[list[str]]) -> None:

        # 1) Move every existing short-term memory to long-term memory
        for s, p, o in list(self.st_graph):
            self.lt_graph.add((s, p, o))

        # Enforce LT capacity via random eviction of triples
        if self.max_long_term_memory_size >= 0:
            while len(self.lt_graph) > self.max_long_term_memory_size:
                t = random.choice(list(self.lt_graph))
                self.lt_graph.remove(t)

        # 2) Clear ST and 3) add new observations to ST only (latest observations)
        self.st_graph.remove((None, None, None))
        for s, p, o in observations:
            s_ref, p_ref, o_ref = URIRef(s), URIRef(p), URIRef(o)
            self.st_graph.add((s_ref, p_ref, o_ref))

    # QA: uniform among matches
    def answer_question(self, question: Sequence[str]) -> Optional[str]:
        s, p = question[0], question[1]
        s_ref, p_ref = URIRef(s), URIRef(p)
        # Prefer answers from ST (latest observations)
        st_candidates = [
            str(o) for (_, _, o) in self.st_graph.triples((s_ref, p_ref, None))
        ]
        if st_candidates:
            return random.choice(st_candidates)
        lt_candidates = [
            str(o) for (_, _, o) in self.lt_graph.triples((s_ref, p_ref, None))
        ]
        return random.choice(lt_candidates) if lt_candidates else None

    # Exploration: BFS to nearest unseen room; random conflict resolution
    def explore(self) -> str:
        current_room = self._get_current_room()

        visited = self._get_visited_rooms()
        adjacency = self._build_adjacency()

        path = self._bfs_to_new(current_room, adjacency, visited)
        if path:
            return path[0][1]

        options = [d for (d, _) in adjacency.get(current_room, [])]
        if options:
            return random.choice(options)
        return random.choice(["north", "south", "east", "west", "stay"])  # dead-end

    def _build_adjacency(self) -> dict[str, list[tuple[str, str]]]:
        # Build two grouped maps, prefer ST if present for (room, dir)
        grouped_st: dict[tuple[str, str], list[str]] = defaultdict(list)
        for s, p, o in self.st_graph:
            s_str, p_str, o_str = str(s), str(p), str(o)
            d = self._as_direction(p_str)
            if d is None or o_str.endswith("wall"):
                continue
            grouped_st[(s_str, d)].append(o_str)

        grouped_lt: dict[tuple[str, str], list[str]] = defaultdict(list)
        for s, p, o in self.lt_graph:
            s_str, p_str, o_str = str(s), str(p), str(o)
            d = self._as_direction(p_str)
            if d is None or o_str.endswith("wall"):
                continue
            grouped_lt[(s_str, d)].append(o_str)

        adjacency: dict[str, list[tuple[str, str]]] = {}
        keys = set(grouped_lt.keys()) | set(grouped_st.keys())
        for key in keys:
            room, dir_ = key
            # Prefer ST choices; else LT
            if key in grouped_st and grouped_st[key]:
                nbr = random.choice(grouped_st[key])
                adjacency.setdefault(room, []).append((dir_, nbr))
            elif key in grouped_lt and grouped_lt[key]:
                nbr = random.choice(grouped_lt[key])
                adjacency.setdefault(room, []).append((dir_, nbr))
        return adjacency

    def _last_segment(self, value: str) -> str:
        return value.rsplit("/", 1)[-1]

    def _as_direction(self, predicate: str) -> Optional[str]:
        last = self._last_segment(predicate)
        if last in self.directions:
            return last
        if predicate in self.directions:
            return predicate
        return None

    def _get_current_room(self) -> Optional[str]:
        # Prefer short-term (latest) (agent, at_location, room)
        st_rooms = [
            str(o)
            for (s, p, o) in self.st_graph
            if self._last_segment(str(s)) == "agent"
            and self._last_segment(str(p)) == "at_location"
        ]

        assert len(st_rooms) == 1

        return st_rooms[0]

    def _get_visited_rooms(self) -> set[str]:
        # Visited rooms are those seen historically in LT
        return {
            str(o)
            for (s, p, o) in self.lt_graph
            if self._last_segment(str(s)) == "agent"
            and self._last_segment(str(p)) == "at_location"
        }

    def _bfs_to_new(
        self,
        start: str,
        adjacency: dict[str, list[tuple[str, str]]],
        visited: set[str],
    ) -> Optional[list[tuple[str, str, str]]]:
        q = deque([start])
        parent: dict[str, Optional[tuple[str, str]]] = {start: None}
        while q:
            cur = q.popleft()
            if cur not in visited and cur != start:
                return self._reconstruct(parent, start, cur)
            for d, nbr in adjacency.get(cur, []):
                if nbr not in parent:
                    parent[nbr] = (cur, d)
                    q.append(nbr)
        return None

    def _reconstruct(
        self, parent: dict[str, Optional[tuple[str, str]]], start: str, goal: str
    ) -> list[tuple[str, str, str]]:
        edges: list[tuple[str, str, str]] = []
        cur = goal
        while cur != start and cur in parent:
            link = parent[cur]
            if not link:
                break
            prev, d = link
            edges.append((prev, d, cur))
            cur = prev
        edges.reverse()
        return edges

    def _run_test_episode(self) -> tuple[float, int]:
        score = 0.0
        # Reset graphs for a new episode
        self.st_graph.remove((None, None, None))
        self.lt_graph.remove((None, None, None))
        obs, _info = self.env.reset()

        step = 0
        self.manage_memory(obs["room"])

        while True:
            action_pair = self._generate_action_pair(obs)
            obs, reward, done, _truncated, _info = self.env.step(action_pair)
            score += reward
            step += 1
            self.manage_memory(obs["room"])
            if done:
                break

        return score, step

    def get_num_main_triples(self) -> int:
        return len(self.lt_graph)
