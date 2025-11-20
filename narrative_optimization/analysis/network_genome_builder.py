"""
Network Genome Builder
======================

Constructs relational graphs for each domain and projects graph-theoretic
features back onto individual records so transformers + betting systems can
reason about lineage, prestige, and pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import networkx as nx
import numpy as np


@dataclass
class NetworkFeatureBundle:
    node_metrics: Dict[str, Dict[str, float]]
    graph_metadata: Dict[str, Any]


class NetworkGenomeBuilder:
    """Domain-aware helper that builds and annotates relational graphs."""

    def __init__(self, domain_name: str, persist_dir: Optional[Path] = None):
        self.domain_name = domain_name
        self.graph = nx.Graph()
        self.persist_dir = (
            Path(persist_dir)
            if persist_dir
            else Path("narrative_optimization/results/network_genomes")
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def annotate_records(self, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build graph (if needed) and append network metrics to each record.
        """
        records_list = list(records)
        if not records_list:
            return {"records": [], "graph_metadata": {}}

        self.graph.clear()
        if self.domain_name == "nhl":
            self._build_graph_nhl(records_list)
        elif self.domain_name == "nfl":
            self._build_graph_nfl(records_list)
        elif self.domain_name == "supreme_court":
            self._build_graph_supreme_court(records_list)
        else:
            self._build_graph_generic(records_list)

        bundle = self._compute_node_metrics()
        self._persist(bundle)

        for record in records_list:
            features = self._extract_features_for_record(record, bundle.node_metrics)
            record.setdefault("network_features", {}).update(features)

        return {"records": records_list, "graph_metadata": bundle.graph_metadata}

    # ------------------------------------------------------------------ #
    # Graph construction helpers
    # ------------------------------------------------------------------ #

    def _build_graph_nhl(self, records: List[Dict[str, Any]]) -> None:
        for record in records:
            home = record.get("home_team")
            away = record.get("away_team")
            if not home or not away:
                continue

            weight = 1.0
            if record.get("is_playoff"):
                weight += 0.5
            if record.get("is_rivalry"):
                weight += 0.35

            context = record.get("temporal_context") or {}
            rest_adv = abs(float(context.get("rest_advantage") or 0.0))
            record_diff = abs(float(context.get("record_differential") or 0.0))
            weight += min(rest_adv / 2.0, 0.4)
            weight += min(record_diff / 0.4, 0.3)

            matchup_key = tuple(sorted([home, away]))
            if self.graph.has_edge(*matchup_key):
                self.graph[matchup_key[0]][matchup_key[1]]["weight"] += weight
            else:
                self.graph.add_edge(matchup_key[0], matchup_key[1], weight=weight)

    def _build_graph_nfl(self, records: List[Dict[str, Any]]) -> None:
        for record in records:
            home = record.get("home_team")
            away = record.get("away_team")
            if not home or not away:
                continue

            playoff_flag = 1 if record.get("playoff") or record.get("is_playoff") else 0
            division_flag = 1 if record.get("div_game") or record.get("division_game") else 0
            weight = 1.0 + 0.5 * playoff_flag + 0.3 * division_flag
            spread = abs(float(record.get("spread_line") or 0.0))
            weight += min(spread / 10, 0.25)

            matchup_key = tuple(sorted([home, away]))
            if self.graph.has_edge(*matchup_key):
                self.graph[matchup_key[0]][matchup_key[1]]["weight"] += weight
            else:
                self.graph.add_edge(matchup_key[0], matchup_key[1], weight=weight)

            # Add QB lineage edges (QB nodes prefixed with qb:)
            home_qb = (record.get("home_qb") or {}).get("name")
            away_qb = (record.get("away_qb") or {}).get("name")
            if home_qb:
                self.graph.add_edge(f"qb:{home_qb}", home, weight=0.6)
            if away_qb:
                self.graph.add_edge(f"qb:{away_qb}", away, weight=0.6)

    def _build_graph_supreme_court(self, records: List[Dict[str, Any]]) -> None:
        for record in records:
            metadata = record.get("metadata") or {}
            author = metadata.get("author")
            area = metadata.get("area_of_law")
            case = record.get("case_id") or record.get("case_name")
            if author and case:
                self.graph.add_edge(f"justice:{author}", f"case:{case}", weight=1.0)
            if area and case:
                self.graph.add_edge(f"issue:{area}", f"case:{case}", weight=0.8)

            outcome = record.get("outcome") or {}
            margin = abs(float(outcome.get("vote_margin") or 0.0))
            unanimous = int(outcome.get("unanimous")) if outcome else 0
            if author and area:
                self.graph.add_edge(
                    f"justice:{author}",
                    f"issue:{area}",
                    weight=0.4 + 0.05 * margin + 0.1 * unanimous,
                )

    def _build_graph_generic(self, records: List[Dict[str, Any]]) -> None:
        for record in records:
            entity = (
                record.get("name")
                or record.get("title")
                or record.get("player_name")
                or record.get("company_id")
                or record.get("id")
            )
            category = (
                record.get("category")
                or record.get("market_category")
                or record.get("genre")
                or record.get("tournament_name")
            )
            if entity and category:
                self.graph.add_edge(str(entity), f"cluster:{category}", weight=0.5)

            founders = record.get("founders") or []
            if isinstance(founders, list):
                for founder in founders:
                    self.graph.add_edge(f"founder:{founder}", str(entity), weight=0.4)

            teammates = record.get("home_lineup") or record.get("away_lineup")
            if isinstance(teammates, list):
                for player in teammates:
                    name = player.get("name") if isinstance(player, dict) else player
                    if name and entity:
                        self.graph.add_edge(str(entity), f"player:{name}", weight=0.35)

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #

    def _compute_node_metrics(self) -> NetworkFeatureBundle:
        if not self.graph.nodes:
            return NetworkFeatureBundle(node_metrics={}, graph_metadata={})

        degrees = dict(self.graph.degree(weight="weight"))
        strength = nx.pagerank(self.graph, weight="weight")
        clustering = nx.clustering(self.graph, weight="weight")

        node_metrics: Dict[str, Dict[str, float]] = {}
        for node in self.graph.nodes:
            node_metrics[node] = {
                "degree": float(degrees.get(node, 0.0)),
                "centrality": float(strength.get(node, 0.0)),
                "clustering": float(clustering.get(node, 0.0)),
            }

        metadata = {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
        }
        return NetworkFeatureBundle(node_metrics=node_metrics, graph_metadata=metadata)

    def _extract_features_for_record(
        self, record: Dict[str, Any], node_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        if self.domain_name in {"nhl", "nfl"}:
            home = record.get("home_team")
            away = record.get("away_team")
            home_metrics = node_metrics.get(home, {})
            away_metrics = node_metrics.get(away, {})
            rivalry = self._edge_strength(home, away)
            return {
                "home_degree": home_metrics.get("degree", 0.0),
                "away_degree": away_metrics.get("degree", 0.0),
                "home_centrality": home_metrics.get("centrality", 0.0),
                "away_centrality": away_metrics.get("centrality", 0.0),
                "centrality_delta": home_metrics.get("centrality", 0.0)
                - away_metrics.get("centrality", 0.0),
                "rivalry_strength": rivalry,
            }
        elif self.domain_name == "supreme_court":
            metadata = record.get("metadata") or {}
            author = metadata.get("author")
            area = metadata.get("area_of_law")
            author_node = f"justice:{author}" if author else None
            area_node = f"issue:{area}" if area else None
            return {
                "author_influence": node_metrics.get(author_node, {}).get(
                    "centrality", 0.0
                ),
                "issue_gravity": node_metrics.get(area_node, {}).get(
                    "centrality", 0.0
                ),
            }
        else:
            entity = record.get("name") or record.get("title")
            return node_metrics.get(entity, {})

    def _edge_strength(self, node_a: Optional[str], node_b: Optional[str]) -> float:
        if not node_a or not node_b:
            return 0.0
        if not self.graph.has_edge(node_a, node_b):
            return 0.0
        edge_data = self.graph.get_edge_data(node_a, node_b, default={})
        return float(edge_data.get("weight", 0.0))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _persist(self, bundle: NetworkFeatureBundle) -> None:
        if not bundle.node_metrics:
            return
        out_path = self.persist_dir / f"{self.domain_name}_network_genome.json"
        payload = {
            "graph_metadata": bundle.graph_metadata,
            "sample_nodes": dict(list(bundle.node_metrics.items())[:20]),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


__all__ = ["NetworkGenomeBuilder", "NetworkFeatureBundle"]


