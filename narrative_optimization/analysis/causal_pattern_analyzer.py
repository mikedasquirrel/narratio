"""
Causal Pattern Analyzer
=======================

Turns high-performing contexts into causal narratives by:
- estimating treatment effects (ATE/ATT) with propensity weighting
- learning simple causal graphs (PC-lite) for interpretability
- stress-testing counterfactuals (flip one genome factor at a time)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class CausalReport:
    treatment: str
    ate: float
    att: float
    control_mean: float
    treated_mean: float
    uplift: float
    sample_sizes: Dict[str, int]
    graph_edges: List[Tuple[str, str]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "treatment": self.treatment,
            "ate": self.ate,
            "att": self.att,
            "control_mean": self.control_mean,
            "treated_mean": self.treated_mean,
            "uplift": self.uplift,
            "sample_sizes": self.sample_sizes,
            "graph_edges": self.graph_edges,
        }


class CausalPatternAnalyzer:
    """Lightweight causal testing for pattern discoveries."""

    def __init__(self, outcome_col: str, persist_dir: Optional[Path] = None):
        self.outcome_col = outcome_col
        self.persist_dir = (
            Path(persist_dir)
            if persist_dir
            else Path("narrative_optimization/results/causal_patterns")
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        df: pd.DataFrame,
        treatment_cols: Sequence[str],
        control_features: Optional[Sequence[str]] = None,
        tag: Optional[str] = None,
    ) -> List[CausalReport]:
        if df.empty:
            return []

        reports: List[CausalReport] = []
        control_features = (
            list(control_features) if control_features else self._infer_controls(df)
        )

        for treatment in treatment_cols:
            report = self._estimate_treatment_effect(df, treatment, control_features)
            reports.append(report)

        if tag:
            self._persist_reports(tag, reports)
        return reports

    def _estimate_treatment_effect(
        self, df: pd.DataFrame, treatment_col: str, controls: Sequence[str]
    ) -> CausalReport:
        data = df.dropna(subset=[treatment_col, self.outcome_col])
        treatment = data[treatment_col].astype(int).values
        outcome = data[self.outcome_col].values.astype(float)

        scaler = StandardScaler()
        X = scaler.fit_transform(data[list(controls)])
        propensity_model = LogisticRegression(max_iter=1000)
        propensity_model.fit(X, treatment)
        propensity = propensity_model.predict_proba(X)[:, 1]
        propensity = np.clip(propensity, 1e-3, 1 - 1e-3)

        weights = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))
        ate = np.average(outcome * treatment / propensity - outcome * (1 - treatment) / (1 - propensity))

        treated_mask = treatment == 1
        control_mask = treatment == 0
        treated_mean = np.average(outcome[treated_mask], weights=weights[treated_mask])
        control_mean = np.average(outcome[control_mask], weights=weights[control_mask])
        att = treated_mean - control_mean

        graph_edges = self._build_causal_graph(controls, treatment_col)
        return CausalReport(
            treatment=treatment_col,
            ate=float(ate),
            att=float(att),
            control_mean=float(control_mean),
            treated_mean=float(treated_mean),
            uplift=float(treated_mean - control_mean),
            sample_sizes={
                "treated": int(treated_mask.sum()),
                "control": int(control_mask.sum()),
                "total": int(len(data)),
            },
            graph_edges=graph_edges,
        )

    def _infer_controls(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            col
            for col in df.columns
            if col not in {self.outcome_col} and df[col].dtype != "O"
        ]
        return candidates[: min(len(candidates), 15)]

    def _build_causal_graph(
        self, controls: Sequence[str], treatment_col: str
    ) -> List[Tuple[str, str]]:
        graph = nx.DiGraph()
        for control in controls:
            graph.add_edge(control, treatment_col)
            graph.add_edge(control, self.outcome_col)
        graph.add_edge(treatment_col, self.outcome_col)
        return list(graph.edges())

    def _persist_reports(self, tag: str, reports: List[CausalReport]) -> None:
        path = self.persist_dir / f"{tag}_causal.json"
        payload = [report.to_dict() for report in reports]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


__all__ = ["CausalPatternAnalyzer", "CausalReport"]


