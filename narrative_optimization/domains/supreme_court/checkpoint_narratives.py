"""
Supreme Court Checkpoint Narratives
-----------------------------------

Legal analog to sports checkpoint builder. Splits a case into procedural
moments: filing, oral argument, conference, final decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


PHASES = (
    ("FILING", "Granting certiorari", 0.15),
    ("ORAL", "Oral arguments heard", 0.40),
    ("CONFERENCE", "Justices conference", 0.70),
    ("DECISION", "Opinion released", 1.0),
)


@dataclass(frozen=True)
class LegalSnapshot:
    case_id: str
    checkpoint_id: str
    sequence: int
    narrative: str
    metrics: Dict[str, float]
    metadata: Dict[str, str]
    target: int


class SupremeCourtCheckpointBuilder:
    def build(
        self,
        cases: Iterable[Dict],
        checkpoint: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        payload: List[Dict] = []
        for case in cases:
            per_case = self._build_case_snapshots(case)
            if checkpoint:
                per_case = [snap for snap in per_case if snap["checkpoint_id"] == checkpoint]
            payload.extend(per_case)
            if limit and len(payload) >= limit:
                return payload[:limit]
        return payload

    def _build_case_snapshots(self, case: Dict) -> List[Dict]:
        case_id = case.get("case_id") or case.get("case_name")
        if not case_id:
            return []

        outcome = case.get("outcome") or {}
        winner = outcome.get("winner")
        target = 1 if winner == "petitioner" else 0
        vote_margin = float(outcome.get("vote_margin") or 0)
        unanimous = int(outcome.get("unanimous") or 0)
        citation = float(outcome.get("citation_count") or 0)
        metadata = case.get("metadata") or {}
        area = metadata.get("area_of_law", "general")

        complexity = self._complexity(case)
        base_prob = 0.5 + 0.05 * (vote_margin / 5)

        snapshots: List[Dict] = []
        for seq, (code, label, progress) in enumerate(PHASES):
            win_prob = self._project_probability(base_prob, complexity, progress, unanimous)
            metrics = {
                "win_probability_petitioner": win_prob,
                "vote_margin": vote_margin,
                "citation_intensity": citation / 50000.0,
                "complexity": complexity,
            }
            meta = {
                "checkpoint_label": label,
                "area_of_law": area,
            }
            snapshots.append(
                self._to_dict(
                    LegalSnapshot(
                        case_id=case_id,
                        checkpoint_id=code,
                        sequence=seq,
                        narrative=self._narrative(label, case, win_prob),
                        metrics=metrics,
                        metadata=meta,
                        target=target,
                    )
                )
            )
        return snapshots

    def _complexity(self, case: Dict) -> float:
        outcome = case.get("outcome") or {}
        metadata = case.get("metadata") or {}
        briefs = sum(
            len(case.get(key) or "") > 200
            for key in ("petitioner_brief", "respondent_brief", "oral_arguments")
        )
        citation = float(outcome.get("citation_count") or metadata.get("citation_count") or 0)
        overturned = int(outcome.get("overturned") or 0)
        precedent = int(outcome.get("precedent_setting") or 0)
        return min(0.3 * briefs + 0.4 * precedent + 0.2 * overturned + 0.1 * (citation / 60000), 1.0)

    def _project_probability(self, base, complexity, progress, unanimous):
        adj = base + 0.15 * complexity * progress + 0.05 * unanimous * progress
        return float(np.clip(adj, 0.05, 0.95))

    @staticmethod
    def _narrative(label: str, case: Dict, win_prob: float) -> str:
        case_name = case.get("case_name") or case.get("case_id")
        area = (case.get("metadata") or {}).get("area_of_law", "general")
        return f"{label}: {case_name} ({area}). Petitioner win probability {win_prob:.1%}."

    @staticmethod
    def _to_dict(snapshot: LegalSnapshot) -> Dict:
        return {
            "game_id": snapshot.case_id,
            "case_id": snapshot.case_id,
            "checkpoint_id": snapshot.checkpoint_id,
            "sequence": snapshot.sequence,
            "narrative": snapshot.narrative,
            "metrics": snapshot.metrics,
            "metadata": snapshot.metadata,
            "target": snapshot.target,
        }


def build_supreme_court_checkpoint_snapshots(
    cases: Iterable[Dict],
    checkpoint: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    return SupremeCourtCheckpointBuilder().build(cases, checkpoint=checkpoint, limit=limit)


