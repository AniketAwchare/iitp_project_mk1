"""
dashboard/interface.py — Abstract UI contract.

All dashboard backends (Streamlit now, React/Next.js later) must
accept a MetricsSnapshot and render it. The Streamlit app imports
this class and uses push_snapshot() to update state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class MetricsSnapshot:
    """
    One point-in-time snapshot of all four observability module outputs.
    Passed to whatever UI backend is active.
    """
    timestamp:         str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Module 1 — Consistency
    consistency_score: Optional[float] = None
    consistency_flagged: bool           = False

    # Module 2 — Drift
    drift_detected:    Optional[bool]  = None
    drift_magnitude:   Optional[float] = None

    # Module 3 — Retrieval Alignment
    alignment_score:   Optional[float] = None
    alignment_flagged: bool             = False

    # Module 4 — Anomaly
    anomaly_score:     Optional[float] = None
    is_anomaly:        bool             = False

    # Raw query / response for display
    query:             str  = ""
    response:          str  = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":           self.timestamp,
            "consistency_score":   self.consistency_score,
            "consistency_flagged": self.consistency_flagged,
            "drift_detected":      self.drift_detected,
            "drift_magnitude":     self.drift_magnitude,
            "alignment_score":     self.alignment_score,
            "alignment_flagged":   self.alignment_flagged,
            "anomaly_score":       self.anomaly_score,
            "is_anomaly":          self.is_anomaly,
            "query":               self.query,
            "response":            self.response,
        }


class DashboardInterface:
    """
    Base class for dashboard backends.
    Subclass this to implement React/Next.js or any other UI.
    Streamlit's app.py uses push_snapshot() directly via session state.
    """

    def __init__(self):
        self._history: List[MetricsSnapshot] = []

    def push_snapshot(self, snapshot: MetricsSnapshot) -> None:
        """Record a snapshot. UI backends override this to trigger re-render."""
        self._history.append(snapshot)

    def get_history(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._history]

    def clear(self) -> None:
        self._history.clear()
