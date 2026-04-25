"""
logger.py — Structured JSON logger for query / response pairs.
Every request flowing through the pipeline is persisted as a JSONL record.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class QueryLogger:
    """Append-only JSONL logger. One file per calendar day."""

    def __init__(self, log_path: str = "./logs"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

    def _log_file(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.log_path / f"queries_{date_str}.jsonl"

    def log(
        self,
        query: str,
        response: str,
        retrieved_context: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write one record and return it (so callers can reuse it)."""
        record: Dict[str, Any] = {
            "id":               str(uuid.uuid4()),
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "query":            query,
            "response":         response,
            "retrieved_context": retrieved_context or [],
            "query_embedding":  query_embedding,
            "metadata":         metadata or {},
        }
        with open(self._log_file(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    def load_logs(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load all records from today's log file (newest-first if n is given)."""
        records: List[Dict[str, Any]] = []
        lf = self._log_file()
        if lf.exists():
            with open(lf, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records[-n:] if n else records

    def load_all_logs(self) -> List[Dict[str, Any]]:
        """Load records from every log file in the directory."""
        records: List[Dict[str, Any]] = []
        for lf in sorted(self.log_path.glob("queries_*.jsonl")):
            with open(lf, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records
