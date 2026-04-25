"""
llm_observability
=================
Observability & Reliability Framework for Production LLM Systems.
Domain: Financial Services (Banking, Insurance, Trading)
Author: Aniket Awchare | IIT Patna M.Tech AI & Data Science 2025-26
"""

__version__ = "1.0.0"
__author__  = "Aniket Awchare"

from .core.config    import Config
from .core.pipeline  import LLMPipeline
from .core.logger    import QueryLogger

from .modules.consistency import ConsistencyScorer
from .modules.drift       import DriftDetector
from .modules.retrieval   import RetrievalAlignmentScorer
from .modules.anomaly     import AnomalyDetector

__all__ = [
    "Config",
    "LLMPipeline",
    "QueryLogger",
    "ConsistencyScorer",
    "DriftDetector",
    "RetrievalAlignmentScorer",
    "AnomalyDetector",
]
