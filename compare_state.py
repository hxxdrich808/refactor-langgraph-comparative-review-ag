"""
State definition used by the LangGraph agent.
"""

from typing import Dict, Any, List

# The state is a simple dictionary that holds all data needed during the run.
CompareState = Dict[str, Any]

# Example of expected keys (not exhaustive):
# - entities: List[str]
# - criteria: List[str]
# - findings: Dict[Tuple[str, str], str]  # mapping (entity, criterion) -> snippet
# - final_table: Optional[str]
# - verdict: Optional[str]
# - current_pair_index: int
# - qdrant_vectors: Optional[Dict[str, List[float]]]  # vector representations per entity
