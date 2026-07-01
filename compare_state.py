"""
TypedDict definition for the agent's state.
"""

from typing import TypedDict, List, Dict, Tuple

class CompareState(TypedDict):
    entities: List[str]                # 3 names to compare
    criteria: List[str]                 # 3–5 comparison criteria
    findings: Dict[Tuple[str, str], str]      # (entity, criterion) -> note
    final_table: str | None
    verdict: str | None
    llm_type: str
    use_qdrant: bool
    current_pair_index: int
