from typing import TypedDict, List, Dict, Optional

class CompareState(TypedDict):
    entities: List[str]
    criteria: List[str]
    findings: Dict[str, List[str]]
    final_table: Optional[str]
    verdict: Optional[str]
    current_pair_index: int
    # Qdrant client is not stored in state; accessed via qdrant_client module
