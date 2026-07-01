from typing import TypedDict, List, Dict

class CompareState(TypedDict):
    entities: List[str]
    criteria: List[str]
    findings: Dict[tuple[str, str], str]
    final_table: str | None
    verdict: str | None
    llm_type: str
    use_qdrant: bool
    current_pair_index: int
