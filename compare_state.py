from typing import TypedDict, List, Dict, Optional

class CompareState(TypedDict):
    entities: List[str]
    criteria: List[str]
    findings: Dict[tuple[str, str], str]
    final_table: Optional[str]
    verdict: Optional[str]
