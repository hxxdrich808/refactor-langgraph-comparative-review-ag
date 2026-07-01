from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class CompareState:
    """
    Holds the entire state of the comparative review process.
    All nodes read from and write to this single instance via the same keys.
    """
    entities: List[str] = field(default_factory=list)
    criteria: List[str] = field(default_factory=list)
    findings: Dict[Tuple[str, str], str] = field(default_factory=dict)
    current_pair_index: int = 0
    table_markdown: str = ""
    verdict: str = ""
