from typing import List, TypedDict, Dict


class CriticResult(TypedDict):
    verdict: str  # "accept" or "revise"
    reason: str


class ResearchState(TypedDict):
    query: str
    subqueries: List[str]
    research_output: List[str]
    final_answer: str

    # Phase 2 additions
    critic_results: List[CriticResult]
    needs_revision: bool
