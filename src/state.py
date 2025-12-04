from typing import List, TypedDict

class ResearchState(TypedDict):
    query: str
    subqueries: List[str]
    research_output: List[str]
    final_answer: str