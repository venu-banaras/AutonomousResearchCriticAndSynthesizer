from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List
from ..state import ResearchState

# LLM for the MVP
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt for simulated research
research_prompt = PromptTemplate(
    input_variables=["subquery"],
    template=(
        "You are a research assistant.\n"
        "Write a short factual-style paragraph answering the question below.\n"
        "Keep it between 3 to 5 sentences.\n\n"
        "Question: {subquery}"
    ),
)

def research_subqueries(state: ResearchState) -> ResearchState:
    """
    Generate a short research-style summary for each subquery.
    :param state:
    :return:
    """
    subqueries = state["subqueries"]
    research_outputs: List[str] = []

    for sub in subqueries:
        chain = research_prompt | llm
        result = chain.invoke({"subquery": sub}).content

        research_outputs.append(result.strip())

    state["research_output"] = research_outputs
    return state
