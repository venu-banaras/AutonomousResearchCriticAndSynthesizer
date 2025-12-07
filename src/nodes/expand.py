from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from typing import List
from ..state import ResearchState
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)

# Simple LLM instance for now
llm = ChatOllama(model="llama3.1", temperature=0)

# Prompt template for splitting query

expand_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a query decomposition assistant.\n"
        "Break the following user query into 2 or 3 specific sub-questions.\n"
        "Return ONLY a numbered list of the sub-questions.\n\n"
        "User query: {query}"
    ),
)


def expand_query(state: ResearchState) -> ResearchState:
    """
    Expand user query into 2 or 3 specific sub-questions.
    :param state:
    :return:
    """
    query = state["query"]
    chain = expand_prompt | llm
    response = chain.invoke({"query": query}).content

    # Parse numbered list into python list
    lines = response.strip().splitlines()
    subqueries: List[str] = []
    print("Expanding prompts into subqueries...")

    for line in lines:
        line = line.strip()
        if line[:2].isdigit() or line.startswith("1.") or line.startswith("1)"):
            # Remove leading numbers and punctuation
            cleaned = line.split(".", 1)[-1].strip()
            subqueries.append(cleaned)
        else:
            # Fallback: add non-numbered lines directly
            subqueries.append(line)
    state["subqueries"] = subqueries
    print(f"Expanded prompts into subqueries: {subqueries}")
    return state