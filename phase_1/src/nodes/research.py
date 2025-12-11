from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from typing import List
from ..state import ResearchState
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
# LLM for the MVP
llm = ChatOllama(model="llama3.1", temperature=0)

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
    print(f"Generating {len(subqueries)} short research-style prompts...")

    for sub in subqueries:
        chain = research_prompt | llm
        result = chain.invoke({"subquery": sub}).content

        research_outputs.append(result.strip())

    # state["research_output"] = research_outputs
    print(f"Generated {len(subqueries)} short research-style prompts.\n")
    print(f"Research output prompts: {research_outputs}")
    return {"research_output": research_outputs}
