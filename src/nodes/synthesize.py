from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from typing import List
from ..state import ResearchState
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
# Same LLM used across nodes for consistency
llm = ChatOllama(model="llama3.1", temperature=0)

synth_prompt = PromptTemplate(
    input_variables=["joined_notes", "query"],
    template=(
        "You are a report synthesizer.\n"
        "Combine the following research notes into a coherent, concise answer.\n"
        "Use this structure:\n"
        "- 1–2 sentence introduction that references the user query\n"
        "- bullet-style merged key insights (3 bullets max)\n"
        "- 1 closing sentence summarizing the take-away\n\n"
        "Research notes:\n"
        "{joined_notes}\n\n"
        "User query: {query}"
    ),
)

def synthesize_results(state: ResearchState) -> ResearchState:
    """
    Combine research outputs into a final answer and update state["final_answer"].
    Fallback: if no research outputs available, produce a short answer based on the query.
    :param state:
    :return:
    """
    research_outputs: List[str] = state.get("research_output", [])
    query: str = state.get("query", "")
    print("Synthesizing prompts...")

    if not research_outputs:
        # Minimal graceful fallback
        prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are a concise assistant. Produce a 3-4 sentence summary answering:\n"
                "User query: {query}"
            ),
        )
        chain = prompt | llm
        state["final_answer"] = chain.invoke({"query": query}).content.strip()
        return state
    joined = "\n\n".join(research_outputs)
    chain = synth_prompt | llm
    result = chain.invoke({"joined_notes": joined, "query": query}).content
    state["final_answer"] = result.strip()
    print(f"Synthesized {len(research_outputs)} prompts.\n")
    return state

