from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from ..state import ResearchState, CriticResult

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

critic_prompt = PromptTemplate(
    input_variables=["subquery", "answer"],
    template=(
        "You are an evaluation critic.\n"
        "Evaluate the RESEARCH ANSWER below using these criteria:\n"
        "- Is it specific and informative?\n"
        "- Is it directly relevant to the subquery?\n"
        "- Does it avoid vague or unsupported statements?\n"
        "- Does it answer the subquery clearly?\n\n"
        "Return STRICT JSON ONLY with this format:\n"
        "{{\n"
        "  \"verdict\": \"accept\" or \"revise\",\n"
        "  \"reason\": \"short explanation\"\n"
        "}}\n\n"
        "Subquery: {subquery}\n"
        "Research Answer: {answer}\n"
    ),
)

def critic_node(state: ResearchState) -> ResearchState:
    """
    Evaluate each research output and decide whether to accept or revise it.
    Produces critic_results and sets state['needs_revision'].
    :param state:
    :return:
    """
    subqueries: List[str] =state["subqueries"]
    research_output: List[str] = state["research_output"]

    critic_results: List[CriticResult] = []

    for sub, ans in zip(subqueries, research_output):
        chain = critic_prompt | llm
        raw = chain.invoke({"subquery": sub, "answer": ans}).content.strip()
        print(raw)

        # Try to parse JSON
        try:
            import json
            verdict_obj = json.loads(raw)
        except Exception as e:
            # Fallback for messy outputs
            verdict_obj = {
                "verdict": "revise",
                "reason": f"Unable to parse model output as valid JSON {str(e)}",
            }

        # Normalize verdict
        verdict = verdict_obj.get("verdict", "").lower()
        print("vedict: ",verdict)
        if verdict not in ["accept", "revise"]:
            verdict = "revise"

        result: CriticResult = {
            "verdict": verdict,
            "reason": verdict_obj.get("reason", "No reason provided."),
        }

        critic_results.append(result)

    # Determine if any needs revision
    needs_revision = any(r["verdict"] == "revise" for r in critic_results)

    # Update state
    # state["critic_results"] = critic_results
    # state["needs_revision"] = needs_revision
    return {"critic_results": critic_results, "needs_revision": needs_revision}
