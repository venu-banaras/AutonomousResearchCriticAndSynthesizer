from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from ..state import ResearchState

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

fact_prompt = PromptTemplate(
    input_variables=["subquery", "answer"],
    template=(
        "You are a fact-checking assistant.\n"
        "Analyze the RESEARCH ANSWER for unsupported claims or vague statements.\n"
        "Identify whether the answer is factually strong or needs improvement.\n\n"
        "Return STRICT JSON ONLY in this format:\n"
        "{{\n"
        "  \"verdict\": \"pass\" or \"fail\",\n"
        "  \"weak_points\": \"list vague or unsupported parts\",\n"
        "  \"suggestion\": \"how to improve the answer\"\n"
        "}}\n\n"
        "Subquery: {subquery}\n"
        "Research Answer: {answer}\n"
    ),
)

def fact_check_node(state: ResearchState) -> ResearchState:
    subqueries: List[str] = state["subqueries"]
    outputs: List[str] = state["research_output"]

    fact_results : List[Dict] = []

    for sub, ans in zip(subqueries, outputs):
        chain = fact_prompt | llm
        raw = chain.invoke({"subquery": sub, "answer": ans}).content.strip()

        try:
            import json
            obj = json.loads(raw)
        except Exception as e:
            obj = {
                "verdict": "fail",
                "weak_points": f"Model returned invalid JSON {e}",
                "suggestion": "Regenerate using mode factual detail."
            }

        # Normalize verdict
        verdict = obj.get("verdict", "").lower()
        if verdict not in ["pass", "fail"]:
            verdict = "fail"

        obj["verdict"] = verdict
        fact_results.append(obj)

    # Determine if revision needed
    needs_factfix = any(r["verdict"] == "fail" for r in fact_results)

    # Update state
    # state["fact_checks"] = fact_results
    # state["needs_factfix"] = needs_factfix

    return {"fact_checks": fact_results, "needs_factfix": needs_factfix}
