from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from ..state import ResearchState

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

contradiction_prompt = PromptTemplate(
    input_variables=["outputs"],
    template=(
            "You are a contradiction-detection assistant.\n"
            "Analyze the following research outputs for logical or factual conflicts.\n"
            "Identify contradictions such as:\n"
            "- Opposing claims\n"
            "- Conflicting statistics\n"
            "- Incompatible trends or predictions\n\n"
            "Return STRICT JSON ONLY:\n"
            "{{\n"
            "  \"verdict\": \"conflict\" or \"no_conflict\",\n"
            "  \"conflicting_pairs\": [\"summary 1 vs summary 2\", ...],\n"
            "  \"summary\": \"short explanation\"\n"
            "}}\n\n"
            "Research Outputs:\n"
            "{outputs}\n"
    ),
)


def contradiction_node(state: ResearchState) -> ResearchState:
    outputs: List[str] = state["research_outputs"]
    joined = "\n\n----\n\n".join(outputs)

    chain = contradiction_prompt | llm
    raw = chain.invoke({"outputs": joined}).content.strip()

    try:
        import json
        report = json.loads(raw)
    except Exception as e:
        report = {
            "verdict": "conflict",
            "conflicting_pairs": [f"JSON parse fail {e}"],
            "summary": "Model output invalid JSON.",
        }

    verdict = report.get("verdict", "").lower()
    if verdict not in ["conflict", "no_conflict"]:
        verdict = "conflict"

    needs_fix = verdict == "conflict"

    return {"contradiction_report": report, "needs_contradiction_fix": needs_fix}
