from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

critic_unit_prompt = PromptTemplate(
    input_variables=["subquery", "answer"],
    template=(
        "You are an evaluation critic.\n"
        "Evaluate the research answer using these criteria:\n"
        "- Specificity and clarity\n"
        "- Relevance to the subquery\n"
        "- Avoidance of vague or unsupported claims\n"
        "- Provides a meaningful answer\n\n"
        "Return STRICT JSON ONLY:\n"
        "{{\n"
        "  \"verdict\": \"accept\" or \"revise\",\n"
        "  \"reason\": \"short explanation\"\n"
        "}}\n\n"
        "Subquery: {{subquery}}\n"
        "Research Answer: {{answer}}\n"
    ),
)

def critic_unit(subquery: str, answer: str) -> dict:
    """
    Evaluate a single research output and return a verdict dictionary.
    Does NOT modify global state. Pure functional node.
    """
    chain = critic_unit_prompt | llm
    raw = chain.invoke({"subquery": subquery, "answer": answer}).content.strip()

    # Parse JSON
    try:
        import json
        parsed = json.loads(raw)
    except Exception as e:
        parsed = {
            "verdict": "revise",
            "reason": f"Model output invalid or non-JSON or needs revision {e}.",
        }

    # Normalize verdict
    verdict = parsed.get("verdict", "").lower()
    if verdict not in ["accept", "revise"]:
        verdict = "revise"

    return {
        "verdict": verdict,
        "reason": parsed.get("reason", "No explanation provided."),
    }