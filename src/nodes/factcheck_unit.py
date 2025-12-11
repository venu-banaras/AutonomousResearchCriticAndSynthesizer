from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

factcheck_unit_prompt = PromptTemplate(
    input_variables=["subquery", "answer"],
    template=(
        "You are a fact-checking assistant.\n"
        "Evaluate the research answer for factual strength and specificity.\n"
        "Identify vague, unsupported, or logically weak statements.\n"
        "Your job is NOT to access the internet — only to analyze internal consistency.\n\n"
        "Return STRICT JSON ONLY:\n"
        "{{\n"
        "  \"verdict\": \"pass\" or \"fail\",\n"
        "  \"weak_points\": \"short description of vague or unsupported parts\",\n"
        "  \"suggestion\": \"how to strengthen the answer\"\n"
        "}}\n\n"
        "Subquery: {{subquery}}\n"
        "Research Answer: {{answer}}\n"
    ),
)

def factcheck_unit(subquery: str, answer: str) -> dict:
    """
    Fact-check a single research answer for internal coherence and vague claims.
    Pure function: returns a dict and does not mutate global state.
    """
    chain = factcheck_unit_prompt | llm
    raw = chain.invoke({"subquery": subquery, "answer": answer}).content.strip()

    # Try to parse JSON, fallback to fail
    try:
        import json
        parsed = json.loads(raw)
    except Exception as e:
        parsed = {
            "verdict": "fail",
            "weak_points": f"Model returned invalid JSON {e}.",
            "suggestion": "Rewrite with specific evidence and details.",
        }

    # Normalize verdict
    verdict = parsed.get("verdict", "").lower()
    if verdict not in ["pass", "fail"]:
        verdict = "fail"
    return {
        "verdict": verdict,
        "weak_points": parsed.get("weak_points", ""),
        "suggestion": parsed.get("suggestion", ""),
    }