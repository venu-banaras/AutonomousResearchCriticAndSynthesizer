from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Each research query holds one sub-query
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

research_unit_prompt = PromptTemplate(
    input_variables=["subquery"],
    template=(
        "You are a research assistant.\n"
        "Write a factual, detailed answer to the following subquery.\n"
        "Keep the answer between 3 to 5 sentences.\n"
        "Avoid vague statements like 'some studies' or 'in general'.\n\n"
        "Subquery: {{subquery}}\n"
    ),
)

def research_unit(subquery: str) -> str:
    """
    Takes a single subquery and returns a single research paragraph.
    This function is designed for parallel mapping over subqueries.
    """
    chain = research_unit_prompt | llm
    response = chain.invoke({"subquery": subquery}).content.strip()

    return  response