from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.1:8b",temperature=0)

template = PromptTemplate(
    template="""
    Write a short story about a robot learning emotions. Include:
- a {beginning_input}
- a conflict
- a resolution
Keep it under 200 words.
""",
input_variables=['beginning_input']
)

prompt = template.invoke({'beginning_input':'beggining'})

result = llm.invoke(prompt)

print(result.content)