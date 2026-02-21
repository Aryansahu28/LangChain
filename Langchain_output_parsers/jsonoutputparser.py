from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOllama(model="llama3.1:8b")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Write a detail description about {topic} /n{format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)

