from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3.1:8b")

template1 = PromptTemplate(
    template="Write a detail description about {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Summarize this {text} into 5 lines",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'Anne Hathway'})

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'Anne Frank'})

print(result)

