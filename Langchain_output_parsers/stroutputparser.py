from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

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

result1 = model.invoke(prompt1)

print(result1)
print()
print()
print("*************************************************************************8")

prompt2 = template2.invoke({'text':result1.content})

result = model.invoke(prompt2)

print(result.content)



