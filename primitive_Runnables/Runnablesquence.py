from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3.1:8b")

prompt = PromptTemplate(
    template="Write a poem on the topic {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt,model,parser)

result = chain.invoke({'topic':'Russia'})

print(result)