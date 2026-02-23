from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3.1:8b")

prompt1 = PromptTemplate(
    template="Write a poem on the topic {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="Explain this text {text}",
    input_variables=['text']
)

parser = StrOutputParser()

poem_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'poem': RunnablePassthrough(),
    'illustration':RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(poem_chain,parallel_chain)
result = final_chain.invoke({'topic':'Russia'})

print(f"Poem : \n {result['poem']} \n Illustration:\n {result['illustration']}")