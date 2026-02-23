from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3.1:8b")

prompt1 = PromptTemplate(
    template=" Illustration on the topic {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template=" Give me 5 lines this text : {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'Context':RunnableSequence(prompt1,model,parser),
    'epitome':RunnableSequence(prompt2,model,parser)
})

result = chain.invoke({'topic':'Russia','text':'Ukraine'})

print(result)

final_result = """The context is {} \n and summary of it is {}""".format((result['Context']),result['epitome'])


print(final_result)