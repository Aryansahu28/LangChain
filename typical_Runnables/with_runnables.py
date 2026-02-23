from abc import ABC , abstractmethod
import random

class Runnables(ABC):
    def __init__(self):
        pass

    def invoke(self):
        pass

class NakliLLM(Runnables):

    def __init__(self):
        print("LLM is created")

    def invoke(self,prompt):
        response_list = [
            "Delhi is the capital of India",
            "IPL is cricket league",
            "ML stands for Machine Learning"
        ]

        return {'response':random.choice(response_list)}

    
    def predict(self,prompt):
        response_list = [
            "Delhi is the capital of India",
            "IPL is cricket league",
            "ML stands for Machine Learning"
        ]

        return {'response':random.choice(response_list)}
    
class NaklipromptTemplate(Runnables):

    def __init__(self,template,input_variable):
        self.template = template
        self.input_variable = input_variable

    def invoke(self,input_dict):
        return self.template.format(**input_dict)
    
    def format(self,input_dict):
        return self.template.format(**input_dict)

class RunnableConnector(Runnables):
    
    def __init__(self,runnable_list):
        self.runnable_list = runnable_list

    def invoke(self,input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data['response']
    
template = NaklipromptTemplate(
    template="Write a {length} lines poem on the topic {topic}",
    input_variable=['length','topic']
)

llm = NakliLLM()

chain = RunnableConnector([template,llm])

print(chain.invoke({'length':'20','topic':'India'}))
    