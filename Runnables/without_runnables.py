import random

class NakliLLM:
    def __init__(self):
        print("LLM is created")
    
    def predict(self,prompt):
        response_list = [
            "Delhi is the capital of India",
            "IPL is cricket league",
            "ML stands for Machine Learning"
        ]

        return {'response':random.choice(response_list)}
        
class NaklipromptTemplate:

    def __init__(self,template,input_variable):
        self.template = template
        self.input_variable = input_variable
    
    def format(self,input_dict):
        return self.template.format(**input_dict)


class NakliChain:
    def __init__(self,llm,prompt):
        self.llm = llm
        self.prompt = prompt
    
    def run(self,input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']
    
llm = NakliLLM()
template = NaklipromptTemplate(
    template="Write a poem about {topic}",
    input_variable=['topic']

)

# print(llm.predict(template.format({'topic':'India'})))

# # print(template.format({'topic':'India'}))

chain = NakliChain(llm,template)
print(chain.run({'topic':'India'}))