import requests
from dotenv import load_dotenv
import os
import json
load_dotenv()
from langchain_ollama  import ChatOllama
from langchain_core.tools import InjectedToolArg
from langchain.tools import tool
from typing import Annotated
from langchain_core.messages import HumanMessage,AIMessage



@tool
def get_conversion_factor(base_currency:str,target_currency:str) -> float:

    """
       This function fetches the currency conversion factor between a given base currency and a target currency
    """
    EXCHANGE_RATE = os.getenv('EXCHANGE_RATE') 
    # tool 1 fetch api 

    if not EXCHANGE_RATE:
        raise ValueError("EXCHANGE_RATE API key not found in environment variables.")

    url = f'https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE}/latest/{target_currency}'

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data["conversion_rates"][base_currency]
    else:
        print("Error:", response.status_code, response.text)
# tool 2 convert 
@tool 
def convert(base_currency_value:int,conversion_rate:Annotated[float,InjectedToolArg])-> float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """
    return base_currency_value * conversion_rate

# llm initialization 

llm =  ChatOllama(model="llama3.1:8b")

# prompt Human
message = [HumanMessage("What is the conversion factor between INR and USD, and based on that can you convert 10 usd to inr")]

# tool binding
llm_with_tools = llm.bind_tools([get_conversion_factor,convert])

# llm call
ai_message = llm_with_tools.invoke(message)

# ai message append
message.append(ai_message)

# print(ai_message.tool_calls)

tools = ai_message.tool_calls


# tool message call
for tool_call in tools:
    if tool_call['name'] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)
        # print(tool_message1)
        conversion_rate = float(tool_message1.content)
        message.append(tool_message1)
    if tool_call['name'] == "convert":
        tool_call['args']['conversion_rate']=conversion_rate
        tool_message2 = convert.invoke(tool_call)
        message.append(tool_message2)


# llm call with tool message

result = llm.invoke(message)
print(result.content)