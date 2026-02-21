from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

model = ChatOllama(model="llama3.1:8b")

chat_messages=[SystemMessage(content="You are a helpful assistant")]

while True:
    user_input = input(f"You : ")
    chat_messages.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_messages)
    print(f"AI : {result.content}")
    chat_messages.append(AIMessage(content=result.content))

print(chat_messages)
