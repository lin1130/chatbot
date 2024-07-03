import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

load_dotenv()
secret_key = os.getenv('OPENAI_API_KEY')

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125")

# messages = [
#     SystemMessage(
#         content="""You're an assistant knowledgeable about
#         programming and algorithms. Only answer coding questions."""
#     ),
#     HumanMessage(content="Please provide Python code for Two Sum Question.")
# ]

messages = [
    SystemMessage(
        content="""You're an assistant knowledgeable about 
        healthcare. Only answer healthcare-related questions."""
    ),
    HumanMessage(content="What is Medicaid managed care?"),
]
chat_model.invoke(messages)
print(chat_model)
