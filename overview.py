import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    PromptTemplate, SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough

review_chroma_path = "chroma_data/"
review_vector_db = Chroma(
    persist_directory=review_chroma_path,
    embedding_function=OpenAIEmbeddings()
)
review_retriever = review_vector_db.as_retriever(k=10)


load_dotenv()
secret_key = os.getenv('OPENAI_API_KEY')
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
# messages = [
#     SystemMessage(
#         content="""You're an assistant knowledgeable about
#         programming and algorithms. Only answer coding questions."""
#     ),
#     HumanMessage(content="Please provide Python code for Two Sum Question.")
# ]

# Step 1
# Chat Models
messages = [
    SystemMessage(
        content="""You're an assistant knowledgeable about 
        healthcare. Only answer healthcare-related questions."""
    ),
    HumanMessage(content="What is Medicaid managed care?"),
]
chat_model.invoke(messages)

# Prompt Templates
review_template_str = """
Your job is to use patient reviews to answer questions about their experience at a hospital.
Use the following context to answer questions. Be as detailed as possible,
but don't make up any information that's not from the context. If you don't know the answer,
say you don't know.

{context}

{question}
"""
review_template = ChatPromptTemplate.from_template(review_template_str)
context = "I had a great stay!"
question = "Did anyone have a positive experience?"
review_template.format(context=context, question=question)
# LangChain Expression Language (LCEL)
review_system_template_str = """
Your job is to use patient reviews to answer questions about their experience at a hospital. 
Use the following context to answer questions. Be as detailed as possible, 
but don't make up any information that's not from the context. If you don't know an answer, 
say you don't know.

{context}
"""
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=review_system_template_str
    )
)
review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"], template="{question}"
    )
)
messages = [review_system_prompt, review_human_prompt]
review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)
# message = review_prompt_template.format_messages(context=context, question=question)
# chain together and format the model's response
output_parser = StrOutputParser()
review_chain = review_prompt_template | chat_model | output_parser
review_chain.invoke({"context": context, "question": question})

