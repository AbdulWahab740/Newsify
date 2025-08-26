from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from utils.retreive import search
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()
GROK_API_KEY = st.secrets["GROK_API_KEY"]


def create_prompt():
    return PromptTemplate(
        template="""
        You are a news assistant who answers the questions of news about the Tech Industry.
        Use the given context to answer the query accurately.
        Tell everything you know about that And we have to answer in the news telling style
        If unsure, say you don't know politely.
        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )


def setup_llm():
    return ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0.2,
        max_tokens=2000,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
        api_key=GROK_API_KEY
    )

def build_chains(query):
    prompt = create_prompt()
    llm = setup_llm()

    parallel_chain = RunnableParallel({
        "context": RunnableLambda(search),  # ✅ Wrap search function
        "question": RunnablePassthrough()   # ✅ Passes through query
    })
    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser

    return chain.invoke(query)
