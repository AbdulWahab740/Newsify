import streamlit as st
from utils.qna import build_chains

st.title("Newsify")
st.markdown(
    """
    **Welcome to Newsify!**
    A application that allows you to ask questions about recent news.
    The app uses a vector store to retrieve relevant scraped articles and answer your questions.
    """
)
st.subheader("Ask a question about any recent news: ")
question = st.text_input("Enter your question here:")
if question:
    st.markdown(f"You asked: {question}")
    response = build_chains(question)
    st.write(f"{response}")

    
