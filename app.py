import streamlit as st
from qna import build_chains

st.title("Tech Ups Ai")
st.markdown(
    """
    **Welcome to Tech Ups AI**
    An application that allows you to ask questions about recent tech news.
    The app uses a vector store to retrieve relevant articles and answer your questions.
    """
)
st.subheader("Ask a question about any recent tech news:")
question = st.text_input("Enter your question here:")
if question:
    st.write(f"You asked: {question}")
    response = build_chains(question)
    st.write(f"{response}")

    