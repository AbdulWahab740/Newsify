import feedparser
from newspaper import Article, ArticleException
import datetime
from datetime import timedelta
import logging
import time
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def fetch_articles_from_rss(rss_url: str, days_ago: int = 7):
    articles = []
    feed = feedparser.parse(rss_url)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    date_threshold = today - timedelta(days=days_ago)

    for entry in feed.entries:
        try:
            publish_date_str = entry.get('published') or entry.get('updated', '')
            if publish_date_str:
                try:
                    publish_date = datetime.datetime.strptime(publish_date_str, "%a, %d %b %Y %H:%M:%S %z").date()
                except ValueError:
                    try:
                        publish_date = datetime.datetime.fromisoformat(publish_date_str.replace('Z', '+00:00')).date()
                    except ValueError:
                        publish_date = datetime.datetime.strptime(publish_date_str.split('T')[0], "%Y-%m-%d").date()
            else:
                publish_date = today

            if publish_date < date_threshold:
                continue

            articles.append({
                "title": entry.get('title', 'No Title'),
                "url": entry.get('link', ''),
                "publish_date": publish_date,
                "summary": entry.get('summary', '')
            })
        except Exception as e:
            logging.warning(f"Failed parsing RSS entry: {e}")
    return articles

def extract_full_text_with_newspaper(article_url: str) -> str:
    try:
        article = Article(article_url, language='en')
        article.download()
        article.parse()
        return article.text
    except Exception:
        return ""

def get_tech_news_documents(days_back: int = 7):
    tech_rss_feeds = [
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://arstechnica.com/feed/",
    ]

    all_raw_articles = []
    for feed_url in tech_rss_feeds:
        all_raw_articles.extend(fetch_articles_from_rss(feed_url, days_back))
        time.sleep(0.5)

    processed_documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for article in all_raw_articles:
        content = article["summary"]
        if len(content) < 200 or "read more" in content.lower():
            full_content = extract_full_text_with_newspaper(article["url"])
            if full_content:
                content = full_content
            elif not content.strip():
                continue

        chunks = splitter.split_text(content)
        for chunk in chunks:
            processed_documents.append(Document(
                page_content=chunk,
                metadata={
                    "title": article["title"],
                    "url": article["url"],
                    "publish_date": article["publish_date"].isoformat(),
                    "category": "Tech"
                }
            ))
    return processed_documents

def build_faiss_vectorstore(documents):
    vectorstore = FAISS.from_documents(documents, embedding_function)
    vectorstore.save_local("faiss_vectorstore")
    return vectorstore

def create_prompt():
    return PromptTemplate(
        template="""
        You are a helpful assistant who gives updates of the Tech Industry in an engaging way.
        Use the given context to answer the query accurately.
        If unsure, say you don't know politely.
        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

def build_chains(vectorstore, prompt, llm):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser

    return chain, retriever

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

def answer_question(question):
    if not question:
        return "Please ask a question about recent tech news."
    using_pre = False
    if not os.path.exists("faiss_vectorstore"):
        
        documents = get_tech_news_documents(days_back=7)
        vectorstore = build_faiss_vectorstore(documents)
    else:
        using_pre = True
        vectorstore = FAISS.load_local("faiss_vectorstore", embedding_function, allow_dangerous_deserialization=True)
        # Get all publish_dates from stored documents
        all_dates = [
            datetime.datetime.fromisoformat(doc.metadata['publish_date']).date()
            for doc in vectorstore.docstore._dict.values()
            if 'publish_date' in doc.metadata
        ]
        last_date = max(all_dates) if all_dates else datetime.datetime.now(datetime.timezone.utc).date()
        today = datetime.datetime.now(datetime.timezone.utc).date()
        if (today - last_date).days > 7:
            st.success("Updating vectorstore with new tech news articles...")
            # want to send a message to the user that the vectorstore is being updated
            documents = get_tech_news_documents(days_back=7)
            vectorstore.add_documents(documents)
            vectorstore.save_local("faiss_vectorstore")
    
    prompt = create_prompt()
    llm = setup_llm()
    chain, _ = build_chains(vectorstore, prompt, llm)
    return chain.invoke(question)
