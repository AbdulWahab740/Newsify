from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from utils.data_fetch import extract_full_text_with_gnews


embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_faiss_vectorstore(documents):
    if not documents:   # ✅ Avoids FAISS crash
        print("[WARN] No documents to build vectorstore")
        return None  

    vectorstore = FAISS.from_documents(documents, embedding_function)
    vectorstore.save_local("faiss_vectorstore")
    return vectorstore

def process_articles_for_vectorstore(results: List[dict]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=200
    )
    documents = []

    for result in results:
        article_text, metadata = extract_full_text_with_gnews(result)
        if article_text:
            chunks = text_splitter.split_text(article_text)
            for chunk in chunks:
                documents.append(
                Document(page_content=chunk,metadata=metadata.model_dump())
            )

    return documents

