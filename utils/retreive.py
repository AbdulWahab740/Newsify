from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from utils.vector_store import build_faiss_vectorstore, process_articles_for_vectorstore
from utils.scrape import get_news
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def hybrid_score(bm25_scores, emb_scores, alpha=0.5):
    """
    alpha = weight for embeddings (0.5 = equal blend, tune as needed)
    """
    scaler = MinMaxScaler()
    
    bm25_scaled = scaler.fit_transform(np.array(bm25_scores).reshape(-1,1)).flatten()
    emb_scaled = scaler.fit_transform(np.array(emb_scores).reshape(-1,1)).flatten()
    
    final_scores = alpha * emb_scaled + (1 - alpha) * bm25_scaled
    return final_scores

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def hybrid_retreive(documents):
    # Build FAISS vectorstore
    vectorstore = build_faiss_vectorstore(documents)
    if not vectorstore:   # ✅ no docs case
        return None

    # Create retrievers
    bm25_retriever = BM25Retriever.from_documents(documents)   # keyword search
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # semantic search

    # Combine them into Hybrid
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]   # adjust balance BM25 vs FAISS
    )

    return hybrid_retriever

# def search(query):
#     results = get_news(query).get("articles", [])
#     print(results)
#     if not results:
#         print("[ERROR] No articles found.")
#         return
#     documents = process_articles_for_vectorstore(results)
#     print(documents)
#     hybrid = hybrid_retreive(documents)
#     if not hybrid:
#         print("[ERROR] No retriever built (no docs available).")
#         return

#     results = hybrid.invoke(query)
#         # Vectorize query + titles
#     vectorizer = TfidfVectorizer()
#     titles = [result.metadata.get("title", "") for result in results]
#     tfidf_matrix = vectorizer.fit_transform([query] + titles)

#     # Compute similarity of query (index 0) with each title
#     cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

#     # Sort results
#     ranked = sorted(zip(titles, cos_sim), key=lambda x: x[1], reverse=True)

#     for title, score in ranked:
#         print(f"{title} | Score: {score:.4f}")
#     # for doc in results:
#     #     print(f"Title: {doc.metadata.get('title')} | Publisher: {doc.metadata.get('publisher')}")

from langchain_core.vectorstores.utils import maximal_marginal_relevance

def search(query, alpha=0.5, k=10):
    results = get_news(query).get("articles", [])
    if not results:
        print("[ERROR] No articles found.")
        return

    # Process articles -> LangChain Documents
    documents = process_articles_for_vectorstore(results)
    if not documents:
        print("[ERROR] No documents created from articles.")
        return

    # Build Hybrid Retriever
    hybrid = hybrid_retreive(documents)
    if not hybrid:
        print("[ERROR] No retriever built (no docs available).")
        return

    # Initial retrieval
    retrieved_docs = hybrid.invoke(query)

    # --- Step 1: Compute TF-IDF cosine similarity (extra lexical score) ---
    vectorizer = TfidfVectorizer()
    titles = [doc.metadata.get("title", "") for doc in retrieved_docs]
    tfidf_matrix = vectorizer.fit_transform([query] + titles)
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # --- Step 2: Normalize both retriever scores + TF-IDF ---
    bm25_scores = [doc.metadata.get("bm25_score", 1.0) for doc in retrieved_docs]
    emb_scores = [doc.metadata.get("vector_score", 1.0) for doc in retrieved_docs]

    final_scores = hybrid_score(bm25_scores, emb_scores, alpha=alpha)
    rerank_scores = hybrid_score(final_scores, cos_sim, alpha=0.5)

    # --- Step 3: Sort + Deduplicate by URL ---
    ranked_docs = sorted(
        zip(retrieved_docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    seen_urls = set()
    unique_ranked = []
    for doc, score in ranked_docs:
        url = doc.metadata.get("url")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_ranked.append((doc, score))
        if len(unique_ranked) >= k:
            break

    # --- Step 4: Apply Maximal Marginal Relevance (MMR) Diversification ---
    embeddings = [embedding_function.embed_query(doc.page_content) for doc, _ in unique_ranked]
    mmr_indices = maximal_marginal_relevance(
        np.array(embedding_function.embed_query(query)), embeddings, k=min(10, len(unique_ranked))
    )

    diversified_results = [unique_ranked[i] for i in mmr_indices]

    final_context = []
    for doc, score in diversified_results:
        context_piece = f"""
        Title: {doc.metadata.get('title')}
        Publisher: {doc.metadata.get('publisher')}
        Date: {doc.metadata.get('published_date')}
        URL: {doc.metadata.get('url')}
    Summary: {doc.page_content[:500]}...   # take first 500 chars to avoid context overflow
    """
        final_context.append(context_piece)

    context_str = "\n\n".join(final_context)

    return context_str
