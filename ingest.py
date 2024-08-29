from langchain.vectorstores import Qdrant
import streamlit as st
import os
import pickle

url = "http://localhost:6333"

def build_vectore_store(texts, embeddings):
    """
    Store the embedding vectors of text chunks into vector store (Qdrant).
    """
    if texts:       
        with st.spinner("Loading PDF ..."):           
            doc_store = Qdrant.from_documents(
                                        texts,
                                        embeddings,
                                        url=url,
                                        prefer_grpc=False,
                                        collection_name="vector_db_cnbc_news"
                                    )
        st.success("File Loaded Successfully!!")
        
    else:
        doc_store = None

