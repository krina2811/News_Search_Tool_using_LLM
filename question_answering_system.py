import os
import streamlit as st
import time
import langchain
from ingest import build_vectore_store
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain_community.llms import LlamaCpp
import accelerate 
import transformers
import json
from huggingface_hub import hf_hub_download
import llama_cpp
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

if __name__=="__main__":

    file_path = "./vector/news_data.pkl"
    st.title("NewsBot: News Research Tool 📈")
    st.sidebar.title("News Article URLs")
    main_placeholder = st.empty()

    url = "http://localhost:6333"

    client = QdrantClient(
        url=url, prefer_grpc=False
    )

    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    process_url_clicked = st.sidebar.button("Process Urls")

    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

    if process_url_clicked:
        loader = UnstructuredURLLoader(urls)
        data = loader.load()
                                                        
        text_split = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=400,
            chunk_overlap=40
        )
        
        docs = text_split.split_documents(data)
        build_vectore_store(docs, embeddings)


    db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db_cnbc_news")
    retriever = db.as_retriever(search_kwargs={"k":1})


    model_name = "TheBloke/Mistral-7B-OpenOrca-GGUF"
    model_file = "mistral-7b-openorca.Q4_K_M.gguf"
    model_path = hf_hub_download(model_name, filename=model_file)


    model_kwargs = {
    "n_ctx":512,    # Context length to use
    "n_threads":4,   # Number of CPU threads to use
    "n_gpu_layers":15,# Number of model layers to offload to GPU. Set to 0 if only using CPU
    }   

    
    ## Instantiate model from downloaded file
    llm = LlamaCpp(model_path=model_path, **model_kwargs)


    
    ########################### query ######################################################
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:"""    
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    
    query = main_placeholder.text_input("Enter question:")

    if query:
        chain_type_kwargs = {"prompt": prompt}
        start_time = time.time()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
        print("............", qa)
        response = qa(query)
        print(f"\n--- {time.time() - start_time} seconds ---")
        print(response)
        answer = response['result']
        source_document = response['source_documents'][0].page_content
        doc = response['source_documents'][0].metadata['source']
        
    
        st.write(f"Answer: {answer}")
        st.write(f"Source:{doc}")

    

