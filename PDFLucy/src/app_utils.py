import pandas as pd
import streamlit as st
import os
import tiktoken
import numpy as np
import time
import re
import math

#sklearn cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

from config import MODELS, TEMPERATURE, MAX_TOKENS, EMBEDDING_MODELS, PROCESSED_DOCUMENTS_DIR, REPORTS_DOCUMENTS_DIR, STOP_WORD_LIST

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Import the necessary modules to work with the APIs
###### FEEL FREE TO COMMENT OUT PACKAGES YOU DON'T NEED ######
from anthropic import Anthropic
import google.generativeai as genai
import openai

anth_client=None
if os.getenv('ANTHROPIC_API_KEY'):
    anth_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
if os.getenv('GOOGLE_GEMINI_API_KEY'):
    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
if os.getenv('OPENAI_API_KEY'):
    openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_BASE'):
    openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_TYPE'):
    openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_VERSION'):
    openai.api_version = os.getenv('OPENAI_API_VERSION')


def initialize_session_state():
    """ Initialise all session state variables with defaults """
    SESSION_DEFAULTS = {
        "cleared_responses" : False,
        "generated_responses" : False,
        "chat_history": [],
        "uploaded_file": None,
        "generation_model": MODELS[0],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [],
        "embedding_model": EMBEDDING_MODELS[0],
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

# USE THIS EMBEDDING FUNCTION THROUGHOUT THIS FILE
# Will download the model the first time it runs
embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELS[0],
        cache_folder="../models/sentencetransformers"
    )  

bm25_retriever = None

class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75, k=1):
        self.docs = docs
        self.corpus = [[word for word in text.lower().split() if word not in STOP_WORD_LIST] for text in docs]
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in self.corpus ) / len(self.corpus )
        self.doc_freqs = self._calculate_doc_freqs()
        self.idf = self._calculate_idf()
        self.doc_lengths = [len(doc) for doc in self.corpus ]
        self.k = k # this k is the number of documents to return

    def _calculate_doc_freqs(self) -> dict[str, int]:
        doc_freqs = {}
        for doc in self.corpus:
            for term in set(doc):
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        return doc_freqs

    def _calculate_idf(self) -> dict[str, float]:
        idf = {}
        num_docs = len(self.corpus)
        for term, freq in self.doc_freqs.items():
            idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def _score_document(self, query: list[str], doc_index: int) -> float:
        score = 0
        doc = self.corpus[doc_index]
        doc_len = self.doc_lengths[doc_index]

        for term in query:
            if term not in doc:
                continue
            tf = doc.count(term)
            numerator = self.idf[term] * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += numerator / denominator

        return score

    def get_relevant_documents(self, query: str):
        #return the top k documents ... currently just returning the first k documents (change this)
        top_docs = self.docs[:self.k]
        query_keywords = [word for word in query.lower().split() if word not in STOP_WORD_LIST]

        ####### YOUR CODE HERE ########
        ####### YOUR CODE HERE ########
        ####### YOUR CODE HERE ########

        return top_docs

def get_retriever():
    k=1
    global bm25_retriever
    if st.session_state["embedding_model"]=="BM25":
        bm25_retriever.k = k
        return bm25_retriever
    db = FAISS.load_local("../data/faiss-db/", embedding_function, allow_dangerous_deserialization=True)
    retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": k})
    return retriever


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_response(prompt, model, system_prompt="", temperature=0, second_try=False):
    """Generate a response from a given prompt and model."""
    
    response = "No model selected"
    if second_try:
        time.sleep(3)

    #### Note: if system_prompt is not relevant for LLM just add it as a prefix to the prompt
    try:
        if model.startswith("Google: "):
            model_name = model[8:]
            response = "Not implemented yet."
            ####### YOUR CODE HERE ########
        elif model.startswith("OpenAI: "):
            model_name = model[8:]
            response = "Not implemented yet."
            ####### YOUR CODE HERE ########
        elif model.startswith("Anthropic: "):
            model_name = model[11:]
            response = "Not implemented yet."
            ####### YOUR CODE HERE ########
        else:
            response = "Not implemented yet."
        ###### FEEL FREE TO USE OTHER LLMs #########
    except Exception as e:
        st.warning(f"{model} API call failed. Waiting 3 seconds and trying again.")
        response = generate_response(prompt, model, system_prompt, temperature, second_try=True)

    # return only the response string
    return response


def create_knowledge_base(docs):
    """Create knowledge base for chatbot."""

    print(f"Loading {PROCESSED_DOCUMENTS_DIR}")
    docs_orig = docs
        
    print(f"Splitting {len(docs_orig)} documents")

    ###### ADD YOUR CODE HERE ######
    ###### ADD YOUR CODE HERE ######
    ###### ADD YOUR CODE HERE ######
    ###### ADD YOUR CODE HERE ######
        
    print(f"Created {len(docs)} documents")

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    if st.session_state["embedding_model"]=="BM25":
        global bm25_retriever
        bm25_retriever = BM25(texts)
        return

    print("""
        Computing embedding vectors and building FAISS db.
        WARNING: This may take a long time. You may want to increase the number of CPU's in your noteboook.
        """
    )
    db = FAISS.from_texts(texts, embedding_function, metadatas=metadatas)  
    # Save the FAISS db 
    db.save_local("../data/faiss-db/")

    print(f"FAISS VectorDB has {db.index.ntotal} documents")
    

def generate_kb_response(prompt, model, retriever, system_prompt="",template=None, temperature=0):

    relevant_docs = retriever.get_relevant_documents(prompt)

    # string together the relevant documents
    relevant_docs_str = ""
    for doc in relevant_docs:
        relevant_docs_str += doc.page_content + "\n\n"

    print(f"Prompt: {prompt}")
    print(f"Context: {relevant_docs_str}")

    if template is None:
        prompt_full = f"""Answer based on the following context

        {relevant_docs_str}

        Question: {prompt}"""
    else:
        prompt_full = template.format(prompt=prompt, context=relevant_docs_str)

    response = generate_response(prompt_full, model=model, system_prompt=system_prompt, temperature=temperature)
    
    return response

