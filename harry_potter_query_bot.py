# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

import os
import glob
import torch
import tiktoken
import chromadb
import streamlit as st

from accelerate.utils import set_seed
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from accelerate import Accelerator, notebook_launcher
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["OPENAI_API_KEY"] = os.getenv('OpenAI_API')
os.environ["HF_AUTH_TOKEN"] = os.getenv('HF_TOKEN')

accelerator = Accelerator()
device = accelerator.device
set_seed(42)

files = glob.glob('./harry_potter/*.txt')

model_name = 'BAAI/bge-small-en'
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def load_chunk_persist_text(path) -> Chroma:
    documents = []
    for file in os.listdir(path):
        if file.endswith('.txt'):
            txt_path = os.path.join(path, file)
            loader = TextLoader(txt_path, encoding='iso-8859-1')
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=10,
                                                   length_function=tiktoken_len
                                                   )
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=hf,
        persist_directory="./Harry_Potter_Chroma_DB",
    )
    vectordb.persist()
    return vectordb

def load_persisted_chroma(directory: str) -> Chroma:
    vectordb = Chroma(persist_directory=directory, embedding_function=hf)
    return vectordb

db = load_persisted_chroma('./Harry_Potter_Chroma_DB')

openai = ChatOpenAI(model_name='gpt-3.5-turbo',
                    streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                    temperature=0)

qa = RetrievalQA.from_chain_type(
    llm = openai,
    chain_type='stuff',
    retriever=db.as_retriever(
        search_type='mmr',
        search_kwargs={'k':3, 'fetch_k':10}
    ),
    return_source_documents=True
    )

def create_agent_chain():
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def create_agent_chain():
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_llm_response(query):
    vectordb = load_persisted_chroma('./Harry_Potter_Chroma_DB')
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer

def run():
    st.set_page_config(page_title="Harry Potter Searcher", page_icon=":dizzy:")
    st.header("Query Text Source")

    form_input = st.text_input('Type anything you want to know from the Harry Potter Series')
    submit = st.button("Generate")

    if submit:
        st.write(get_llm_response(form_input))

if __name__ == "__main__":
    run()
