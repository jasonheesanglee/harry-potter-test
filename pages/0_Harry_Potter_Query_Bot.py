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

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)
# LOGGER.info(f'sqlite version: {sqlite3.sqlite_version}')
# LOGGER.info(f'sys version: {sys.version}')

import os
import glob
import torch
import tiktoken
import chromadb
import streamlit as st

from langchain.chains import LLMChain, RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader

from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
files = glob.glob('../harry_potter/*.txt')

os.environ["OPENAI_API_KEY"] = st.secrets['OpenAI_API']
os.environ["HF_AUTH_TOKEN"] = st.secrets['HF_TOKEN']

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

model_name = 'meta-llama/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')#quantization_config=bnb_config)
# accelerator = Accelerator()
# device = accelerator.device
# set_seed(42)
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    temperature=0.2,
    return_full_text=True,
    max_new_tokens=500,
)
prompt_template = '''
### [INST]
Instruction: Answer the question based on your knowledge from the book.
Here is context to help:

{context}

### Question
{question}
[/INST]
'''


model_name = 'BAAI/bge-small-en'
model_kwargs = {'device': 'auto'}
encode_kwargs = {'normalize_embeddings': True}

llama2 = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
llm_chain = LLMChain(llm=llama2, prompt=prompt)

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
        persist_directory="../Harry_Potter_Chroma_DB",
    )
    vectordb.persist()
    return vectordb

def load_persisted_chroma(directory: str) -> Chroma:
    vectordb = Chroma(persist_directory=directory, embedding_function=hf)
    return vectordb
# db = load_chunk_persist_text('./harry_potter')
db = load_persisted_chroma('../Harry_Potter_Chroma_DB')

# openai = ChatOpenAI(model_name='gpt-3.5-turbo',
#                     streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
#                     temperature=0)

retriever = db.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3, 'fetch_k':10}
)

rag_chain = (
    {'context': retriever, 'question':RunnablePassthrough()}
    | llm_chain
)

qa = RetrievalQA.from_chain_type(
    llm = llama2,
    chain_type='stuff',
    retriever=db.as_retriever(
        search_type='mmr',
        search_kwargs={'k':3, 'fetch_k':10}
    ),
    return_source_documents=True
    )


def create_agent_chain():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    llm = llama2
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

# def create_agent_chain():
#     model_name = "gpt-3.5-turbo"
#     llm = ChatOpenAI(model_name=model_name)
#     chain = load_qa_chain(llm, chain_type="stuff")
#     return chain

def get_llm_response(query):
    vectordb = load_persisted_chroma('../Harry_Potter_Chroma_DB')
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
