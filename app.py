import pandas as pd
import numpy as np

from dotenv import load_dotenv, find_dotenv

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import GigaChat
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

st.set_page_config(
    page_title="ИнтеллектУм",
    page_icon="images/logo.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)


#setting model, embeddings, messages and vector db
if "gigachat" not in st.session_state:
    st.session_state["gigachat"] = GigaChat(
        model="GigaChat",
        verify_ssl_certs=False
    )

    st.session_state["embeddings"] = GigaChatEmbeddings(verify_ssl_certs=False)
    
if "pinecone_db" not in st.session_state:

    
    load_custom_document = False
    index_name = "obshestvo-opredelenya-1"
    
    if load_custom_document:
        loader = TextLoader(index_name+".txt", encoding="utf8")
    
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=10)
        docs = text_splitter.split_documents(documents)
        
        pinecone_db = PineconeVectorStore.from_documents(docs, st.session_state["embeddings"], index_name=index_name)
        
    else:
        
        pinecone_db = PineconeVectorStore.from_existing_index(index_name, embedding=st.session_state["embeddings"])

    st.session_state["pinecone_db"] = pinecone_db


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "prompts_array" not in st.session_state:

    system_prompt = (
        "Тебя зовут ИнтеллектУм, ты - умный чат-бот для помощи ученикам в их обучении"
        "Используй данный тебе контекст для ответа на вопрос"
        "Если ты незнаешь ответ, не отвечай"
        "Контекст: {context}"
    )
    
    st.session_state["prompts_array"] = [
            ("system", system_prompt),
            ("human", "{input}"),
        ]



def gigachat_answer_generator(prompt, human_input):

    qa_chain = create_stuff_documents_chain(st.session_state["gigachat"], prompt)

    rag_chain = create_retrieval_chain(st.session_state["pinecone_db"].as_retriever(), qa_chain)

    final_chain = rag_chain.pick("answer")
    
    for chunk in final_chain.stream({"input" : human_input}):
        print(chunk)
        yield chunk


#APP
# st.image("img", unsafe_allow_html=True)
st.image("images/logo.png", width=100)
# st.markdown("<img src=Desktop/проекты/Большая_Перемена/кейс/question_answering/logo.png />", unsafe_allow_html=True)
st.header(":green-background[Задай свой вопрос по обществознанию]")


for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(f":violet-background[{message['content']}]" if message['role'] == "human" else f":grey-background[{message['content']}]")
        

if user_prompt := st.chat_input("Спроси что-нибудь, а я помогу"):
    st.session_state.messages.append({"role": "human", "avatar":"images/student_logo.png", "content": user_prompt})
    with st.chat_message("human", avatar="images/student_logo.png"):
        st.markdown(f":violet-background[{user_prompt}]")

    cur_prompt = ChatPromptTemplate.from_messages(st.session_state["prompts_array"])

    with st.chat_message("assistant", avatar="images/chat_assistant_logo.png"):
        with st.spinner(''):
            stream = gigachat_answer_generator(cur_prompt, user_prompt)
            response = st.write_stream(stream)

    st.session_state["prompts_array"] = st.session_state["prompts_array"][:-1]
    st.session_state["prompts_array"].append(("human", user_prompt))
    st.session_state["prompts_array"].append(("ai", response))
    st.session_state["prompts_array"].append(("human", "{input}"))
    
    st.session_state.messages.append({"role": "assistant", "avatar":"images/chat_assistant_logo.png",  "content": response})