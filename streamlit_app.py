"""
Ce projet est un fork de LlamaIndex Chat with Streamlit (https://github.com/carolinedlu/llamaindex-chat-with-streamlit-docs/tree/main?ref=blog.streamlit.io), adapté pour une interaction spécifique avec les documents SAPS via OpenAI et LlamaIndex.
"""

import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="BeerCan Chat : prototype sur API-openAI et LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = "your api key"
st.title("Vous pouvez chater avec les document 💬🦙")
st.info("Plus d'informations sur BeerCan : Check (https://beercan.fr/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Vous pouvez m'interroger sur les documents de ma base de données"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Chargement et indexation des données. Cela devrait prendre 1 à 2 minutes maximum."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="Réponds et écris toujours en français. Tu es un expert en vulgarisation scientifique et un journaliste. Ton travail consiste à répondre aux questions qui te sont posées en vulgarisant ou non les réponses que tu apportes. tu peux poser la question de savoir si l'utilisateur veux une réponse plus vulgarisée ou plus experte. Cite l'origine de tes sources d'information. Concentre tes réponses sur des réponses factuelles liées aux documents dont tu disposes. Ne produit pas d'hallucinations ou d'éléments imaginaires."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Votre question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
