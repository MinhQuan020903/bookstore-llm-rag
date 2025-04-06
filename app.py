import streamlit as st
import os
from dotenv import load_dotenv

# Load the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Manually set environment variables if needed
# os.environ["PINECONE_API"] = "YOUR_API_KEY"
# os.environ["PINECONE_ENV"] = "YOUR_ENV"

# Debug print
print(f".env file exists: {os.path.exists(dotenv_path)}")
print(f"PINECONE_API in env: {'PINECONE_API' in os.environ}")

from modules.RAG import RAG


with st.sidebar:
    user_id = st.text_input("User ID", value="4614")

st.title("💬 Chatbot")
st.caption("🚀 LLM Recommender System Chatbot")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    recommendation_agent = RAG(user_id=user_id)

    response = recommendation_agent.agent(prompt)
    print("REPONSE ===>", response)
    print("KEYS ===>", response.keys())
    
    msg = {"role": "assistant", "content": response["output"]}

    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])