import os
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOllama(model="llama3", temperature=0.3)

# Initialize Search Tool
search_tool = TavilySearchResults(max_results=5)

# Streamlit page config
st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("🤖 Research Assistant (Ollama + Tavily)")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful research assistant.")
    ]

# Function to decide if search is needed
def should_search(query):
    keywords = ["latest", "news", "recent", "2025", "2026", "current"]
    return any(word in query.lower() for word in keywords)

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input box
if prompt := st.chat_input("Ask a question..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(HumanMessage(content=prompt))

    # Decide whether to search
    if should_search(prompt):
        search_results = search_tool.invoke(prompt)
        full_prompt = f"""
        Use the following web search results to answer the question:

        {search_results}

        Question: {prompt}
        """
        response = llm.invoke(st.session_state.messages + [HumanMessage(content=full_prompt)])
    else:
        response = llm.invoke(st.session_state.messages)

    answer = response.content

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(AIMessage(content=answer))
