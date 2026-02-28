import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools import RequestsGetTool
from langchain_experimental.tools import PythonREPLTool

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# --------------------------------------
# Load API Key
# --------------------------------------
load_dotenv()

# --------------------------------------
# Streamlit UI
# --------------------------------------
st.set_page_config(page_title="Research Bot", layout="wide")
st.title("🔎 LangChain Research Assistant")

# --------------------------------------
# Initialize LLM
# --------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# --------------------------------------
# Tool 1 – Web Search
# --------------------------------------
search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="Search the web for up-to-date information."
)

# --------------------------------------
# Tool 2 – Wikipedia
# --------------------------------------
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wiki_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="Search Wikipedia for general knowledge."
)

# --------------------------------------
# Tool 3 – Python Tool
# --------------------------------------
python_tool = PythonREPLTool()

# --------------------------------------
# Tool 4 – Requests Tool
# --------------------------------------
requests_tool = RequestsGetTool()

# --------------------------------------
# Tool 5 – Local Vector Store (RAG)
# --------------------------------------
documents = [
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
    Document(page_content="Retrieval-Augmented Generation improves factual reliability.")
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

rag_tool = Tool(
    name="Local Knowledge Base",
    func=qa_chain.run,
    description="Answer questions from local stored documents."
)

# --------------------------------------
# Memory
# --------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history")

# --------------------------------------
# Combine Tools
# --------------------------------------
tools = [
    search_tool,
    wiki_tool,
    python_tool,
    requests_tool,
    rag_tool
]

# --------------------------------------
# Create Agent
# --------------------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# --------------------------------------
# Chat Interface
# --------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask your research question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.run(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
