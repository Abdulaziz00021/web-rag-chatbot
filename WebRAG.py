import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

st.set_page_config("page_title(Web RAG Chatbot")
st.title("Web RAG Ai Assistant")
st.markdown(" Ask anything  e.g., *latest AI trends, stock prices, global news...*")

user_query = st.text_input("Enter your Question")

search_tool = TavilySearchResults(tavily_api_key=tavily_api_key, max_results=5)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

if st.button("Ask the Web") and user_query:
    with st.spinner("searching and generating answer..."):
        results = search_tool.run(user_query)
        context = "\n".join([r["content"] for r in results])
        prompt = f"""
               You are a helpful AI assistant. Use the following web data to answer clearly and factually.
               context:
               {context}
               Question: {user_query}
               
               Instructions:
- write the answer in **paragraph form** (not bullet points or numbered list.
- Be clear, concise, and natural, like a human explaining to another human.
- If the context is not enough, mention it politely and explain based on your general knowledge.
"""
        response = llm.invoke(prompt)
        st.subheader("Answer")
        st.write(response.content)



