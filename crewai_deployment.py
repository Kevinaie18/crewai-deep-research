import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import Tool
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from tinydb import TinyDB

# Initialize LLM model
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Define functions for research tasks
def scrape_webpage(url):
    """Agent function for web scraping."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.text for p in soup.find_all('p')])
    except Exception as e:
        return f"Error during scraping: {str(e)}"

def analyze_market_trends(content):
    """Agent function for market trend analysis."""
    prompt = f"Analyze the following research content and identify key market trends:\n{content}"
    return llm.invoke(prompt)

def competitive_analysis(content):
    """Agent function for competitor analysis."""
    prompt = f"Based on the following content, provide an overview of the competitive landscape:\n{content}"
    return llm.invoke(prompt)

def investment_risks(content):
    """Agent function for risk assessment in investments."""
    prompt = f"Evaluate the potential risks based on the following research:\n{content}"
    return llm.invoke(prompt)

# Define AI agents using LangChain tools
agents = {
    "Scraper Agent": Tool(
        name="Web Scraper",
        func=scrape_webpage,
        description="Retrieves and extracts information from a given webpage."
    ),
    "Market Research Agent": Tool(
        name="Market Research",
        func=analyze_market_trends,
        description="Analyzes research content to identify market trends."
    ),
    "Competitive Intelligence Agent": Tool(
        name="Competitor Analysis",
        func=competitive_analysis,
        description="Provides insights into the competitive landscape based on data."
    ),
    "Risk Analysis Agent": Tool(
        name="Investment Risk Assessment",
        func=investment_risks,
        description="Assesses potential risks in a given research domain."
    )
}

# Initialize LangChain agent framework
agent_executor = initialize_agent(
    list(agents.values()),
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Local database for storing research results
db = TinyDB("research_results.json")

# Streamlit Interface
st.title("🔍 AI-Powered Deep Research")

# User input for URL
url = st.text_input("Enter a URL to analyze:")
task = st.selectbox("Select Research Focus:", list(agents.keys()))

if st.button("Run Analysis"):
    if url:
        st.write(f"🔄 {task} in progress...")

        # Scraping the content
        scraped_content = scrape_webpage(url)
        st.write("📌 Extracted Content:")
        st.write(scraped_content[:500] + "...")

        # Running AI analysis based on selected agent
        st.write(f"🧠 {task} is analyzing the data...")
        result = agent_executor.run(f"{task}: {scraped_content}")

        st.write("📌 Analysis Result:")
        st.write(result)

        # Store results
        db.insert({"url": url, "task": task, "result": result})
    else:
        st.warning("Please enter a valid URL.")
