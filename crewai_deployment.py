import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from tinydb import TinyDB, Query

# Initialisation de l'agent LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Fonction de scraping web
def scrape_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.text for p in soup.find_all('p')])
    except Exception as e:
        return f"Erreur lors du scraping : {str(e)}"

# Définition des outils pour l'agent
tools = [
    Tool(
        name="Web Scraper",
        func=scrape_webpage,
        description="Utilise cette fonction pour récupérer des informations depuis un site web."
    )
]

# Initialisation de l'agent LangChain
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Base de données locale avec TinyDB pour stocker les résultats
db = TinyDB("research_results.json")

# Interface Streamlit
st.title("🔍 Deep Research avec LangChain Agents")

# Entrée de l'URL
url = st.text_input("Entrez une URL à analyser :")

if st.button("Lancer l'analyse"):
    if url:
        st.write("🔄 Scraping en cours...")
        scraped_content = scrape_webpage(url)
        
        st.write("📌 Contenu extrait :")
        st.write(scraped_content[:500] + "...")  # Afficher un extrait

        st.write("🧠 Analyse et résumé en cours...")
        result = agent.run(f"Donne-moi un résumé du contenu suivant : {scraped_content}")

        st.write("📌 Résumé :")
        st.write(result)

        # Stocker les résultats dans la base locale
        db.insert({"url": url, "résumé": result})
    else:
        st.warning("Veuillez entrer une URL valide.")
