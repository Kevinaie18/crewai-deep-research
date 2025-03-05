import os
import pandas as pd
import streamlit as st
from crewai import Crew, Task, Agent
from langchain.llms import OpenAI
from scrapy import Selector
import requests
from tinydb import TinyDB, Query

# Désactiver ChromaDB pour éviter les erreurs de compatibilité avec SQLite
os.environ["CREWAI_USE_CHROMADB"] = "false"
os.environ["CREWAI_STORAGE_BACKEND"] = "memory"

# --- DATABASE CONFIGURATION ---
DB_FILE = "investment_data.json"
db = TinyDB(DB_FILE)

def save_results(company_name, sector, revenue_currency, risk_assessment, exit_strategy):
    db.insert({
        "company_name": company_name,
        "sector": sector,
        "revenue_currency": revenue_currency,
        "risk_assessment": risk_assessment,
        "exit_strategy": exit_strategy
    })

def display_results():
    df = pd.DataFrame(db.all())
    st.dataframe(df)

# --- AGENTS CONFIGURATION ---
macro_agent = Agent(
    name="Macro Analyst",
    role="Researching macroeconomic trends, currency stability, and regulatory impacts in the target sector.",
    model="gpt-4o"
)

sector_agent = Agent(
    name="Sector Analyst",
    role="Identifying key trends, competitors, and pricing structures in the target industry.",
    model="gpt-4o"
)

scraper_agent = Agent(
    name="Company Identifier",
    role="Scraping and collecting information on potential investment targets.",
    model="gpt-4o"
)

risk_agent = Agent(
    name="Risk Evaluator",
    role="Assessing financial, operational, and exit strategy risks for investment decisions.",
    model="gpt-4o"
)

reporting_agent = Agent(
    name="Investment Analyst",
    role="Compiling findings into structured reports.",
    model="gpt-4o"
)

# --- TASKS CONFIGURATION ---
macro_task = Task(
    description="Analyze macroeconomic stability, inflation hedge mechanisms, and WAEMU FX regulations.",
    agent=macro_agent
)

sector_task = Task(
    description="Assess key industry trends, market growth rates, pricing power, and regulatory compliance.",
    agent=sector_agent
)

company_task = Task(
    description="Identify SMEs matching investment criteria using web scraping and data aggregation.",
    agent=scraper_agent
)

risk_task = Task(
    description="Evaluate financial risks, compliance factors, and propose exit strategies.",
    agent=risk_agent
)

reporting_task = Task(
    description="Compile findings into a structured investment report, including scalability, ESG impact, and recommended investment rationale.",
    agent=reporting_agent
)

# --- CREW ASSEMBLY & EXECUTION ---
investment_crew = Crew(
    agents=[macro_agent, sector_agent, scraper_agent, risk_agent, reporting_agent],
    tasks=[macro_task, sector_task, company_task, risk_task, reporting_task]
)

# Execute the Crew
investment_crew.kickoff()

# --- STREAMLIT DASHBOARD ---
st.title("Investment Opportunities Dashboard")
if st.button("Show Results"):
    display_results()
