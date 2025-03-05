import os
os.environ["CREWAI_USE_CHROMADB"] = "false"
import sqlite3
import pandas as pd
import streamlit as st
from crewai import Crew, Task, Agent
from langchain.llms import OpenAI
from scrapy import Selector
import requests

# --- DATABASE CONFIGURATION ---
DB_FILE = "investment_data.db"

def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS investment_opportunities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_name TEXT,
                        sector TEXT,
                        revenue_currency TEXT,
                        risk_assessment TEXT,
                        exit_strategy TEXT)''')
    conn.commit()
    conn.close()

initialize_database()

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

def save_results(company_name, sector, revenue_currency, risk_assessment, exit_strategy):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO investment_opportunities (company_name, sector, revenue_currency, risk_assessment, exit_strategy) VALUES (?, ?, ?, ?, ?)",
                   (company_name, sector, revenue_currency, risk_assessment, exit_strategy))
    conn.commit()
    conn.close()

def display_results():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM investment_opportunities", conn)
    conn.close()
    st.dataframe(df)

# Execute the Crew
investment_crew.kickoff()

# --- STREAMLIT DASHBOARD ---
st.title("Investment Opportunities Dashboard")
if st.button("Show Results"):
    display_results()
