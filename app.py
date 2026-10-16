"""Entry point: `streamlit run app.py`. Defines navigation between the three pages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="Airline Passenger Satisfaction", page_icon=":material/flight:", layout="wide")

pages = [
    st.Page("pages/1_Project_Overview.py", title="Project Overview", icon=":material/dashboard:", default=True),
    st.Page("pages/2_Prediction_Center.py", title="Prediction Center", icon=":material/psychology:"),
    st.Page("pages/3_Analytics_Insights.py", title="Analytics & Insights", icon=":material/insights:"),
]
st.navigation(pages).run()
