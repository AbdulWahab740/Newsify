import requests
import streamlit as st

APIKEY = st.secrets["APIKEY"]
def get_news(query):
    if (len(query.split(" "))> 1):
        query = "+".join(query.split(" "))
    print(query)
    response = requests.get(f"https://gnews.io/api/v4/search?q={query}&lang=en&max=5&apikey={APIKEY}")
    response.raise_for_status()
    data = response.json()
    return data
