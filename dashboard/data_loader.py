# data_loader.py
import requests
import numpy as np
import pandas as pd
import pymongo
from datetime import datetime
from analysis import clean_text_series
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_db():
    return pymongo.MongoClient("mongodb://localhost:27017/")["steamdb"]

@st.cache_data(show_spinner=False)
def get_game_name(app_id: str) -> str:
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        res = requests.get(url, timeout=5)
        return res.json()[str(app_id)]["data"]["name"]
    except Exception:
        return f"App {app_id}"

@st.cache_data(show_spinner=False)
def load_df(collection, _db):
    docs = list(_db[collection].find())
    if not docs: return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["review_text"] = df["review"] if "review" in df.columns else df.get("review_text","")
    if "language" not in df.columns: df["language"] = "unknown"
    if "voted_up" not in df.columns: df["voted_up"] = np.nan

    if "author" in df.columns:
        try:
            pt = df["author"].apply(lambda x: x.get("playtime_forever") if isinstance(x, dict) else None)
            df["playtime_hours"] = (pd.to_numeric(pt, errors="coerce").fillna(0)/60).round(2)
        except Exception:
            pass

    if "timestamp_created" in df.columns:
        df["review_date"] = pd.to_datetime(df["timestamp_created"], unit="s", errors="coerce")
    elif "created" in df.columns:
        df["review_date"] = pd.to_datetime(df["created"], errors="coerce")
    else:
        df["review_date"] = pd.to_datetime(df.get("review_date"), errors="coerce")

    df["cleaned_review"] = clean_text_series(df["review_text"])
    return df
