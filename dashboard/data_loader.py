# data_loader.py — Mongo + chargement robuste
import os
import requests
import numpy as np
import pandas as pd
import pymongo
import streamlit as st
from analysis import clean_text_series

# -- .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def _build_uri_from_env() -> str:
    running_in_docker = os.getenv("RUNNING_IN_DOCKER", "").lower() == "true"
    uri_docker = os.getenv("MONGO_URI_DOCKER")
    if running_in_docker and uri_docker:
        return uri_docker
    uri = os.getenv("MONGO_URI")
    if uri:
        return uri
    user = os.getenv("MONGO_USER", "steam")
    pwd  = os.getenv("MONGO_PASS", "steam")
    host = os.getenv("MONGO_HOST", "localhost")
    port = os.getenv("MONGO_PORT", "27017")
    dbnm = os.getenv("MONGO_DB", "steamdb")
    return f"mongodb://{user}:{pwd}@{host}:{port}/{dbnm}?authSource={dbnm}"

def _mask(uri: str) -> str:
    try:
        if "@" in uri and "://" in uri:
            head, tail = uri.split("://", 1)
            creds_host = tail.split("@", 1)
            if len(creds_host) == 2 and ":" in creds_host[0]:
                user = creds_host[0].split(":", 1)[0]
                tail = f"{user}:***@{creds_host[1]}"
            return f"{head}://{tail}"
    except Exception:
        pass
    return uri

@st.cache_resource(show_spinner=False)
def get_db():
    uri = _build_uri_from_env()
    print(f"[data_loader.get_db] Using MONGO URI: {_mask(uri)}")
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client.get_default_database()
    if db is None:
        db = client[os.getenv("MONGO_DB", "steamdb")]
    print(f"[data_loader.get_db] Connected DB: {db.name}")
    return db

@st.cache_data(show_spinner=False)
def get_game_name(app_id: str) -> str:
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        res = requests.get(url, timeout=5)
        return res.json()[str(app_id)]["data"]["name"]
    except Exception:
        return f"App {app_id}"

def _first_nonempty_series(df: pd.DataFrame, candidates) -> pd.Series:
    """Retourne la première série non vide parmi les colonnes candidates. Déroule les dicts si besoin."""
    if df is None or df.empty:
        return pd.Series(dtype="object")
    for col in candidates:
        if col in df.columns:
            s = df[col]
            if s.apply(lambda x: isinstance(x, dict)).any():
                s = s.apply(lambda x: (
                    x.get("cleaned_review") or x.get("review") or x.get("text") or x.get("content") or x.get("body")
                ) if isinstance(x, dict) else x)
            nonempty = s.fillna("").astype(str).str.strip().str.len().sum()
            if nonempty > 0:
                return s
    return pd.Series([""] * len(df), index=df.index, dtype="object")

@st.cache_data(show_spinner=False)
def load_df(collection, _db):
    """Chargement générique d'une collection reviews_<appid> (mode per-collection)."""
    docs = list(_db[collection].find())
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)

    # Texte : privilégie le "clean" existant, sinon reconstruit depuis brut
    clean_candidates = ["cleaned_review", "clean_text", "review_clean", "text_clean"]
    raw_candidates   = ["review_text", "review", "content", "text", "body", "reviewBody", "review_text_en"]

    clean_series = _first_nonempty_series(df, clean_candidates)
    raw_series   = _first_nonempty_series(df, raw_candidates)

    if not isinstance(raw_series, pd.Series):
        raw_series = pd.Series([raw_series] * len(df), index=df.index, dtype="object")
    raw_series = raw_series.fillna("").astype(str)

    if isinstance(clean_series, pd.Series) and clean_series.fillna("").astype(str).str.strip().str.len().sum() > 0:
        df["cleaned_review"] = clean_series.fillna("").astype(str)
    else:
        df["cleaned_review"] = clean_text_series(raw_series)

    # Pour l'explorateur
    if raw_series.str.strip().str.len().sum() == 0:
        df["review_text"] = df["cleaned_review"]
    else:
        df["review_text"] = raw_series

    # Champs usuels
    if "language" not in df.columns:
        df["language"] = "unknown"
    if "voted_up" not in df.columns and "votes_up" in df.columns:
        df["voted_up"] = (pd.to_numeric(df["votes_up"], errors="coerce").fillna(0) > 0)
    elif "voted_up" not in df.columns:
        df["voted_up"] = np.nan

    # Dates
    date_fields_priority = ["review_date", "timestamp_created", "created", "posted", "date", "time_created", "timestamp", "timestamp_updated"]
    found = next((f for f in date_fields_priority if f in df.columns), None)
    if found in ["timestamp_created", "timestamp", "time_created"]:
        df["review_date"] = pd.to_datetime(df[found], unit="s", errors="coerce")
    elif found:
        df["review_date"] = pd.to_datetime(df[found], errors="coerce")
    else:
        df["review_date"] = pd.NaT

    # Sentiment calculé si pas présent (per-collection n'a pas toujours 'compound')
    if "compound" not in df.columns:
        df["compound"] = 0.0

    # Colonnes minimales garanties
    for col in ["review_date", "language", "voted_up", "cleaned_review", "review_text", "compound"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df
