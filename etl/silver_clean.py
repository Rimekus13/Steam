
import pandas as pd
from .mongo_utils import col_raw, bulk_upsert_clean
from .text_utils import clean_text, detect_lang, sentiment_scores

def _read_raw_all(app_id: str, for_airflow: bool = False) -> pd.DataFrame:
    cur = col_raw(app_id, for_airflow).find({}, {"_id": 0})
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df
    df = df.rename(columns={"recommendationid": "review_id", "review": "review_text"})
    return df

def to_silver(app_id: str, dt: str, for_airflow: bool = False) -> str:
    df = _read_raw_all(app_id, for_airflow=for_airflow)
    if df.empty:
        return dt

    for c in ["timestamp_created","timestamp_updated"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["review_date"] = pd.to_datetime(df["timestamp_created"], unit="s", utc=True).dt.date.astype(str)

    df["cleaned_review"] = df["review_text"].fillna("").astype(str).map(clean_text)

    df["language"] = df["language"].fillna("")
    mask_lang = df["language"].eq("") | df["language"].eq("unknown")
    df.loc[mask_lang, "language"] = df.loc[mask_lang, "cleaned_review"].map(detect_lang)

    sents = df["cleaned_review"].map(sentiment_scores)
    df["compound"] = [s["compound"] for s in sents]
    df["sentiment"] = pd.cut(df["compound"], bins=[-1.0, -0.05, 0.05, 1.0], labels=["neg","neu","pos"], include_lowest=True)

    keep = ["app_id","review_id","author","review_date","language","voted_up","votes_up","votes_funny","cleaned_review","compound","sentiment","timestamp_created","timestamp_updated"]
    if "app_id" not in df.columns:
        df["app_id"] = app_id
    rows = df[keep].to_dict("records")

    bulk_upsert_clean(rows, for_airflow=for_airflow)
    return dt
