import pandas as pd
import numpy as np

POS, NEG = 0.05, -0.05

def get_vader():
    # Real implementation would import nltk SentimentIntensityAnalyzer
    # but tests patch this function, so leaving light.
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

def compute_sentiment(text: str) -> float:
    sia = get_vader()
    text = text if isinstance(text, str) else ""
    return float(sia.polarity_scores(text).get("compound", 0.0))

def classify_sentiment(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        return "neutral"
    if s > POS:
        return "positive"
    if s < NEG:
        return "negative"
    return "neutral"

def moving_avg(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean().bfill()


def detect_anomalies_zscore(df: pd.DataFrame, value_col: str, z: float = 2.0):
    vals = df[value_col].astype(float)
    mu = vals.mean()
    sigma = vals.std(ddof=0) or 1e-9
    zscores = (vals - mu) / sigma

    out = df.copy()

    # Mask NumPy -> liste -> bool Python natif
    mask_np = (zscores.abs() >= z).to_numpy()          # array de np.bool_
    mask_py = [bool(v) for v in mask_np.tolist()]      # liste de True/False (type bool Python)

    out["is_anomaly"] = mask_py                         # dtype=object, valeurs bool Python
    return out



def before_after_delta(df: pd.DataFrame, date_col: str, value_col: str, pivot_date) -> dict:
    s_before = df[df[date_col] < pivot_date][value_col].astype(float)
    s_after  = df[df[date_col] >= pivot_date][value_col].astype(float)
    mb = float(s_before.mean()) if len(s_before) else float("nan")
    ma = float(s_after.mean())  if len(s_after) else float("nan")
    delta = (ma - mb) if (not np.isnan(ma) and not np.isnan(mb)) else float("nan")
    return {"mean_before": mb, "mean_after": ma, "delta": delta}