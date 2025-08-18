import pandas as pd

def apply_filters(df, languages=None, date_range=None, sentiment_range=None, search_terms=None):
    out = df.copy()
    # normalize columns
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    if languages:
        out = out[out["language"].isin(languages)]
    if date_range:
        start, end = date_range
        out = out[(out["timestamp"] >= pd.to_datetime(start)) & (out["timestamp"] <= pd.to_datetime(end))]
    if sentiment_range and "sentiment" in out.columns:
        lo, hi = sentiment_range
        out = out[(out["sentiment"] >= lo) & (out["sentiment"] <= hi)]
    if search_terms:
        # Simple contains any term
        pattern = "|".join(map(lambda s: pd.util.hashing.hash_object(s) and s, search_terms))  # noop but keeps simple
        # safer: iterate
        mask = pd.Series(False, index=out.index)
        for term in search_terms:
            if not isinstance(term, str) or not term:
                continue
            mask = mask | out["review"].fillna("").str.contains(term, case=False, na=False)
        out = out[mask]
    return out