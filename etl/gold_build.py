
import pandas as pd
from collections import Counter
from itertools import islice
from .mongo_utils import col_clean, replace_collection
from .text_utils import tokenize_no_stop

def _read_clean_all(for_airflow: bool = False) -> pd.DataFrame:
    cur = col_clean(for_airflow).find({}, {"_id": 0})
    return pd.DataFrame(list(cur))

def _cooccurrences(tokens_list, top_k=2000):
    bigrams = Counter()
    for toks in tokens_list:
        for i in range(len(toks)-1):
            bigrams[(toks[i], toks[i+1])] += 1
    items = list(islice(bigrams.items(), None))
    df = pd.DataFrame([(a,b,c) for (a,b),c in items], columns=["term","co_term","count"]).sort_values("count", ascending=False)
    if len(df) > top_k:
        df = df.head(top_k)
    return df

def build_gold(for_airflow: bool = False):
    df = _read_clean_all(for_airflow=for_airflow)
    if df.empty:
        replace_collection("cooccurrences_counts", [], for_airflow=for_airflow)
        replace_collection("cooccurrences_percent", [], for_airflow=for_airflow)
        return

    tokens_list = df["cleaned_review"].map(tokenize_no_stop).tolist()
    co_counts = _cooccurrences(tokens_list, top_k=5000)

    total_by_term = co_counts.groupby("term")["count"].transform("sum")
    co_pct = co_counts.copy()
    co_pct["percent"] = (co_pct["count"] / total_by_term).fillna(0.0)

    replace_collection("cooccurrences_counts", co_counts.to_dict("records"), for_airflow=for_airflow)
    replace_collection("cooccurrences_percent", co_pct.to_dict("records"), for_airflow=for_airflow)
