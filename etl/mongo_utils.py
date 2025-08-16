
from typing import Iterable, List, Dict
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.errors import BulkWriteError
from .config import Config

def get_client(for_airflow: bool = False) -> MongoClient:
    uri = Config.mongo_uri_docker if for_airflow else Config.mongo_uri
    return MongoClient(uri)

def get_db(for_airflow: bool = False):
    return get_client(for_airflow)[Config.mongo_db]

def col_raw(app_id: str, for_airflow: bool = False):
    return get_db(for_airflow)[f"reviews_{app_id}"]

def col_clean(for_airflow: bool = False):
    return get_db(for_airflow)["reviews_clean"]

def col_co_counts(for_airflow: bool = False):
    return get_db(for_airflow)["cooccurrences_counts"]

def col_co_percent(for_airflow: bool = False):
    return get_db(for_airflow)["cooccurrences_percent"]

def ensure_indexes(app_id: str, for_airflow: bool = False):
    c = col_raw(app_id, for_airflow)
    c.create_index([("recommendationid", ASCENDING)], name="uniq_reco", unique=True)
    c.create_index([("timestamp_updated", ASCENDING)])
    c.create_index([("author.steamid", ASCENDING)])

    cl = col_clean(for_airflow)
    cl.create_index([("app_id", ASCENDING), ("review_id", ASCENDING)], name="uniq_clean", unique=True)
    cl.create_index([("review_date", ASCENDING)])
    cl.create_index([("language", ASCENDING)])
    cl.create_index([("sentiment", ASCENDING)])

    col_co_counts(for_airflow).create_index([("term", ASCENDING), ("co_term", ASCENDING)], name="pairs")
    col_co_percent(for_airflow).create_index([("term", ASCENDING), ("co_term", ASCENDING)], name="pairs")

def bulk_upsert_raw(app_id: str, reviews: List[Dict], for_airflow: bool = False):
    if not reviews:
        return
    ops = []
    for r in reviews:
        key = {"recommendationid": r.get("recommendationid")}
        ops.append(UpdateOne(key, {"$set": r, "$setOnInsert": {"app_id": app_id}}, upsert=True))
    try:
        col_raw(app_id, for_airflow).bulk_write(ops, ordered=False)
    except BulkWriteError:
        pass

def bulk_upsert_clean(rows: List[Dict], for_airflow: bool = False):
    if not rows:
        return
    ops = []
    for r in rows:
        key = {"app_id": r["app_id"], "review_id": r["review_id"]}
        ops.append(UpdateOne(key, {"$set": r}, upsert=True))
    col_clean(for_airflow).bulk_write(ops, ordered=False)

def replace_collection(name: str, docs: Iterable[Dict], for_airflow: bool = False):
    c = get_db(for_airflow)[name]
    c.drop()
    docs = list(docs)
    if docs:
        c.insert_many(docs)
