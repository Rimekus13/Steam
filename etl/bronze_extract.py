# etl/bronze_extract.py
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import gzip, json

from .http import fetch_reviews_page
from .state import load_state, save_state
# plus besoin de bulk_upsert_raw / ensure_indexes pour le RAW → on les retire

def _raw_out_dir(app_id: str, dt: str) -> Path:
    """Dossier de sortie pour stocker le RAW (datalake)."""
    d = Path(f"data/raw/{app_id}/{dt}")
    d.mkdir(parents=True, exist_ok=True)
    return d

def extract_app(app_id: str, mode: str = "incr", for_airflow: bool = False, max_pages: int = 200) -> str:
    """API Steam → RAW datalake (JSON.gz) avec logs de debug."""
    state = load_state(app_id)
    max_seen = state.get("max_timestamp_updated", 0)
    cursor = "*" if mode == "full" else state.get("last_cursor", "*")

    dt = datetime.utcnow().strftime("%Y-%m-%d")
    pages = 0
    total = 0
    pbar = tqdm(desc=f"Extract {app_id}")
    while True:
        pages += 1
        if pages > max_pages:
            print(f"[INFO] Stop after max_pages={max_pages}")
            break

        data = fetch_reviews_page(app_id, cursor)
        reviews = data.get("reviews") or []
        if pages == 1:
            print(f"[DEBUG] First page: {len(reviews)} reviews, cursor={data.get('cursor')!r}")

        if not reviews:
            print("[INFO] No reviews on this page → stop.")
            break

        # --- Sauvegarde RAW dans le datalake ---
        out_dir = _raw_out_dir(app_id, dt)
        out_path = out_dir / f"reviews_{pages:04d}.json.gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as gf:
            json.dump(reviews, gf, ensure_ascii=False)
        total += len(reviews)
        pbar.update(len(reviews))

        # watermark
        max_page_updated = 0
        for r in reviews:
            tsu = r.get("timestamp_updated") or 0
            if tsu > max_seen:
                max_seen = tsu
            if tsu > max_page_updated:
                max_page_updated = tsu

        cursor = (data.get("cursor") or "")
        if mode == "incr" and state.get("max_timestamp_updated") and max_page_updated <= state["max_timestamp_updated"]:
            print("[INFO] Incremental stop: reached previous watermark.")
            break
        if not cursor:
            print("[INFO] No cursor → stop.")
            break

    pbar.close()
    print(f"[INFO] Pages fetched={pages-1}, total reviews saved≈{total}")
    save_state(app_id, {"max_timestamp_updated": max_seen, "last_cursor": cursor or "*"})
    return dt
