
import argparse
from datetime import datetime
from etl.config import Config
from etl.bronze_extract import extract_app
from etl.silver_clean import to_silver
from etl.gold_build import build_gold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full","incr"], default="incr")
    args = parser.parse_args()

    if not Config.app_ids:
        raise SystemExit("Configure APP_IDS in .env")

    dts = set()
    for app_id in Config.app_ids:
        dt = extract_app(app_id, mode=args.mode, for_airflow=False)
        dts.add((app_id, dt))

    for app_id, dt in dts:
        to_silver(app_id, dt=dt, for_airflow=False)

    build_gold(for_airflow=False)
    print("OK: Mongo collections ready: reviews_{app_id}, reviews_clean, cooccurrences_counts, cooccurrences_percent")

if __name__ == "__main__":
    main()
