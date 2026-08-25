from google.cloud import bigquery
import sys
import pandas as pd
import os
import time

def run_query(sql_path, project="stackoverflow-predictor"):
    client = bigquery.Client(project=project)
    with open(sql_path, "r") as f:
        query = f.read()

    print("Submitting query...")
    query_job = client.query(query)
    print(f"Job ID: {query_job.job_id} — waiting for BigQuery to finish executing...")

    query_job.result()
    print("Query finished executing on BigQuery. Downloading results...")

    t0 = time.time()

    # Storage API will be quick.
    df = query_job.to_dataframe(create_bqstorage_client=True, progress_bar_type="tqdm")

    print(f"Downloaded {len(df):,} rows in {time.time() - t0:.1f} seconds.")
    return df

if __name__ == "__main__":
    sql_path = sys.argv[1]
    df = run_query(sql_path)
    print(df.shape)
    print(df.head())

    os.makedirs("data", exist_ok=True)
    out_name = os.path.splitext(os.path.basename(sql_path))[0] + ".parquet"
    out_path = os.path.join("data", out_name)
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")