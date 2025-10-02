import json
import os
import subprocess
import time
from pathlib import Path

import requests
import yaml
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
STORAGE_URL = os.getenv("STORAGE_URL")
BLOB_READ_WRITE_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")


def download(src: str, dst: str) -> None:
    """Download a file from Vercel Blob Storage"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}"}
    response = requests.get(src, headers=headers)
    response.raise_for_status()
    dst.write_bytes(response.content)


def upload(src: str, dst: str, content_type: str) -> str:
    """Upload a file to Vercel Blob Storage"""
    url = f"https://blob.vercel-storage.com/{dst}"
    headers = {
        "Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}",
        "x-add-random-suffix": "0",
        "Content-Type": content_type,
    }
    with open(src, "rb") as f:
        response = requests.put(url, headers=headers, data=f)
    response.raise_for_status()
    return response.json()["url"]


def update(task_id: str, status: str) -> None:
    """Update task status in the database"""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute('UPDATE "tasks" SET status=%s WHERE id=%s;', (status, task_id))
        conn.commit()


# Polling for tasks
while True:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            # Get a pending task
            cur.execute(
                """
                SELECT id, project_id
                  FROM "tasks"
                 WHERE status = %s
                 ORDER BY created_at ASC
                 LIMIT 1
                 FOR UPDATE SKIP LOCKED;
                """,
                ("pending",),
            )
            row = cur.fetchone()

            if not row:
                conn.rollback()
                time.sleep(5)
                continue

            task_id = row["id"]
            project_id = row["project_id"]

            cur.execute(
                'UPDATE "tasks" SET status=%s WHERE id=%s;', ("running", task_id)
            )
            conn.commit()
            break

try:
    # Download input.csv
    src = f"{STORAGE_URL}/{project_id}/input.csv"
    dst = "data/input.csv"
    download(src, dst)

    # Download config.json
    src = f"{STORAGE_URL}/{project_id}/config.json"
    dst = "config.json"
    download(src, dst)

    # Convert JSON to YAML
    with open("config.json", "r") as f:
        config = json.load(f)
    with open("config.yaml", "w") as g:
        yaml.safe_dump(config, g, sort_keys=False)

    # Run main.py
    subprocess.run(["uv", "run", "python", "main.py"], check=True)

    # Upload result.csv
    src = f"runs/{project_id}/runner/output/result.csv"
    dst = f"{project_id}/result.csv"
    upload(src, dst, "text/csv")

    update(task_id, "succeeded")  # Update task status

except KeyboardInterrupt:
    update(task_id, "canceled")  # Update task status
    raise

except Exception:
    update(task_id, "failed")  # Update task status
    raise
