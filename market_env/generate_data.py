#!/usr/bin/env python3
import os
import time
import json
import math
import random
import csv
from datetime import datetime, timedelta, timezone

import requests


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRICES_PATH = os.path.join(DATA_DIR, "prices_30d.txt")
AGENTS_PATH = os.path.join(DATA_DIR, "agents_30d.txt")


def fetch_binance_1m(symbol: str, start: datetime, end: datetime):
    """Fetch 1m klines from Binance between start and end UTC datetimes.
    Returns list of (openTimeMs, close) sorted.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": 1000,
    }
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    out = []
    cur = start_ms
    session = requests.Session()
    while cur < end_ms:
        p = params.copy()
        p["startTime"] = cur
        p["endTime"] = end_ms
        resp = session.get(url, params=p, timeout=20)
        if resp.status_code != 200:
            # backoff simple
            time.sleep(1.0)
            continue
        arr = resp.json()
        if not arr:
            break
        for k in arr:
            open_ms = int(k[0])
            close = float(k[4])
            out.append((open_ms, close))
        # advance to after last open
        last_open = int(arr[-1][0])
        cur = last_open + 60_000  # next minute
        # avoid spamming
        time.sleep(0.2)
    out.sort(key=lambda x: x[0])
    return out


def write_prices_from_klines(klines, start):
    os.makedirs(DATA_DIR, exist_ok=True)
    # Build minute index from start
    with open(PRICES_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minute", "price"]) 
        for open_ms, close in klines:
            minute = int((open_ms - int(start.timestamp()*1000)) / 60_000)
            if minute >= 0:
                w.writerow([minute, f"{close:.6f}"])


def generate_agents(total_minutes=30*24*60, num_agents=2000, seed=None):
    if seed is not None:
        random.seed(seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(AGENTS_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_min", "credibility", "scope", "impact", "probability"]) 
        for _ in range(num_agents):
            ts = random.randint(0, total_minutes-1)
            cred = random.uniform(0, 10)
            scope = random.uniform(0, 10)
            impact = random.uniform(-10, 10)
            prob = random.uniform(0, 1)
            w.writerow([ts, f"{cred:.6f}", f"{scope:.6f}", f"{impact:.6f}", f"{prob:.6f}"])


def main():
    # Window: 30 days window starting 6 months ago
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=180)
    end = start + timedelta(days=30)

    print("Fetching BTCUSDT 1m klines from", start, "to", end)
    klines = fetch_binance_1m("BTCUSDT", start, end)
    if len(klines) < 30*24*60*0.8:  # less than 80% minutes
        print("Warning: received only", len(klines), "minutes")
    write_prices_from_klines(klines, start)
    print("Wrote:", PRICES_PATH)

    print("Generating agents...")
    generate_agents(total_minutes=30*24*60, num_agents=2000)
    print("Wrote:", AGENTS_PATH)


if __name__ == "__main__":
    main()


