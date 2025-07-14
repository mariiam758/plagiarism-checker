import requests
import feedparser
import json
import os
import time

def fetch_arxiv_abstracts(category="cs.LG", max_results=500):
    base_url = "http://export.arxiv.org/api/query?"
    all_entries = []
    step = 50
    start = 0
    headers = {'User-Agent': 'MyArxivClient/1.0'}

    while start < max_results:
        url = f"{base_url}search_query=cat:{category}&start={start}&max_results={min(step, max_results - start)}"
        print(f"Fetching results {start} to {start + step}...")
        response = requests.get(url, headers=headers)
        feed = feedparser.parse(response.content)

        if not feed.entries:
            print("No entries found, stopping.")
            break

        all_entries.extend(feed.entries)
        start += step
        time.sleep(1)  # be kind to the API

    abstracts = []
    for entry in all_entries:
        abstract = entry.summary.replace('\n', ' ').strip()
        abstracts.append(abstract)

    return abstracts

if __name__ == "__main__":
    # Get script directory and create data path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Fetch data
    abstracts = fetch_arxiv_abstracts()

    # Save JSON
    output_path = os.path.join(data_dir, "arxiv_abstracts.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"docs": abstracts}, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(abstracts)} abstracts to {output_path}")
