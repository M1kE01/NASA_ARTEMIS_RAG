import re, json, hashlib, asyncio, datetime as dt
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
import trafilatura
from dateutil import parser as dparse
from tqdm.asyncio import tqdm

NEWS_INDEX = "https://www.nasa.gov/artemis-news-and-articles/"
BLOG_INDEX = "https://www.nasa.gov/blogs/artemis/"
CUTOFF = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)  # taking only recent articles

def slugify(s): return re.sub(r'[^a-z0-9]+','-',s.lower()).strip('-')
def doc_id_from(url, title):
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"nasa_artemis_{slugify(title)[:60]}_{h}"

async def fetch(client, url):
    r = await client.get(url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.text, str(r.url)

def extract_links_from_news_index(html, base):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = urljoin(base, a.get("href",""))
        text = (a.get_text(" ", strip=True) or "").lower()
        # Keep official news releases OR mission articles that are about Artemis
        if ("/news-release/" in href) or ("/missions/" in href) or ("/humans-in-space/" in href):
            if "artemis" in href.lower() or "artemis" in text:
                links.append(href)
    return list(dict.fromkeys(links))

POST_RE = re.compile(
    r"^https?://(www\.)?nasa\.gov/blogs/(artemis|missions)/\d{4}/\d{2}(/\d{2})?/[^/]+/?$"
)

def extract_links_from_blog_index(html, base):
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for a in soup.select("h2 a[href], a.more-link[href], a.read-more[href]"):
        href = urljoin(base, a.get("href",""))
        if POST_RE.match(href):
            urls.append(href)
    return list(dict.fromkeys(urls))

def next_blog_page_url(url, page):
    return f"https://www.nasa.gov/blogs/artemis/page/{page}/"

def clean_trafilatura(html, source_url):
    # trafilatura returns JSON or text; we’ll take text + metadata
    downloaded = trafilatura.extract(html, include_comments=False, include_tables=False, url=source_url,
                                     output_format="json", with_metadata=True, favor_recall=True)
    if not downloaded:
        txt = trafilatura.extract(html, url=source_url) or ""
        return {"text": txt.strip()}
    return json.loads(downloaded)

def parse_date(meta_json, fallback_html):
    # Try multiple places for a robust published date
    for k in ("date","published","date_publish","date_modified","date_download"):
        if meta_json.get(k):
            try: return dparse.parse(meta_json[k])
            except: pass
    # OG/time tags fallback
    soup = BeautifulSoup(fallback_html, "html.parser")
    for sel,attr in [("meta[property='article:published_time']","content"),
                     ("time[datetime]","datetime")]:
        el = soup.select_one(sel)
        if el and el.get(attr): 
            try: return dparse.parse(el.get(attr))
            except: pass
    return None

async def harvest_year_news(client, years=(2025, 2024), per_year_limit=200):
    collected = []
    for yr in years:
        idx = f"https://www.nasa.gov/{yr}-news-releases/"
        try:
            html, url = await fetch(client, idx)
        except Exception:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a[href]"):
            href = urljoin(url, a.get("href",""))
            if "/news-release/" in href:
                collected.append(href)
    # dedupe & cap
    return list(dict.fromkeys(collected))[: sum([200 for _ in years])]

async def harvest_news(client, limit=120):
    html, url = await fetch(client, NEWS_INDEX)
    links = extract_links_from_news_index(html, url)
    # (Optional) follow more pages by discovering "Next" link:
    soup = BeautifulSoup(html, "html.parser")
    # If pagination exists, try page/2/.. up to 5
    for p in range(2,6):
        try:
            h2, _ = await fetch(client, f"https://www.nasa.gov/artemis-news-and-articles/page/{p}/")
            links += extract_links_from_news_index(h2, NEWS_INDEX)
        except Exception:
            break
    links = list(dict.fromkeys(links))[:limit]
    return links

async def harvest_blog(client, limit_per_pages=6, max_pages=8):
    links = []
    first_html, _ = await fetch(client, BLOG_INDEX)
    links += extract_links_from_blog_index(first_html, BLOG_INDEX)
    for p in range(2, max_pages+1):
        try:
            html, _ = await fetch(client, next_blog_page_url(BLOG_INDEX, p))
            links += extract_links_from_blog_index(html, BLOG_INDEX)
        except Exception:
            break
    return list(dict.fromkeys(links))[:limit_per_pages*max_pages]

async def pull_and_normalize(client, url):
    html, final_url = await fetch(client, url)
    meta = clean_trafilatura(html, final_url)
    title = meta.get("title") or BeautifulSoup(html,"html.parser").find("title").get_text(strip=True)
    text  = meta.get("text","").strip()
    published_at = parse_date(meta, html)
    # Heuristic filter for our Artemis scope:
    if published_at and published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=dt.timezone.utc)
    if published_at and published_at < CUTOFF:
        return None
    if len(text) < 400:  # too short to be useful for RAG
        return None
    doc = {
        "doc_id": doc_id_from(final_url, title),
        "title": title,
        "url": final_url,
        "source": "nasa",
        "collection": "artemis_news" if "/news-release/" in final_url else "artemis_blog",
        "published_at": published_at.isoformat() if published_at else None,
        "license_note": "NASA text generally public domain; logos/insignia restricted.",
        "text": text
    }
    return doc

async def main():
    out = []
    async with httpx.AsyncClient(headers={"User-Agent":"artemis-rag/0.1"}) as client:
        news_links  = await harvest_news(client)
        blog_links  = await harvest_blog(client)
        year_links = await harvest_year_news(client, years=(2025, 2024))
        all_links = list(dict.fromkeys(year_links + news_links + blog_links)) 
        for url in tqdm(all_links):
            try:
                doc = await pull_and_normalize(client, url)
                if doc: out.append(doc)
            except Exception as e:
                print("skip", url, e)
    out.sort(key=lambda d: d.get("published_at") or "", reverse=True)
    with open("data/processed/nasa_artemis.jsonl","w",encoding="utf-8") as f:
        for d in out: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved {len(out)} docs to data/processed/nasa_artemis.jsonl")

if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)
    asyncio.run(main())
