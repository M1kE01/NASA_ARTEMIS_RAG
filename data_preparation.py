import os
import json
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
import getpass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(find_dotenv(), override=True)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = getpass.getpass("Enter your GOOGLE_API_KEY (input hidden): ").strip()
    if not GOOGLE_API_KEY:
        print("A Google API key is required to run cli_rag.")
        sys.exit(1)

if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
    raise RuntimeError("Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env")

DATA_JSONL = os.getenv(
    "DATA_JSONL",
    r"C:\Users\mihai\Projects\RAG_NASA\data\processed\nasa_artemis.jsonl",
)

p = Path(DATA_JSONL)
if not p.exists():
    raise FileNotFoundError(f"Cannot find {p}")
with p.open("r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(data)} source documents from {p}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)

docs = []
for d in data:
    text = d["text"]
    pieces = splitter.split_text(text)

    # track deterministic spans for each chunk (start_char, end_char)
    spans, pos = [], 0
    for ch in pieces:
        start = text.find(ch, pos)
        if start < 0:
            start = pos
        end = start + len(ch)
        spans.append((start, end))
        pos = end

    for i, ch in enumerate(pieces):
        chunk_id = f"{d['doc_id']}#c{i:04d}"
        docs.append(
            Document(
                page_content=ch,
                metadata={
                    "chunk_id": chunk_id,
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "url": d["url"],
                    "published_at": d.get("published_at"),
                    "start_char": spans[i][0],
                    "end_char": spans[i][1],
                    "source": "nasa",
                },
            )
        )

print(f"Created {len(docs)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004", google_api_key=GOOGLE_API_KEY
)
texts = [d.page_content for d in docs]
vecs = embeddings.embed_documents(texts)  # 768-d vectors

rows = []
for ddoc, text, emb in zip(docs, texts, vecs):
    m = ddoc.metadata
    rows.append(
        {
            "doc_id": m["doc_id"],
            "title": m["title"],
            "url": m["url"],
            "published_at": m.get("published_at"),
            "chunk_id": m["chunk_id"],
            "text": text,
            "start_char": m["start_char"],
            "end_char": m["end_char"],
            "embedding": emb,
        }
    )

UPSERT_BATCH = """
UNWIND $rows AS row
MERGE (d:Document {doc_id: row.doc_id})
  ON CREATE SET
    d.title = row.title,
    d.url = row.url,
    d.source = 'nasa',
    d.published_at = row.published_at
MERGE (c:Chunk {chunk_id: row.chunk_id})
  ON CREATE SET
    c.doc_id = row.doc_id,
    c.title = row.title,
    c.url = row.url,
    c.text = row.text,
    c.start_char = row.start_char,
    c.end_char = row.end_char,
    c.embedding = row.embedding
  ON MATCH SET
    c.title = row.title,
    c.url = row.url,
    c.text = row.text,
    c.start_char = row.start_char,
    c.end_char = row.end_char,
    c.embedding = row.embedding
MERGE (c)-[:CHUNK_OF]->(d)
"""

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
with driver.session(database="neo4j") as s:
    s.run(UPSERT_BATCH, rows=rows)
driver.close()

print(f"Upserted {len(rows)} chunks to Neo4j.")
