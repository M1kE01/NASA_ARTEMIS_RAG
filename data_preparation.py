import numpy as np
import pandas as pd
import os
import getpass
import langchain
import openai
from dotenv import load_dotenv, find_dotenv
import langchain_community
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.chat_models import init_chat_model
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from neo4j import GraphDatabase

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(find_dotenv(), override=True)
print("Loaded .env from:", find_dotenv())

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in .env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_API_KEY)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=120
)

with open(r"C:\Users\mihai\Projects\RAG_NASA\data\processed\nasa_artemis.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

docs = []
for d in data:
    text = d["text"]
    pieces = splitter.split_text(text)

    # track character spans so we can cite exact positions
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

texts = [d.page_content for d in docs]
vecs = embeddings.embed_documents(texts)   # 768-d vectors

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

UPSERT = """
MERGE (d:Document {doc_id:$doc_id})
  ON CREATE SET d.title=$title, d.url=$url, d.source='nasa', d.published_at=$published_at
MERGE (c:Chunk {chunk_id:$chunk_id})
  ON CREATE SET
    c.doc_id=$doc_id,
    c.title=$title,          // convenience copy for downstream
    c.text=$text,
    c.start_char=$start_char,
    c.end_char=$end_char,
    c.embedding=$embedding
MERGE (c)-[:CHUNK_OF]->(d)
"""

with driver.session(database="neo4j") as s:
    for ddoc, text, emb in zip(docs, texts, vecs):
        m = ddoc.metadata
        s.run(
            UPSERT,
            doc_id=m["doc_id"],
            title=m["title"],
            url=m["url"],
            published_at=m.get("published_at"),
            chunk_id=m["chunk_id"],
            text=text,
            start_char=m["start_char"],
            end_char=m["end_char"],
            embedding=emb,
        )

driver.close()
print(f"Loaded {len(docs)} chunks into Neo4j.")


