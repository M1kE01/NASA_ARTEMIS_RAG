import os
import sys
from typing import List, Dict
import webbrowser
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Silence LangSmith
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not (GOOGLE_API_KEY and NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
    print("Missing env vars. Ensure GOOGLE_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD are set in .env")
    sys.exit(1)

emb = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def vector_hits(qvec, k=40) -> List[Dict]:
    cy = """
    CALL db.index.vector.queryNodes('chunk_embedding', $k, $qvec)
    YIELD node, score
    MATCH (node)-[:CHUNK_OF]->(d:Document)
    RETURN node.chunk_id AS chunk_id, node.doc_id AS doc_id, node.start_char AS s, node.end_char AS e,
           node.text AS text, coalesce(node.title, d.title) AS title, d.url AS url, score
    LIMIT $k
    """
    with driver.session(database="neo4j") as s:
        return s.run(cy, k=k, qvec=qvec).data()

def fulltext_hits(query, k=40) -> List[Dict]:
    cy = """
    CALL db.index.fulltext.queryNodes('chunk_text', $q)
    YIELD node, score
    MATCH (node)-[:CHUNK_OF]->(d:Document)
    RETURN node.chunk_id AS chunk_id, node.doc_id AS doc_id, node.start_char AS s, node.end_char AS e,
           node.text AS text, coalesce(node.title, d.title) AS title, d.url AS url, score
    ORDER BY score DESC
    LIMIT $k
    """
    with driver.session(database="neo4j") as s:
        return s.run(cy, q=query, k=k).data()

def dedupe_by_doc(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        did = r["doc_id"]
        if did in seen:
            continue
        seen.add(did)
        out.append(r)
    return out

def hybrid_retrieve(question: str, k=8, alpha=0.6) -> List[Dict]:
    """alpha weights vector vs full-text. Returns top-k rows, deduped by document."""
    qvec = emb.embed_query(question)
    v = vector_hits(qvec, k=60)

    # Lucene-friendly query with Artemis priors
    ft_query = f'({question}) OR "April 2026" OR 2026 OR Artemis OR "Artemis II" OR instrument* OR AIRES OR "L-MAPS" OR L\\-MAPS OR "UCIS-Moon"'
    f = fulltext_hits(ft_query, k=60)

    # fuse scores
    scores = {}
    for r in v:
        scores[r["chunk_id"]] = {"row": r, "v": r["score"], "t": 0.0}
    for r in f:
        scores.setdefault(r["chunk_id"], {"row": r, "v": 0.0, "t": 0.0})
        scores[r["chunk_id"]]["t"] = max(scores[r["chunk_id"]]["t"], r["score"])

    def norm(d):
        if not d: return {}
        vals = list(d.values()); lo, hi = min(vals), max(vals)
        return {k: (x - lo) / (hi - lo) if hi > lo else 0.0 for k, x in d.items()}

    v_norm = norm({k:v["v"] for k,v in scores.items()})
    t_norm = norm({k:v["t"] for k,v in scores.items()})

    ranked = sorted(scores.items(),
                    key=lambda kv: alpha*v_norm.get(kv[0],0) + (1-alpha)*t_norm.get(kv[0],0),
                    reverse=True)
    rows = [scores[cid]["row"] for cid, _ in ranked]

    # dedupe by doc, then cap to k
    rows = dedupe_by_doc(rows)[:k]
    return rows

def link_for_phrase(phrase: str) -> str:
    """Return the best document URL whose text contains the given phrase."""
    q = f'"{phrase}" OR ({phrase})'  # exact phrase preferred
    cy = """
    CALL db.index.fulltext.queryNodes('chunk_text', $q)
    YIELD node, score
    MATCH (node)-[:CHUNK_OF]->(d:Document)
    RETURN d.url AS url, coalesce(node.title, d.title) AS title, node.chunk_id AS chunk_id,
           node.start_char AS s, node.end_char AS e, score
    ORDER BY score DESC
    LIMIT 1
    """
    with driver.session(database="neo4j") as s:
        row = s.run(cy, q=q).single()
    if not row:
        return ""
    print("\nTop match:")
    print(f" {row['title']}\n  {row['chunk_id']}:{row['s']}-{row['e']}")
    return row["url"] or ""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer strictly from the context. After each factual claim, append a citation like [chunk_id:start-end]. "
               "If something isn't present, say so explicitly."),
    ("human", "Q: {q}\n\nContext:\n{ctx}")
])
chain = PROMPT | llm | StrOutputParser()

def format_ctx(rows: List[Dict]) -> str:
    return "\n\n".join(f"[{r['chunk_id']}:{r['s']}-{r['e']}]\n{r['text']}" for r in rows)

def print_sources(rows: List[Dict], limit=None):
    print("\nSources:")
    to_show = rows if limit is None else rows[:limit]
    for i, r in enumerate(to_show, 1):
        cite = f"{r['chunk_id']}:{r['s']}-{r['e']}"
        title = r['title'] or r['doc_id']
        url = r['url'] or "(no URL on document)"
        print(f" {i}. {title}\n    {cite}\n    {url}")

def best_link(rows: List[Dict]) -> str:
    return rows[0]["url"] or ""

def chat_once(question: str, k=8, alpha=0.6, link_only=False):
    rows = hybrid_retrieve(question, k=k, alpha=alpha)
    if link_only or "just one link" in question.lower():
        link = best_link(rows)
        if link:
            print("\nLink:", link)
        else:
            print("\nNo URL available on the top result.")
        print_sources(rows, limit=1)
        return rows

    ctx = format_ctx(rows)
    answer = chain.invoke({"q": question, "ctx": ctx})
    print("\nAnswer:\n" + answer)
    print_sources(rows)
    return rows

if __name__ == "__main__":
    print("Artemis RAG (Neo4j + Gemini) — CLI")
    print("Commands: :q quit | :k <int> set top-k | :alpha <0..1> set hybrid weight | :link print one best link")
    last_rows = []
    k = 8
    alpha = 0.6
    while True:
        try:
            q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not q:
            continue
        if q in {":q", ":quit", ":exit"}:
            print("bye.")
            break
        if q.startswith(":k "):
            try:
                k = max(1, int(q.split()[1]))
                print(f"top-k set to {k}")
            except Exception:
                print("usage: :k 5")
            continue
        if q.startswith(":alpha "):
            try:
                alpha = float(q.split()[1]); alpha = max(0.0, min(1.0, alpha))
                print(f"alpha set to {alpha}")
            except Exception:
                print("usage: :alpha 0.5")
            continue
        if q.startswith(":link"):
            parts = q.split(" ", 1)
            if len(parts) == 2 and parts[1].strip():
                phrase = parts[1].strip().strip('"').strip("'")
                url = link_for_phrase(phrase)
                print("\nLink:", url or "(no URL found for phrase)")
            else:
                if last_rows:
                    print("\nLink:", best_link(last_rows) or "(no URL)")
                    print_sources(last_rows, limit=1)
                else:
                    print("No previous query. Try :link \"April 2026\"")
            continue


        last_rows = chat_once(q, k=k, alpha=alpha, link_only=False)
