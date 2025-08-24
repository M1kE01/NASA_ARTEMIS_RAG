import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI"); USER=os.getenv("NEO4J_USERNAME"); PWD=os.getenv("NEO4J_PASSWORD")

emb = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
driver = GraphDatabase.driver(NEO4J_URI, auth=(USER, PWD))

def vector_hits(qvec, k=40):
    cy = """
    CALL db.index.vector.queryNodes('chunk_embedding', $k, $qvec)
    YIELD node, score
    RETURN node.chunk_id AS chunk_id, node.doc_id AS doc_id, node.start_char AS s, node.end_char AS e,
           node.text AS text, node.title AS title, coalesce(node.url, '') AS url, score
    LIMIT $k
    """
    with driver.session(database="neo4j") as s:
        return s.run(cy, k=k, qvec=qvec).data()

def fulltext_hits(query, k=40):
    cy = """
    CALL db.index.fulltext.queryNodes('chunk_text', $q)
    YIELD node, score
    RETURN node.chunk_id AS chunk_id, node.doc_id AS doc_id, node.start_char AS s, node.end_char AS e,
           node.text AS text, node.title AS title, coalesce(node.url, '') AS url, score
    ORDER BY score DESC LIMIT $k
    """
    with driver.session(database="neo4j") as s:
        return s.run(cy, q=query, k=k).data()

def hybrid_retrieve(question, k=10, alpha=0.6):
    qvec = emb.embed_query(question)
    v = vector_hits(qvec, k=60)
    ft_query = f'({question}) OR instrument* OR AIRES OR "L-MAPS" OR L\\-MAPS OR "UCIS-Moon" OR LTV'
    f = fulltext_hits(ft_query, k=60)

    scores = {}
    for r in v:
        scores[r["chunk_id"]] = {"row": r, "v": r["score"], "t": 0.0}
    for r in f:
        scores.setdefault(r["chunk_id"], {"row": r, "v": 0.0, "t": 0.0})
        scores[r["chunk_id"]]["t"] = max(scores[r["chunk_id"]]["t"], r["score"])

    def norm(d):
        if not d: return {}
        lo, hi = min(d.values()), max(d.values())
        return {k:(x-lo)/(hi-lo) if hi>lo else 0.0 for k,x in d.items()}

    v_norm = norm({k:v["v"] for k,v in scores.items()})
    t_norm = norm({k:v["t"] for k,v in scores.items()})
    ranked = sorted(scores.items(), key=lambda kv: alpha*v_norm.get(kv[0],0)+(1-alpha)*t_norm.get(kv[0],0), reverse=True)
    return [scores[cid]["row"] for cid,_ in ranked[:k]]

def format_ctx(rows):
    return "\n\n".join(f"[{r['chunk_id']}:{r['s']}-{r['e']}]\n{r['text']}" for r in rows)

if __name__ == "__main__":
    question = "List the instruments selected for the Artemis Lunar Terrain Vehicle and what each measures."
    rows = hybrid_retrieve(question, k=12, alpha=0.6)
    for r in rows[:5]:
        print(f"[{r['chunk_id']}:{r['s']}-{r['e']}]")
        print((r["text"][:260] + "…").replace("\n"," "))
        print()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly from the context. After each claim, append a citation like [chunk_id:start-end]."),
        ("human", "Q: {q}\n\nContext:\n{ctx}")
    ])
    ans = (prompt | llm | StrOutputParser()).invoke({"q": question, "ctx": format_ctx(rows)})
    print(ans)
