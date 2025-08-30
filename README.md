# NASA Artemis RAG — What it is
 A compact end-to-end Retrieval-Augmented Generation system over recent NASA Artemis
 press releases/blog posts that answers questions with exact chunk-level citations.

# Motivation
 While reading Artemis updates, I couldn’t relocate the exact line that stated
 NASA is targeting April 2026 for Artemis II (and mid-2027 for Artemis III) with
 related heat-shield findings. The web search surfaced many pages but not the precise sentence.
 I wanted a tool that lets me ask a question and jump straight to the exact place
 in the official article I remember. This RAG ingests and chunks Artemis posts,
 so answers include precise [chunk_id:start-end] citations that pinpoint the source text.

# Tech stack
 - Python for orchestration
 - Scraping: httpx, BeautifulSoup, trafilatura
 - Chunking: LangChain RecursiveCharacterTextSplitter (stores start/end char spans)
 - Embeddings/LLM: Google Gemini (text-embedding-004, gemini-2.5-flash) via langchain-google-genai
 - Database: Neo4j 5.x with a vector index (cosine, 768d) + full-text index
 - UI: simple CLI (optional Streamlit app)

# How it works
 1) Ingest (parsing_artemis.py)
    - Crawl NASA Artemis pages, normalize each doc to JSONL with {title, url, published_at, text}.

 2) Prepare (data_preparation.py)
    - Split text into ~1k-char chunks, record start/end offsets for citations.
    - Compute embeddings (text-embedding-004).
    - Upsert nodes/edges in Neo4j:
        (:Document {doc_id,title,url,...})
        (:Chunk {chunk_id,text,start_char,end_char,embedding,...})
        (:Chunk)-[:CHUNK_OF]->(:Document)

 3) Index (neo4j_schema_bootstrap.py)
    - Create uniqueness constraints.
    - Create vector index on :Chunk.embedding and full-text index on :Chunk.text.

 4) Retrieve (quick_query_hybrid.py / cli_rag.py)
    - Hybrid search: vector (semantic) + full-text (exact terms).
    - Min-max normalize, weighted fusion (alpha), dedupe by document.

 5) Generate answer
    - Send top chunks to Gemini with a strict prompt; every claim must cite [chunk_id:start-end].
    - CLI supports :k (top-k), :alpha (fusion weight), and :link "phrase" (best single NASA link).