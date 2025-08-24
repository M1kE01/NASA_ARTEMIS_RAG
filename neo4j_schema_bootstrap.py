import os
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv(), override=True)
uri=os.getenv("NEO4J_URI"); user=os.getenv("NEO4J_USERNAME"); pwd=os.getenv("NEO4J_PASSWORD")

cypher = """
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } };

CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text];

CALL db.awaitIndexes();
"""

with GraphDatabase.driver(uri, auth=(user, pwd)).session(database="neo4j") as s:
    for stmt in [x.strip() for x in cypher.strip().split(";") if x.strip()]:
        s.run(stmt)

print("Neo4j schema ready.")
