import os
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv(), override=True)
uri=os.getenv("NEO4J_URI")
user=os.getenv("NEO4J_USERNAME")
pwd=os.getenv("NEO4J_PASSWORD")

with GraphDatabase.driver(uri, auth=(user, pwd)).session(database="neo4j") as s:
    print(s.run("RETURN 1 AS ok").single()["ok"])
