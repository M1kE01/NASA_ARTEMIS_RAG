import numpy as np
import pandas as pd
import os
import langchain
import openai
from dotenv import load_dotenv
import langchain_community
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.chat_models import init_chat_model
import json
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

#llm_model = init_chat_model('gemini-2.5-flash', model_provider='google_genai')

with open(r"C:\Users\mihai\Projects\RAG_WS\data\processed\nasa_artemis.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(data[0]['doc_id'])