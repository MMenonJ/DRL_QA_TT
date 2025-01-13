from haystack import Document
import pandas as pd
import json 
from haystack import Document
from tqdm.auto import tqdm  # progress bar
from haystack.utils import launch_es

with open('released_data/all_passages.json', 'r') as f:
    all_requests = json.load(f)

passages = []
for key in all_requests.keys():
    passages.append({'text' : all_requests[key], 'title' : key.replace("/wiki/","").replace("(disambiguation)","").replace("_"," ")})
print(len(passages))




docs = []
for d in tqdm(passages, total=len(passages)):
    # create haystack document object with text content and doc metadata
    doc = Document(
        content=d["text"],
        meta={
            "title": d["title"],
            "type": "text"
        }
    )
    docs.append(doc)

import os
from haystack.document_stores import ElasticsearchDocumentStore

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(host=host, username="", password="", index="document")

document_store.write_documents(docs)

