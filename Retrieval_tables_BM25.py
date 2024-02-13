from haystack import Document
import pandas as pd
import json 
from haystack import Document
from tqdm.auto import tqdm  # progress bar
from haystack.utils import launch_es


import os
from haystack.document_stores import ElasticsearchDocumentStore

def treatTable(table):
    header = ', '.join(table.columns)
    values = table.values.tolist()
    text = header + "\n"
    for i in range(len(values)):  
        text = text + ', '.join(values[i]) + "\n"
    return(text)


def read_tables(filename):
    processed_tables = []
    with open(filename) as tables:
        tables = json.load(tables)
        for key, table in tables.items():
            current_columns = table["header"]
            current_rows = table["data"]
            current_df = pd.DataFrame(columns=current_columns, data=current_rows)
            document = Document(content=treatTable(current_df), content_type="table", id=key, meta={"title":table["title"], "type": "table"})
            processed_tables.append(document)

    return processed_tables


tables = read_tables("released_data/all_plain_tables.json")

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(host=host, username="", password="", index="tables")

document_store.write_documents(tables)

