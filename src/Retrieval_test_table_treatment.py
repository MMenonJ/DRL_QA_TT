from haystack import Document
import pandas as pd
import json 
from haystack.document_stores import FAISSDocumentStore
import time


from haystack.nodes.retriever import TableTextRetriever
import os 


os.chdir('/workspace/mestrado/faiss_tables_flat/')
#start_time = time.time()

document_store2 = FAISSDocumentStore(faiss_index_path="tables.faiss", faiss_config_path = "tables.json")


retriever = TableTextRetriever(
    document_store=document_store2,
    query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
    passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
    table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
    embed_meta_fields=["title", "section_title"],
    devices=["cuda:0"],
    use_gpu=True,
)


candidate_documents = retriever.retrieve(
    query="How many professions belong to the cast member of the LGBT-related film in which screenplay is by Levin and Jay Presson Allen",
    top_k=1,
    )

#print(type(candidate_documents[0].content))
print(candidate_documents[0].meta["title"])
df = candidate_documents[0].content
header = ', '.join(df.columns)
values = df.values.tolist()
text = header + "\n"
for i in range(len(values)):  
    text = text + ', '.join(values[i]) + "\n"
print(text)

import json
predict_file = "/workspace/UDT-QA/downloads/reader_ott/teste.json"
print(json.load(open(predict_file, 'r'))[0]['ctxs'][0]) 
print(json.load(open(predict_file, 'r'))[0]['ctxs'][0]['text']) 