from haystack import Document
import pandas as pd
import json 
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes.retriever import TableTextRetriever

with open('released_data/all_passages.json', 'r') as f:
    all_requests = json.load(f)

passages = []
for key in all_requests.keys():
    passages.append({'text' : all_requests[key], 'title' : key.replace("/wiki/","").replace("(disambiguation)","").replace("_"," ")})
print(len(passages))


from haystack import Document
from tqdm.auto import tqdm  # progress bar


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

len(docs)

document_store = FAISSDocumentStore(faiss_index_factory_str="HNSW", n_links=64, embedding_dim = 512, return_embedding=True, sql_url="postgresql://postgres:password@localhost:5432/texts")
 
document_store.write_documents(docs)

retriever = TableTextRetriever(
    document_store=document_store,
    query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
    passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
    table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
    embed_meta_fields=["title", "section_title"]
)
document_store.update_embeddings(retriever=retriever)
document_store.save(index_path="faiss/texts/texts.faiss")
