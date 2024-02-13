from haystack import Document
import pandas as pd
import json 
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes.retriever import TableTextRetriever

def read_tables(filename):
    processed_tables = []
    with open(filename) as tables:
        tables = json.load(tables)
        for key, table in tables.items():
            current_columns = table["header"]
            current_rows = table["data"]
            current_df = pd.DataFrame(columns=current_columns, data=current_rows)
            if len(key)>=85:
                key = key[:85]
            
            if len(table['title'])>=85:
                table['title'] = table['title'][:85]
            document = Document(content=current_df, content_type="table", id=key, meta={"title":table["title"], "type": "table"})
            processed_tables.append(document)

    return processed_tables


tables = read_tables("released_data/all_plain_tables.json")
print(len(tables))
 
print(tables[0])

document_store = FAISSDocumentStore(faiss_index_factory_str="HNSW", embedding_dim = 512, return_embedding=True, sql_url="postgresql://postgres:password@localhost:5432/tables")
 
document_store.write_documents(tables)

retriever = TableTextRetriever(
    document_store=document_store,
    query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
    passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
    table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
    embed_meta_fields=["title", "section_title"],
    use_gpu=True 
)
document_store.update_embeddings(retriever=retriever)
document_store.save(index_path="tables.faiss")

