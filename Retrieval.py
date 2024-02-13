from haystack import Document
import pandas as pd
import json 
from haystack.document_stores import FAISSDocumentStore
import time


from haystack.nodes.retriever import TableTextRetriever, DensePassageRetriever
import os 

os.chdir('/workspace/mestrado/faiss_texts/')
document_store_all = FAISSDocumentStore( faiss_index_path="texts.faiss", faiss_config_path = "texts.json")


retriever_all = DensePassageRetriever(
    document_store=document_store_all,
    query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
    passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
   # embed_meta_fields=["title", "section_title"],
   # devices=["cuda:0"],
    use_gpu=False,
)
# candidate_docu



# retriever_all = TableTextRetriever(
#     document_store=document_store_all,
#     query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
#     passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
#     table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
#     embed_meta_fields=["title", "section_title"],
#     devices=["cuda:0"],
#     use_gpu=True,
# )
# candidate_documents = retriever_all.retrieve(
#     query="test",
#     top_k=1,
#     )
start_time = time.time()

#for i in range(1):
#    candidate_documents = retriever_all.retrieve(
#        query="test",
#        top_k=1,
#        )
h = []
for i in range(1):
    h.append("test")
candidate_documents = retriever_all.retrieve_batch(
    queries=h,
    top_k=10,
    )
print("--- %s seconds ---" % (time.time() - start_time))
#retriever = TableTextRetriever(
#    document_store=document_store,
#    query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
#    passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
#    table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
#    embed_meta_fields=["title", "section_title"]
#)
#document_store.update_embeddings(retriever=retriever)
#document_store.save(index_path="my_faiss_index.faiss")
os.chdir('/workspace/mestrado/faiss_all/')
document_store_all = FAISSDocumentStore( faiss_index_path="passages_and_tables.faiss", faiss_config_path = "passages_and_tables.json")

retriever_all = TableTextRetriever(
    document_store=document_store_all,
    query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
    passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
    table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
    embed_meta_fields=["title", "section_title"],
    devices=["cuda:0"],
    use_gpu=True,
)

#h = []
#for i in range(1):
#    h.append("test")

#candidate_documents = retriever_all.retrieve_batch(
#    queries=h,
#    top_k=100,
#    batch_size = 200,
#    )
#print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

#for i in range(1):
#    candidate_documents = retriever_all.retrieve(
#        query="test",
#        top_k=1,
#        )
    
h = []
for i in range(1):
    h.append("test")
candidate_documents = retriever_all.retrieve_batch(
    queries=h,
    top_k=10,
    )

print("--- %s seconds ---" % (time.time() - start_time))



      
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

#os.chdir('/workspace/mestrado/')
#print("####################")

#candidate_documents = retriever.retrieve(
#    query="How many professions belong to the cast member of the LGBT-related film in which screenplay is by Levin and Jay Presson Allen",
#    top_k=10,
#    )

#start_time = time.time()
#os.chdir('/workspace/mmenonj/mestrado/faiss/')

#h = []
#for i in range(100):
#    h.append("test")
#candidate_documents = retriever.retrieve_batch(
#    queries=h,
#    top_k=10,
#    )
#print("--- %s seconds ---" % (time.time() - start_time))
#print(candidate_documents)

#start_time = time.time()


#candidate_documents = retriever.retrieve(
#    query="How many professions belong to the cast member of the LGBT-related film in which screenplay is by Levin and Jay Presson Allen ?",
#    top_k=10,
#    )
#print("--- %s seconds ---" % (time.time() - start_time))


#print(candidate_documents)
#import time
#start_time = time.time()
#os.chdir('/workspace/mmenonj/mestrado/faiss/')
#candidate_documents = retriever.retrieve(
#    query="international climate conferences",
#    top_k=1,
#    )
#print("--- %s seconds ---" % (time.time() - start_time))
#print(candidate_documents[0].embedding)

#print(candidate_documents[0].meta["title"])

#start_time = time.time()
#t = retriever.embed_queries(queries=["international climate conferences"])
#print(len(t[0]))
#print(t)

#from sentence_transformers import SentenceTransformer
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Sentences we want to encode. Example:
#sentence = ["deepset/bert-small-mm_retrieval-question_encoder"]
#embedding = model.encode(sentence)
#print(embedding)



#import numpy as np
#print(np.dot(candidate_documents[0].embedding, t[0])/512)

#start_time = time.time()
#candidate_documents = retriever.retrieve(
#    query="international climate conferences",
#    top_k=200,
#)
#print("--- %s seconds ---" % (time.time() - start_time))




#print(candidate_documents)
#from haystack import Pipeline



#table_qa_pipeline = Pipeline()
#table_qa_pipeline.add_node(component=retriever, name="TableTextRetriever", inputs=["Query"])


#prediction = table_qa_pipeline.run("list of world cup finals?", params={"top_k": 5})
#print("--- %s seconds ---" % (time.time() - start_time))

#for doc in prediction['documents']:
#    print(doc.content_type)
#    if doc.content_type != 'text':
#        print(doc)
#print(prediction)