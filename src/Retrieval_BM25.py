from haystack import Document
import pandas as pd
import json 
from haystack.document_stores import FAISSDocumentStore
import time


from haystack.nodes.retriever import TableTextRetriever, DensePassageRetriever
import os 

import os
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import launch_es
from haystack.nodes import BM25Retriever

#launch_es()

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

# document_store = ElasticsearchDocumentStore(host=host, username="", password="", index="tables")


# retriever = BM25Retriever(document_store=document_store)

# candidate_documents = retriever.retrieve(
#     query="this is a test",
#     top_k=10,
#     )
# start_time = time.time()

# for i in range(100):
#     candidate_documents = retriever.retrieve(
#         query="Baldur's Gate 3 is a role-playing video game developed and published by Larian Studios. It is the third main game in the Baldur's Gate series, which is based on the Dungeons & Dragons tabletop role-playing system. A partial version of the game was released in early access format for macOS, Windows, and the Stadia streaming service, on 6 October 2020. The game remained in early access until its full release on Windows on 3 August 2023. macOS and PlayStation 5 versions are scheduled for 6 September 2023, and an Xbox Series X/S port with an unknown release date. The Stadia version was cancelled following Stadia's closure. The game received critical acclaim upon release.",
#         top_k=20,
#         )
# print("--- %s seconds ---" % (time.time() - start_time))



# start_time = time.time()
# h=[]
# for i in range(100):
#     h.append("Baldur's Gate 3 is a role-playing video game developed and published by Larian Studios. It is the third main game in the Baldur's Gate series, which is based on the Dungeons & Dragons tabletop role-playing system. A partial version of the game was released in early access format for macOS, Windows, and the Stadia streaming service, on 6 October 2020. The game remained in early access until its full release on Windows on 3 August 2023. macOS and PlayStation 5 versions are scheduled for 6 September 2023, and an Xbox Series X/S port with an unknown release date. The Stadia version was cancelled following Stadia's closure. The game received critical acclaim upon release. ")

# candidate_documents = retriever.retrieve_batch(
#     queries=h,
#     top_k=20,
#     )

# print(type(h))
# print(len(h))
# print("--- %s seconds ---" % (time.time() - start_time))


document_store_texts = ElasticsearchDocumentStore(host=host, username="", password="", index="document")

retriever_texts = BM25Retriever(document_store=document_store_texts)    


document_store_tables = ElasticsearchDocumentStore(host=host, username="", password="", index="tables")

retriever_tables = BM25Retriever(document_store=document_store_tables)  
candidate_documents = retriever_tables.retrieve_batch(
    queries=["Who created the series in which the character of Robert , played by actor Nonso Anozie , appeared ? List of Batman (TV series) characters Character, Actor, Description, Episode Appearances The Archer, Art Carney, By company records , the Archer was created specifically for the series by writer Stanley Ralph Ross and not related to the previous comic book character of the same name . The character is presented as a skewed version of Robin Hood , with his henchmen reflecting the Robin Hood motif . The Archer , among other characters created for the series , was adapted for a 2009 episode of the animated television series Batman : The Brave and the Bold, 35 , 36 The Black Widow, Tallulah Bankhead, An original character created for the series , Black Widow is a bank robber who uses a spider motif . No actual origin is provided in the series . The Black Widow , among other characters created for the series , was adapted for a 2009 episode of the animated television series Batman : The Brave and the Bold, 89 , 90 The Bookworm, Roddy McDowall, An original character created for the series , Bookworm bases his crimes on books and literary tropes . "
              , 'ãksaldjkasdjaçksladsjhkfweuiruifasdjkhlfsdnkmajnk'],
        top_k=20,
    )
print(candidate_documents)