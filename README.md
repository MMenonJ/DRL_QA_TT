# A Deep Reinforcement Learning Question Answering System for Complex Questions using Texts and Tables
Repository for the BRACIS'24 paper "A Deep Reinforcement Learning Question Answering System for Complex Questions using Texts and Tables".

For more information please check the paper: http://doi.org/10.1007/978-3-031-79032-4_24

Or check the arXiv version: https://arxiv.org/abs/2407.04858 

## Abstract
```
This paper proposes a novel architecture to generate multi-hop answers to open domain questions that require information from texts and tables, using the Open Table-and-Text Question Answering dataset for validation and training. One of the most common ways to generate answers in this setting is to retrieve information sequentially, where a selected piece of data helps searching for the next piece. As different models can have distinct behaviors when called in this sequential information search, a challenge is how to select models at each step.  Our architecture employs reinforcement learning  to choose between different state-of-the-art tools sequentially until, in the end, a  desired answer is generated. This system achieved an F1-score of 19.03, comparable to iterative systems in the literature.
```

## Installation
To install all the necessary files, please refer to the dockerfile. 

```
docker build -t DRL_QA_TT docker
```

## Dataset
This codebase relies on OTT-QA dataset. Please check the download instructions here: https://github.com/wenhuchen/OTT-QA

## Setting up the retrievers
Before running or training the agent, one must set up the desired retriever: BM25 or Tri-Encoder. 
### BM25
The first step to setup BM25 is following the steps on how to use Elasticsearch: https://docs.haystack.deepset.ai/docs/elasticsearch-document-store

After this, run the files Retrieval_tables_BM25.py and Retrieval_texts_BM25.py to push the documents to the database. Remember to change the correct paths.

### Tri-Encoder
The first step to setup Tri-Encoder is following the steps on how to use FAISS: https://docs.haystack.deepset.ai/docs/elasticsearch-document-store

After this, run the files Retrieval_tables.py and Retrieval_texts.py to push the documents to the database. Remember to change the correct paths.

## FiE reader
The folder FiE contains codes that are an adaptation of the work: *Chain-of-Skills: A Configurable Model for Open-Domain
Question Answering*

The original code can be found in the following github: https://github.com/Mayer123/UDT-QA

## Baselines
To replicate the baselines, run rule_based_test.py for the Tri-Encoder retriever and rule_based_test_BM25.py for the BM25 retriever.

## Training
To train the agents, just run the file indicated by RL algorithm and network, like DQN_GRU.py. For BM25, just use the file with BM25 in the end.

## Testing
Please refer to the codes test_models.py and test_models_bm25.py to run the architecture with the trained model for in the validation dataset.

## Acknowledgments
This research has been carried out with  support by *Itaú Unibanco S.A.* through the scholarship program  *Programa de Bolsas Itaú* (PBI); it is also supported in part by the *Coordenação de Aperfeiçoamento de Pessoal de Nível Superior* (CAPES), Finance Code 001, Brazil. The authors would like to thank the Center for Artificial Intelligence (C4AI-USP), with support by the São Paulo Research Foundation (FAPESP) under grant number 2019/07665-4 and by the IBM Corporation. Paulo Pirozelli is supported by the FAPESP grant 2019/26762-0. 

Any opinions, findings, and conclusions expressed in this manuscript are those of the authors and do not necessarily reflect the views, official policy or position of the Itaú-Unibanco, FAPESP, IBM, and CAPES.

## Cite this work
```
@InProceedings{10.1007/978-3-031-79032-4_24,
author="Jos{\'e}, Marcos M.
and Ca{\c{c}}{\~a}o, Fl{\'a}vio N.
and Ribeiro, Maria F.
and Cheang, Rafael M.
and Pirozelli, Paulo
and Cozman, Fabio G.",
editor="Paes, Aline
and Verri, Filipe A. N.",
title="Question Answering with Texts and Tables Through Deep Reinforcement Learning",
booktitle="Intelligent Systems",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="339--353",
abstract="This paper proposes a novel architecture to generate multi-hop answers to open domain questions that require information from texts and tables, using the Open Table-and-Text Question Answering dataset for validation and training. One of the most common ways to generate answers in this setting is to retrieve information sequentially, where a selected piece of data helps searching for the next piece. As different models can have distinct behaviors when called in this sequential information search, a challenge is how to select models at each step. Our architecture employs reinforcement learning to choose between different state-of-the-art tools sequentially until, in the end, a desired answer is generated. This system achieved an F1-score of 19.03, comparable to iterative systems in the literature.",
isbn="978-3-031-79032-4"
}
```
