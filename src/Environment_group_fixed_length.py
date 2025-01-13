
import gym
import numpy as np
import json
import pandas as pd
import json 
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes.retriever import TableTextRetriever
import os
import random

from FiE.train_qa_fie import train_args, init_distributed_mode, predict_no_sp, load_saved
import torch
from transformers import (AdamW, AutoConfig, AutoTokenizer)
from transformers import ElectraTokenizerFast
from FiE.qa_dataset_fie import QADatasetNoSP, qa_collate_no_sp
from FiE.fie_model import FiEModel
from functools import partial
import json
from torch.utils.data import DataLoader
import time

import warnings
warnings.filterwarnings("ignore")

class OTT_QA_GYM(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, dataset = "train"):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.number_actions = 3
        self.action_space = gym.spaces.Discrete(self.number_actions)
        # Example for using image as input (channel-first; channel-last also works):
        #self.observation_space = spaces.Dict(
        #    {
        #        "agent": spaces.Box(0,  1, shape=(2,), dtype=int),
        #        "target": spaces.Box(0,  1, shape=(2,), dtype=int),
        #    }
        #)
        self.observation_space = gym.spaces.Box(low=-1, high=1,
                                            shape=(11*512,), dtype=np.float32)
        self.dataset = dataset
        if self.dataset == "train":
            f = open('released_data/train.json')
            self.data = json.load(f)
            f.close()
        elif self.dataset == "dev":
            f = open('released_data/dev.json')
            self.data = json.load(f)
            f.close()
            self.f1_scores = []
            self.em_scores = []
#        elif self.dataset == "test":
#            a = 0
        
        self.question = {}
        self.tables = []
        self.texts = []
        self.groups = []
        self.number_passages_first_retrieval = 10
        self.number_passages_group_retrieval = 4
        self.max_steps = 2
        self.count_questions = 0
        self.penalty = -0.02
        self.reward_correct_answer = 2
        self.reward_incorrect_answer = -0.5
        self.end_validation = False

        document_store_texts = FAISSDocumentStore.load(index_path="faiss/texts/texts.faiss", config_path = "faiss/texts/texts.json")
        self.retriever_texts = TableTextRetriever(
            document_store=document_store_texts,
            query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
            passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
            table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
            embed_meta_fields=["title", "section_title"],
            devices=["cuda:0"],
            use_gpu=True,
            max_seq_len_query = 512,
        )

        document_store_tables = FAISSDocumentStore.load(index_path="faiss/tables/tables.faiss", config_path = "faiss/tables/tables.json")
        
        self.retriever_tables = TableTextRetriever(
            document_store=document_store_tables,
            query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
            passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
            table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
            embed_meta_fields=["title", "section_title"],
            devices=["cuda:0"],
            use_gpu=True,
            max_seq_len_query = 512,
        )
        
        self.reader_model_ckpt = "FiE/fie_reader_model_checkpoint/ott_fie_checkpoint_best.pt"
        self.reader_args = train_args()
        init_distributed_mode(self.reader_args)


        if self.reader_args.is_distributed:
            torch.distributed.barrier()
        self.reader_bert_config = AutoConfig.from_pretrained(self.reader_args.model_name)
        self.reader_tokenizer = ElectraTokenizerFast.from_pretrained(self.reader_args.model_name, additional_special_tokens=['[unused1]', '[unused2]'])

        self.reader_model = FiEModel(self.reader_bert_config, self.reader_args)

        self.reader_collate_fc = partial(qa_collate_no_sp, pad_id=self.reader_tokenizer.pad_token_id)
        self.reader_pred_func = predict_no_sp


        self.reader_model = load_saved(self.reader_model, self.reader_model_ckpt)
        self.reader_model.to(self.reader_args.device)

        #self.reset() # Initialize the state

    def treatTable(self, table):
        header = ', '.join(table.columns)
        values = table.values.tolist()
        text = header + "\n"
        for i in range(len(values)):  
            text = text + ', '.join(values[i]) + "\n"
        return(text)


    def retrievalTexts(self):
        if self.count_steps==0:
            candidate_documents = self.retriever_texts.retrieve(
                query=self.question["question"],
                top_k=self.number_passages_first_retrieval,
                )
            for doc in candidate_documents:
                self.texts.append({'id':doc.id, "content":doc.content, "embedding":doc.embedding, "title":doc.meta["title"]})
                self.groups.append([{'id':doc.id, 'type': 'text', "content":doc.content, "embedding":doc.embedding, "title":doc.meta["title"]}])

        else: 
            for i in range(len(self.groups)):
                input_string = self.question["question"]
                text_count = 0
                for j in range(len(self.groups[i])):
                    if self.groups[i][j]["type"] == "text":
                        input_string = input_string + " " + self.groups[i][j]["title"] + ' ' + self.groups[i][j]["content"]
                        text_count += 1
                    else:
                        input_string = input_string + " " + self.groups[i][j]["title"] + ' '  + self.groups[i][j]["content"]

                candidate_documents = self.retriever_texts.retrieve(
                                        query=input_string,
                                        top_k=self.number_passages_group_retrieval+text_count,
                                        )
                for doc in candidate_documents:
                    flag_repeated = False
                    for j in range(len(self.groups[i])):
                        for passage in self.groups[i]:
                            if passage['id'] == doc.id:
                                if passage['content'] == doc.content:
                                    flag_repeated=True
                                break
                    if not flag_repeated:
                        self.groups[i].append({'id':doc.id, 'type': 'text', "content":doc.content, "embedding":doc.embedding, "title":doc.meta["title"]})
                    flag_repeated = False
                    for text in self.texts:
                        if text['id'] == doc.id:
                            if text['content'] == doc.content:
                                flag_repeated=True
                            break
                    if not flag_repeated:
                        self.texts.append({'id':doc.id, "content":doc.content, "embedding":doc.embedding, "title":doc.meta["title"]})

        reward = self.penalty
        return reward

    def retrievalTables(self):
        '''
        This function retrieves tables and add them to the list of tables
        '''
        if self.count_steps==0:
            candidate_documents = self.retriever_tables.retrieve(
                query=self.question["question"],
                top_k=self.number_passages_first_retrieval,
                )
            for doc in candidate_documents:
                self.tables.append({'id':doc.id, "content":self.treatTable(doc.content), "embedding":doc.embedding, "title":doc.meta["title"]})
                self.groups.append([{'id':doc.id, 'type': 'table', "content":self.treatTable(doc.content), "embedding":doc.embedding, "title":doc.meta["title"]}])

        else: 
            for i in range(len(self.groups)):
                input_string = self.question["question"]
                tables_count = 0
                for j in range(len(self.groups[i])):
                    if self.groups[i][j]["type"] == "text":
                        input_string = input_string + " " + self.groups[i][j]["title"] + ' ' + self.groups[i][j]["content"]
                    else:
                        input_string = input_string + " " + self.groups[i][j]["title"] + ' '  + self.groups[i][j]["content"]
                        tables_count += 1

                candidate_documents = self.retriever_tables.retrieve(
                                        query=input_string,
                                        top_k=self.number_passages_group_retrieval+tables_count,
                                        )
                for doc in candidate_documents:
                    flag_repeated = False
                    for j in range(len(self.groups[i])):
                        for passage in self.groups[i]:
                            if passage['id'] == doc.id:
                                if passage['content'] == self.treatTable(doc.content):
                                    flag_repeated=True
                                break
                    if not flag_repeated:
                        self.groups[i].append({'id':doc.id, 'type': 'table', "content":self.treatTable(doc.content), "embedding":doc.embedding, "title":doc.meta["title"]})
                    flag_repeated = False
                    for tables in self.tables:
                        if tables['id'] == doc.id:
                            if tables['content'] == self.treatTable(doc.content):
                                flag_repeated=True
                            break
                    if not flag_repeated:
                        self.tables.append({'id':doc.id, "content":self.treatTable(doc.content), "embedding":doc.embedding, "title":doc.meta["title"]})

        reward = self.penalty
        return reward            
    
    def reader(self):
        ctxs = []
        for i in range(len(self.tables)):
            ctxs.append({'id': self.tables[i]['id'], 'title': self.tables[i]['title'], 'text': self.tables[i]['content'], 'is_gold': False, 'has_answer': False})
        for i in range(len(self.texts)):
            ctxs.append({'id': self.texts[i]['id'], 'title': self.texts[i]['title'], 'text': self.texts[i]['content'], 'is_gold': False, 'has_answer': False})

        input = {'answers': [self.answer], 'question': self.question['question'], 'ctxs': ctxs}
        
        eval_dataset = QADatasetNoSP(self.reader_tokenizer, [input], self.reader_args.max_seq_len, self.reader_args.max_q_len, self.reader_args.max_ans_len, world_size=self.reader_args.world_size, global_rank=self.reader_args.global_rank, neg_num=self.reader_args.neg_num, debug=self.reader_args.debug, num_ctx=self.reader_args.num_ctx)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.reader_args.predict_batch_size, collate_fn=self.reader_collate_fc, pin_memory=True, num_workers=0)
        metrics = self.reader_pred_func(self.reader_args, self.reader_model, eval_dataloader, self.reader_tokenizer, fixed_thresh=None)
        if self.dataset == "dev":
            self.f1_scores.append(metrics['f1'])
            self.em_scores.append(metrics['em'])        
        if metrics['f1'] == 1:
            reward = self.reward_correct_answer
        elif metrics['f1'] == 0:
            reward = self.reward_incorrect_answer
        else:
            reward = metrics['f1']
        print("Golden Answer: '" + str(self.answer)+ "'")	
        print("F1-score: " + str(metrics['f1']))
        return (reward)

    def step(self, action):
        start_time = time.time()
        print("Step number: "+str(self.count_steps))

        if action == 0:  
            # Retrieval_both
            print("Action 0: Retrieval_texts")
            reward = self.retrievalTexts()
            print("Reward: " + str(reward))
            
        elif action == 1:
            # Retrieval_tables
            print("Action 1: Retrieval_tables")
            reward = self.retrievalTables()
            print("Reward: " + str(reward))

        elif self.count_steps == 0:
            print("Step 0 requires a non-sequential retrieval. Doing Retrieval_tables")
            reward = self.retrievalTables()

        if action == 2 or self.count_steps == self.max_steps:
            # Reader
            print("Action 3: Reader")
            reward = self.reader()
            if action != 2:
                reward += self.penalty
            print("Reward: " + str(reward))
            self.done = True

        for i in range(len(self.groups)):
            emb = []
            for j in range(len(self.groups[i])):
                emb.append(self.groups[i][j]["embedding"])
            if i == 0:
                embeddings = np.concatenate((self.question["embedding"], np.mean(emb, axis=0)))
            else:
                embeddings = np.concatenate((embeddings, np.mean(emb, axis=0)))

        print("Number tables: " + str(len(self.tables)))
        print("Number texts: " + str(len(self.texts)))
        self.observation = embeddings
        self.count_steps += 1
        info = {}
        print("--- %s seconds ---" % (time.time() - start_time))
        print("################################################")

        return self.observation, reward, self.done, False, info

    

    def reset(self, seed=None, options=None):
        print("################################################")
        print("################################################")
        if self.dataset == "train":
            self.qa_number = random.randint(0, len(self.data)-1)
            self.answer = self.data[self.qa_number]["answer-text"]
            question_text = self.data[self.qa_number]["question"]
            self.question = {"question": question_text, "embedding": self.retriever_texts.embed_queries(queries=[question_text])[0]}
        else: 
            self.answer = self.data[self.count_questions]["answer-text"]
            question_text = self.data[self.count_questions]["question"]
            self.question = {"question": question_text, "embedding": self.retriever_texts.embed_queries(queries=[question_text])[0]}
            if self.count_questions == len(self.data)-1:
                self.end_validation = True
        self.count_questions +=1
        print("Question " + str(self.count_questions) + "): " + self.question['question'])
        self.tables = []
        self.texts = []
        self.groups = []
        self.done = False
        self.count_steps = 0
        self.observation = np.concatenate((self.question["embedding"], np.zeros(10*512)))
        info = {}
        return self.observation, info  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass
