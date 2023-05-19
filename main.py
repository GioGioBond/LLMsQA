import os
import logging

import re
import json
import torch
import faiss
import numpy as np

from transformers import AutoModel,AutoTokenizer
from langchain.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import *
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
#from
from faiss import normalize_L2
logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',level=logging.DEBUG)

    # res = []
    # chunk_overlap = int(chunk_overlap * chunk_size)
    # for seq in re.split(r'[\.。,，？\?!！；;\n]', texts):
    #     if len(seq) < chunk_size:
    #         res.append(seq)
    #     else:
    #         res += [seq[i * chunk_size:(i+1) * chunk_size] for i in range(len(seq)//chunk_size)]
    #         start_idx = 0
    #         cur_idx = min(start_idx + chunk_size, len(seq))
    #         chunk_seq = seq[start_idx:cur_idx]
    #         while start_idx < len(seq):
    #             res.append(chunk_seq)
    #             start_idx += chunk_size - chunk_overlap
    #             cur_idx = min(start_idx + chunk_size, len(seq))
    #             chunk_seq = seq[start_idx:cur_idx]
    #
    # return res




class QA_model(object):
    def __init__(self, embedding_path, llm_path):
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_path)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.llm = AutoModel.from_pretrained(llm_path, trust_remote_code=True).half().cuda()

        with open(os.path.join(embedding_path + '/config.json'),'r', encoding='utf-8') as emb_f:
            self.emb_config = json.load(emb_f)

    def load_doc(self, file_path):
        return

    def embedding_transfer(self, text):
        if isinstance(text, list):
            return self.embedding.embed_documents(text)
        elif isinstance(text, str):
            return self.embedding.embed_query(text)
        else:
            raise ValueError(f"出现了未知数据类型: {type(text)}")

    def build_index(
        self, texts, emb_index=[], index_path='', ids_texts_map_path='', CREAT=True
    ):
        emb = self.embedding_transfer(texts)
        emb = np.array(emb, dtype='float32')
        emb_index = np.array(emb_index, dtype='int')

        if emb.shape[0] != emb_index.shape[0]:
            raise ValueError(f'embedding的维度{emb.shape}与索引维度{emb_index.shape}不同')
        if CREAT:
            index = faiss.IndexFlatL2(self.emb_config['hidden_size'],)
            #index = faiss.IndexFlatIP(self.emb_config['hidden_size'])
            #index = faiss.IndexIDMap(index)
            #faiss.write_index(index, index_path)
        else:
            index = faiss.read_index(index_path)

        if emb.shape[0] > 0:
            #self.add_index(index, emb, emb_index)
            #index.add_with_ids(emb, emb_index)

            index.add(emb)
            faiss.write_index(index, index_path)
            with open(ids_texts_map_path,'w',encoding='utf-8') as f:
                ids_texts_map = {}
                for i,j in zip(emb_index, texts):
                    ids_texts_map[int(i)] = j
                f.write(json.dumps(ids_texts_map, ensure_ascii=False))
        print('index建立完成')
        return index

    def edit(
        self, index, del_ids=[], add_ids=[], add_emb=[], add_texts=[], partten='', ids_text_map_path=''
    ):
        if partten == 'delete':
            index.remove_ids(np.array(del_ids, dtype='int'))
        elif partten == 'edit':
            index.remove_ids(np.array(del_ids, dtype='int'))
            index.add_with_ids(np.array(add_emb, dtype='float32'), np.array(add_ids, dtype=int32))
            with open(ids_text_map_path, 'r', encoding='utf-8') as f:
                ids_text_map = json.load(f)
                #ids_text_map = {}
                for i in del_ids:
                    del ids_text_map[i]
                for i,j in zip(add_ids, add_texts):
                    ids_text_map[i] = add_texts[j]
                f.write(json.dumps(ids_text_map, ensure_ascii=False))

    def search_text(self, index, query, ids_texts_map_path, top_k=1):
        ids_texts_map = json.load(open(ids_texts_map_path,'r',encoding='utf-8'))
        print(ids_texts_map)
        query_emb = self.embedding_transfer(query)
        query_emb = np.array(query_emb, dtype='float32').reshape(-1, self.emb_config['hidden_size'])

        D, I = index.search(query_emb, top_k)

        print(I, D)
        #D有时候是倒序的，还不清楚原因

        return [ids_texts_map[str(I[i][j])] for i in range(I.shape[0]) for j in range(I.shape[-1])]

    def chat(self, prompt, max_new_tokens=2024, do_sample=False):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        output = self.llm.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample)[0][input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
        torch.cuda.empty_cache()
        return generated_text

def print_log(args):
    for i in locals().keys():
        if type(args) in [list,str]:
            logging.info(f'---{i}--length:{len(args)}---->{args}')
        else:
            logging.info(f'---{i}--------------------->{args}')
def generate_prompt(texts, query, tokens_limit=2048):

    template_size = 120
    context = ''
    for i in texts:
        context += f'{i}\n'
        if len(context) > tokens_limit - template_size - len(query):
            break

    #print_log(context)
    #print_log(query)
    prompt = f'''
要求： 基于已知内容，请用中文以要求的格式简短直接地回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题"。不允许在答案中添加额外成分。

已知内容：

{context[:tokens_limit]}
问题： {query}

格式： 问题:答案

请根据提供的信息回答问题。
'''
    return prompt

def split_text(texts):
    texts = re.sub('\n{2,}','\n',texts)
    splited_list = re.split(r'[\.。？\?!！；;\n]', texts)
    if len(splited_list[-1]) != 0:
        return splited_list
    else:
        return splited_list[:-1]

def read_texts(file_path):
    res = ''
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            res += line
    return res
if __name__ == '__main__':
    #pass
    #emb_path = '/opt/wmh_FileFolder/chatglm/chatglm_model/sentence_transformers'
    emb_path = '/opt/wmh_FileFolder/chatglm/chatglm_model/Erlangshen-TCBert'
    llm_path = '/opt/wmh_FileFolder/chatglm/chatglm_model'
    doc_path = '/opt/wmh_FileFolder/chatglm/chatglm_model/kb/knowledge.txt'
    index_path = '/opt/wmh_FileFolder/chatglm/chatglm_model/kb/myindex'
    ids_texts_map_path = '/opt/wmh_FileFolder/chatglm/chatglm_model/kb/map.txt'
    max_token_size = 2048

    query = '徐伦工作几年了'
    #query = '徐伦工作是'
    #print(read_texts(doc_path))
    texts = split_text(read_texts(doc_path))

    model = QA_model(emb_path, llm_path)

    index = model.build_index(texts, range(len(texts)), index_path, ids_texts_map_path)
    context = model.search_text(index, query, ids_texts_map_path,2)
    prompt = generate_prompt(context, query)
    print('-----------------------------------------------------------------')
    print(prompt)
    print('-----------------------------------------------------------------')
    answer = model.chat(prompt)
    print(answer)

