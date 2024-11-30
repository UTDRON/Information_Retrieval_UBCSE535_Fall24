import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import re
import gensim
import os
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
import sys
import anthropic


src_directory = "/Users/pratik/Documents/UB Fall 2024/CSE 535/Information_Retrieval_UBCSE535_Fall24/proj3/src/"

input_file = src_directory + "data/document_final.json"
input_df = pd.read_json(input_file)

summaries = input_df['title'].tolist()
# summaries = summaries[:150]

dim                                 = 384
torch_device                        = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder                             = SentenceTransformer('paraphrase-MiniLM-L6-v2', device = torch_device)

# attributes_embeddings               = encoder.encode(summaries,device = torch_device)

# scaler                              = StandardScaler()
# attributes_embeddings               = scaler.fit_transform(attributes_embeddings)

# print(attributes_embeddings)

# np.save(src_directory + "data/title_embeddings.npy", attributes_embeddings)

attributes_embeddings               = np.load(src_directory + "data/title_embeddings.npy")
print(attributes_embeddings)

def get_top_k(query,attributes_embeddings):
    k_nearest                           = 4
    threshold                           = 0.7
    series                              = pd.Series(summaries)
    index_values                        = series.index.values
    db_vectors                          = attributes_embeddings.copy().astype(np.float32)
    db_ids                              = np.array(index_values, dtype=np.int64)

    faiss.normalize_L2(db_vectors)
    index                               = faiss.IndexFlatIP(dim)
    index                               = faiss.IndexIDMap(index)
    index.add_with_ids(db_vectors,db_ids)

    
    query_embeddings                    = encoder.encode(query,device = torch_device)


    search_query                        = query_embeddings.copy().astype(np.float32)
    faiss.normalize_L2(search_query)
    similarities, similarities_ids      = index.search(search_query, k=k_nearest)
    similarities                        = np.around(np.clip(similarities,0,1),decimals = 4)
    print("similarity values:", similarities)
    # print("ids:", similarities_ids)
    # for item in similarities_ids:
    #     for id in item:
    #         print(summaries[id])
    
    del index
    # del attributes_embeddings
    return similarities_ids[0]

query = 'flight tickets prices for chicago'
top_k_ids = get_top_k([query],attributes_embeddings)
# get_top_k(['what is your name'],attributes_embeddings)
# get_top_k(['how can i travel to canada'],attributes_embeddings)
# sys.exit()  

top_k_docs = ""
for item in top_k_ids:
    # top_k_docs += input_df['title'][item] + " : "
    top_k_docs += input_df['summary'][item] + "\n"
    print(input_df['title'][item], " :\n", input_df['summary'][item], end = "\n \n")
# print(top_k_docs)

client = anthropic.Anthropic(api_key='anthropic key')

description = (
    f"Generate a summary of the given text. Don't give any context just write summary. Don't start with Summary: "
    "Don't write sentences like: The text mentions, the text discusses. Don't use 3rd person. Write as a narrative"
    f"The given text:\n{top_k_docs}\n"
)
            
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=200,
    messages=[
        {"role": "user", "content": description}
    ]
)
print(response.content[0].text)
    