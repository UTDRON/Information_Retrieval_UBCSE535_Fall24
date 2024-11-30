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

summary_embeddings = np.load("/Users/pratik/Documents/UB Fall 2024/CSE 535/Information_Retrieval_UBCSE535_Fall24/proj3/src/data/title_embeddings.npy")
print(summary_embeddings.shape)
summary_embeddings = summary_embeddings[:100]
print(summary_embeddings.shape)

src_directory = "/Users/pratik/Documents/UB Fall 2024/CSE 535/Information_Retrieval_UBCSE535_Fall24/proj3/src/"

input_file = src_directory + "data/document_final.json"
input_df = pd.read_json(input_file)

summaries = input_df['title'].tolist()
summaries = summaries[:100]

dim                                 = 384
k_nearest                           = 3
threshold                           = 0.7
series                              = pd.Series(summaries)
index_values                        = series.index.values
db_vectors                          = summary_embeddings.copy().astype(np.float32)
db_ids                              = np.array(index_values, dtype=np.int64)

faiss.normalize_L2(db_vectors)