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

src_directory = "/Users/pratik/Documents/UB Fall 2024/CSE 535/Information_Retrieval_UBCSE535_Fall24/proj3/src/"

input_file = src_directory + "data/document_final.json"
input_df = pd.read_json(input_file)

summaries = input_df['title'].tolist()

dim                                 = 384
torch_device                        = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder                             = SentenceTransformer('paraphrase-MiniLM-L6-v2', device = torch_device)

attributes_embeddings               = encoder.encode(summaries,device = torch_device)

scaler                              = StandardScaler()
attributes_embeddings               = scaler.fit_transform(attributes_embeddings)

np.save(src_directory + "data/title_embeddings.npy", attributes_embeddings)