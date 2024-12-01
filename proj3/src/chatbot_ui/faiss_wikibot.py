import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import re
import os
import anthropic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"
multiprocessing.set_start_method("spawn", force=True)


def get_top_k(query):
    src_directory = "/Users/pratik/Documents/UB Fall 2024/CSE 535/Information_Retrieval_UBCSE535_Fall24/proj3/src/"

    input_file = src_directory + "data/document_final.json"
    input_df = pd.read_json(input_file)

    summaries = input_df['title'].tolist()

    # dim                                 = 384
    torch_device                        = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder                             = SentenceTransformer('paraphrase-MiniLM-L6-v2', device = torch_device)

    # scaler                              = StandardScaler()
    attributes_embeddings               = np.load(src_directory + "data/title_embeddings.npy")
    query_embeddings                    = encoder.encode([query],device = torch_device)

    similarities = cosine_similarity(query_embeddings, attributes_embeddings)
    similarities = similarities.flatten()

    top_indices = np.argsort(similarities)[-4:][::-1]  
    top_similarities = similarities[top_indices]
    print("similarity values:", top_similarities)

    top_k_docs = ""
    for item in top_indices:
        top_k_docs += input_df['summary'][item] + "\n"
        print(input_df['title'][item], " :\n", input_df['summary'][item], end = "\n \n")
    
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
    
    del summaries
    del attributes_embeddings
    del encoder
    del input_df
    del query_embeddings
    del similarities

    return response.content[0].text

    
    
    


    