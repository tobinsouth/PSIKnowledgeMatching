# constants
# DATASET = "msmarco"
# DATASET = "scifact"
DATASET = "quora"


# import libraries
import os, json, random
import json
import torch # type: ignore
import numpy as np
from typing import Dict
import logging
import sys

sys.path.append('/u/tsouth/projects/PSIKnowledgeMatching/')
logger = logging.getLogger(__name__)

path_query_embeddings = f'datasets/{DATASET}/query_embeddings.pt'
path_corpus_embeddings = f'datasets/{DATASET}/corpus_embeddings.pt'

# Verify file existence
if not os.path.exists(path_corpus_embeddings):
    raise FileNotFoundError(f"File '{path_corpus_embeddings}' not found.")
if not os.path.exists(path_query_embeddings):
    raise FileNotFoundError(f"File '{path_query_embeddings}' not found.")

# open up  encoded queries and encoded corpus
query_embeddings = torch.load(path_query_embeddings)
corpus_embeddings = torch.load(path_corpus_embeddings)

# do rounding/precompute things if required before passing retrival_search class to be used in search
from tobin.tools import get_edges, fakeround, gauss_noise, uniform_noise
from tqdm import tqdm

n_steps = 6
ROUNDING_TYPES = ['rounded_fakeround', 'gauss_noise', 'uniform_noise']

# for query
query_embedding_range = [query_embeddings.min().item(), query_embeddings.max().item()]
query_diff_ranges = {}


for ROUNDING_TYPE in ROUNDING_TYPES:
    directory = f'datasets/{DATASET}/benchmarks/{ROUNDING_TYPE}'
    os.makedirs(directory, exist_ok=True)

for i in tqdm(range(n_steps)):
    edges, diff = get_edges(i, query_embedding_range)
    # hopefully work with tensor class
    for ROUNDING_TYPE in ROUNDING_TYPES:
        if  ROUNDING_TYPE == 'rounded_fakeround':
            query_embeddings_rounded = fakeround(edges, query_embeddings)
        elif ROUNDING_TYPE == 'gauss_noise':
            query_embeddings_rounded = gauss_noise(query_embeddings, diff)
        elif ROUNDING_TYPE == 'uniform_noise':
            query_embeddings_rounded = uniform_noise(query_embeddings, diff)

        torch.save(query_embeddings_rounded, f'datasets/{DATASET}/benchmarks/{ROUNDING_TYPE}/query_embeddings{i}.pt')

    query_diff_ranges[i] = diff

with open(f'datasets/{DATASET}/benchmarks/query_diff_ranges.json', 'w') as f:
    json.dump(query_diff_ranges, f)



# for corpus_embeddings
corpus_embedding_range = [corpus_embeddings.min().item(), corpus_embeddings.max().item()]
corpus_diff_ranges = {}

for ROUNDING_TYPE in ROUNDING_TYPES:
    directory = f'datasets/{DATASET}/benchmarks/{ROUNDING_TYPE}'
    os.makedirs(directory, exist_ok=True)

for i in tqdm(range(n_steps)):
    edges, diff = get_edges(i, corpus_embedding_range)
    # hopefully work with tensor class
    for ROUNDING_TYPE in ROUNDING_TYPES:
        if  ROUNDING_TYPE == 'rounded_fakeround':
            corpus_embeddings_rounded = fakeround(edges, corpus_embeddings)
        elif ROUNDING_TYPE == 'gauss_noise':
            corpus_embeddings_rounded = gauss_noise(corpus_embeddings, diff)
        elif ROUNDING_TYPE == 'uniform_noise':
            corpus_embeddings_rounded = uniform_noise(corpus_embeddings, diff)

        torch.save(corpus_embeddings_rounded, f'datasets/{DATASET}/benchmarks/{ROUNDING_TYPE}/corpus_embeddings{i}.pt')

    corpus_diff_ranges[i] = diff

with open(f'datasets/{DATASET}/benchmarks/corpus_diff_ranges.json', 'w') as f:
    json.dump(corpus_diff_ranges, f)








