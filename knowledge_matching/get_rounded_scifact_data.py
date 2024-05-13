# constants
# DATASET = "msmarco"
DATASET = "scifact"
# DATASET = "quora"


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
from tobin.tools import get_edges, fakeround, gauss_noise, gauss_noise, uniform_noise
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
            query_embeddings_rounded = fakeround(edges, query_embedding_range)
        elif ROUNDING_TYPE == 'gauss_noise':
            query_embeddings_rounded = gauss_noise(query_embedding_range, diff)
        elif ROUNDING_TYPE == 'uniform_noise':
            query_embeddings_rounded = uniform_noise(query_embedding_range, diff)

        torch.save(query_embeddings, f'datasets/{DATASET}/benchmarks/{ROUNDING_TYPE}/query_embeddings{i}.pt')

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
            corpus_embeddings_rounded = fakeround(edges, corpus_embedding_range)
        elif ROUNDING_TYPE == 'gauss_noise':
            corpus_embeddings_rounded = gauss_noise(corpus_embedding_range, diff)
        elif ROUNDING_TYPE == 'uniform_noise':
            corpus_embeddings_rounded = uniform_noise(corpus_embedding_range, diff)

        torch.save(corpus_embeddings, f'datasets/{DATASET}/benchmarks/{ROUNDING_TYPE}/corpus_embeddings{i}.pt')

    corpus_diff_ranges[i] = diff

with open(f'datasets/{DATASET}/benchmarks/corpus_diff_ranges.json', 'w') as f:
    json.dump(corpus_diff_ranges, f)

# # Load query_ids and corpus_ids and qrels
# path_query_ids = f'datasets/{DATASET}/query_ids.json'
# path_corpus_ids = f'datasets/{DATASET}/corpus_ids.json'
# path_qrels = f'datasets/{DATASET}/qrels_full.json'

# if not os.path.exists(path_query_ids):
#     raise FileNotFoundError(f"File '{path_query_ids}' not found.")
# with open(path_query_ids, 'r') as f:
#     query_ids = json.load(f)

# if not os.path.exists(path_corpus_ids):
#     raise FileNotFoundError(f"File '{path_corpus_ids}' not found.")
# with open(path_corpus_ids, 'r') as f:
#     corpus_ids = json.load(f)

# if not os.path.exists(path_qrels):
#     raise FileNotFoundError(f"File '{path_qrels}' not found.")
# with open(path_qrels, "r") as f:
#     qrels = json.load(f)


# # create model class
# model = DRESNE(query_ids, corpus_ids, query_embeddings, corpus_embeddings)
# retriever = EvaluateRetrieval(model, score_function="cos_sim")

# # pass to retriever.retrieve corpus and query will be empty values, as they should be useless
# results = retriever.retrieve(corpus = {}, queries = {})
# logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
# ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

# # Data to be written to the CSV file
# headers = ["Metric", "@1", "@3", "@5", "@10", "@100", "@1000"]
# data = [
#         ["NDCG"] + [ndcg[f"NDCG@{k}"] for k in retriever.k_values],
#         ["MAP"] + [_map[f"MAP@{k}"] for k in retriever.k_values],
#         ["Recall"] + [recall[f"Recall@{k}"] for k in retriever.k_values],
#         ["Precision"] + [precision[f"P@{k}"] for k in retriever.k_values]
#     ]

# csv_file_path = f'results/{DATASET}/eval_metrics_DRESNE_{DATASET}.csv'
# # Write data to the CSV file
# with open(csv_file_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(headers)
#     for row in data:
#         writer.writerow(row)

# # Logging info
#     logging.info(f"Metrics are written to {csv_file_path}")
# print("Done saving Metrics")






