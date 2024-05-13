# constants
DATASET = "msmarco"
# DATASET = "scifact"
# DATASET = "quora"



# import libraries
import os, json, csv
import torch # type: ignore
import numpy as np
from typing import Dict
import logging
from tqdm import tqdm
import sys

from beir import util
from beir_reengineered import NewSentenceBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from experiement_retrieval_exact_search  import DenseRetrievalExactSearchNoEncoding as DRESNE # type: ignore


sys.path.append('/u/tsouth/projects/PSIKnowledgeMatching/')
logger = logging.getLogger(__name__)


# Load query_ids and corpus_ids and qrels
path_query_ids = f'datasets/{DATASET}/query_ids.json'
path_corpus_ids = f'datasets/{DATASET}/corpus_ids.json'
path_qrels = f'datasets/{DATASET}/qrels_full.json'

if not os.path.exists(path_query_ids):
    raise FileNotFoundError(f"File '{path_query_ids}' not found.")
with open(path_query_ids, 'r') as f:
    query_ids = json.load(f)

if not os.path.exists(path_corpus_ids):
    raise FileNotFoundError(f"File '{path_corpus_ids}' not found.")
with open(path_corpus_ids, 'r') as f:
    corpus_ids = json.load(f)

if not os.path.exists(path_qrels):
    raise FileNotFoundError(f"File '{path_qrels}' not found.")
with open(path_qrels, "r") as f:
    qrels = json.load(f)


n_steps = 6
ROUNDING_TYPES = ['rounded_fakeround', 'gauss_noise', 'uniform_noise']

for ROUNDING_TYPE in ROUNDING_TYPES:
    recall_100 = []
    recall_1000 = []
    for i in tqdm(range(n_steps)):
        path_query_embeddings = f'datasets/{DATASET}/benchmarks/{ROUNDING_TYPE}/query_embeddings{i}.pt'
        path_corpus_embeddings = f'datasets/{DATASET}/benchmarks/{ROUNDING_TYPE}/corpus_embeddings{i}.pt'

        # Verify file existence
        if not os.path.exists(path_corpus_embeddings):
            raise FileNotFoundError(f"File '{path_corpus_embeddings}' not found.")
        if not os.path.exists(path_query_embeddings):
            raise FileNotFoundError(f"File '{path_query_embeddings}' not found.")

        # open up  encoded queries and encoded corpus
        query_embeddings = torch.load(path_query_embeddings)
        corpus_embeddings = torch.load(path_corpus_embeddings)

        # create model class
        model = DRESNE(query_ids, corpus_ids, query_embeddings, corpus_embeddings)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")

        # pass to retriever.retrieve corpus and query will be empty values, as they should be useless
        results = retriever.retrieve(corpus = {}, queries = {})
        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        recall_100.append(recall["Recall@100"])
        recall_1000.append(recall["Recall@1000"])

    # Data to be written to the CSV file
    headers = [[f"{ROUNDING_TYPE}"] + list(range(n_steps))]
    data = [
            ["Recall@100"] + recall_100,
            ["Recall@1000"] + recall_1000
            ]


    # Define the directory path
    directory_path = f'results/{DATASET}/benchmarks/'
    # Create the directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    # Define the CSV file path
    csv_file_path = f'{directory_path}recalls_{DATASET}_{ROUNDING_TYPE}.csv'

    # Write data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)

    # Logging info
        logging.info(f"Metrics are written to {csv_file_path}")
    print(f"Done saving Metrics written to {csv_file_path}")