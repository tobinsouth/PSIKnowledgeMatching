# constants
# DATASET = "msmarco"
DATASET = "quora"
sbert_model_name = "msmarco-distilbert-base-tas-b"
device = "cpu" # cuda for gpu usage


# import libraries
from time import time
from beir import util
from beir_reengineered import NewSentenceBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import os, json, random
import pickle
import json


from beir.retrieval.search import BaseSearch # type: ignore beir/retrieval/search/dense/exact_search.py
from beir.util import cos_sim
import torch # type: ignore
import numpy as np
from typing import Dict
import logging
import csv

from experiement_retrieval_exact_search  import ExperiementRetrievalExactSearch as ERES # type: ignore

logger = logging.getLogger(__name__)



# #### Download nfcorpus.zip dataset and unzip the dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(DATASET)
# out_dir = "datasets"
# data_path = util.download_and_unzip(url, out_dir)


# corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
# # Save qrels to a JSON file
# with open(f"datasets/{DATASET}/qrels_full.json", "w") as f:
#     json.dump(qrels, f)


#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

beir_sbert = NewSentenceBERT(sbert_model_name, device=device)

path_query_embeddings = f"datasets/{DATASET}/query_embeddings_full.pkl"
path_corpus_embeddings = f"datasets/{DATASET}/corpus_embeddings_full.pkl"

with open(f"datasets/{DATASET}/qrels_full.json", "r") as f:
    qrels = json.load(f)

with open(path_query_embeddings, "rb") as f:
    query_embeddings = pickle.load(f)

with open(path_corpus_embeddings, "rb") as f:
    corpus_embeddings = pickle.load(f)

model = ERES(beir_sbert, path_query_embeddings=path_query_embeddings, path_corpus_embeddings=path_corpus_embeddings)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(model.corpus_embeddings, model.query_embeddings)
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)


# Data to be written to the CSV file
headers = ["Metric", "@1", "@3", "@5", "@10", "@100", "@1000"]
data = [
        ["NDCG"] + [ndcg[f"NDCG@{k}"] for k in retriever.k_values],
        ["MAP"] + [_map[f"MAP@{k}"] for k in retriever.k_values],
        ["Recall"] + [recall[f"Recall@{k}"] for k in retriever.k_values],
        ["Precision"] + [precision[f"P@{k}"] for k in retriever.k_values]
    ]

csv_file_path = f'results/{DATASET}/eval_metrics_ERES_{DATASET}.csv'
# Write data to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for row in data:
        writer.writerow(row)

# Logging info
    logging.info(f"Metrics are written to {csv_file_path}")






