# constants
# DATASET = "msmarco"
DATASET = "scifact"
sbert_model_name = "msmarco-distilbert-base-tas-b"
device = "cpu" # cuda for gpu usage


# import libraries
from time import time
from beir import util
from beir_reengineered import NewSentenceBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import os, json, random
import pickle
import json


from beir.retrieval.search import BaseSearch # type: ignore beir/retrieval/search/dense/exact_search.py
from beir.util import cos_sim
import torch # type: ignore
import numpy as np
from typing import Dict
import heapq
import logging
import csv
logger = logging.getLogger(__name__)

#### Download nfcorpus.zip dataset and unzip the dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(DATASET)
out_dir = "datasets"
data_path = util.download_and_unzip(url, out_dir)


corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
# Save qrels to a JSON file
with open(f"datasets/{DATASET}/qrels_full.json", "w") as f:
    json.dump(qrels, f)


#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

beir_sbert = NewSentenceBERT(sbert_model_name, device=device)

model = DRES(beir_sbert, batch_size=256, corpus_chunk_size=512*9999)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)
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

csv_file_path = f'results/{DATASET}/eval_metrics_DRES_{DATASET}.csv'
# Write data to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for row in data:
        writer.writerow(row)

# Logging info
    logging.info(f"Metrics are written to {csv_file_path}")



# Encode queries
queries_l = [queries[qid] for qid in queries]
query_embeddings = model.model.encode_queries(
    queries_l,
    batch_size=model.batch_size,
    show_progress_bar=model.show_progress_bar,
    convert_to_tensor=model.convert_to_tensor
).cpu().numpy()


# Encode documents
corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
corpus_l = [corpus[cid] for cid in corpus_ids]
corpus_embeddings = model.model.encode_corpus(
    corpus_l,
    batch_size=model.batch_size,
    show_progress_bar=model.show_progress_bar,
    convert_to_tensor=model.convert_to_tensor
).cpu().numpy()


# Save as new dataset
os.makedirs(f"datasets/{DATASET}/qrels", exist_ok=True)
with open(f"datasets/{DATASET}/queries.jsonl", "w") as f:
    f.writelines([json.dumps({"_id": qid, "text": queries[qid], "metadata":{}})+"\n" for qid in queries])
with open(f"datasets/{DATASET}/corpus.jsonl", "w") as f:
    f.writelines([json.dumps({"_id": docid, "title": corpus[docid].get("title"), "text": corpus[docid].get("text"), "metadata":{}})+"\n" for docid in corpus])
with open(f"datasets/{DATASET}/qrels/test.tsv", "w") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for qid in qrels:
        for docid in qrels[qid]:
            f.write("{}\t{}\t{}\n".format(qid, docid, qrels[qid][docid]))


# # Save embeddings
corpus_embeddings_dict = dict(zip(corpus_ids, corpus_embeddings))
query_embeddings_dict = dict(zip(queries.keys(), query_embeddings))
import pickle

with open(f"datasets/{DATASET}/corpus_embeddings_full.pkl", "wb") as f:
    pickle.dump(corpus_embeddings_dict, f)
with open(f"datasets/{DATASET}/query_embeddings_full.pkl", "wb") as f:
    pickle.dump(query_embeddings_dict, f)
