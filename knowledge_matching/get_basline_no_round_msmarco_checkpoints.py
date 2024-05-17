# constants
DATASET = "msmarco"
# DATASET = "scifact"
# DATASET = "quora"
sbert_model_name = "msmarco-distilbert-base-tas-b"  # type: ignore
device = "cpu" # cuda for gpu usage
batch_size=256
corpus_chunk_size=512*9999


# import libraries
from time import time
from beir import util
from beir_reengineered import NewSentenceBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import os, json, random
import pickle
import json
from tqdm import tqdm


from beir.retrieval.search import BaseSearch # type: ignore beir/retrieval/search/dense/exact_search.py
from beir.util import cos_sim
import torch # type: ignore
import numpy as np
from typing import Dict
import logging
import csv

from experiement_retrieval_exact_search  import DenseRetrievalExactSearchNoEncoding as DRESNE # type: ignore

logger = logging.getLogger(__name__)



# #### Download nfcorpus.zip dataset and unzip the dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(DATASET)
out_dir = "datasets"
data_path = util.download_and_unzip(url, out_dir)


corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

beir_sbert = NewSentenceBERT(sbert_model_name, device=device)


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



path_query_embeddings = f'datasets/{DATASET}/query_embeddings.pt'

if not os.path.exists(path_query_embeddings):
    raise FileNotFoundError(f"File '{path_query_embeddings}' not found.")

# open up  encoded queries
query_embeddings = torch.load(path_query_embeddings)

## Checkpoints
checkpoint_interval = 100  # Save checkpoint every 100 steps
# Paths
checkpoint_path = f'datasets/{DATASET}/corpus_embeddings_save_checkpoint.pt'
final_path = f'datasets/{DATASET}/corpus_embeddings.pt'

# encode corpus and save for future use
corpus = [corpus[cid] for cid in corpus_ids]

# Function to load embeddings if checkpoint exists
def load_checkpoint():
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        return torch.load(checkpoint_path)
    return None

# Initialize or load checkpoint
corpus_embeddings = load_checkpoint()
start_idx = len(corpus_embeddings) if corpus_embeddings is not None else 0


if corpus_embeddings is None:
    corpus_embeddings = []

# Encode corpus and save periodically
for i in tqdm(range(start_idx, len(corpus), batch_size), desc="Encoding corpus"):
    batch = corpus[i:i + batch_size]
    batch_embeddings = beir_sbert.encode_corpus(
        batch,
        batch_size=batch_size,
        show_progress_bar=False,  # Disable internal progress bar as we're using tqdm
        convert_to_tensor=True
    )
    corpus_embeddings.extend(batch_embeddings)

    # Save checkpoint
    if (i // batch_size + 1) % checkpoint_interval == 0:
        print(f"Saving checkpoint at step {i}...")
        torch.save(corpus_embeddings, checkpoint_path)

# Save the final embeddings
torch.save(corpus_embeddings, final_path)

# Clean up checkpoint
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

print(f"Corpus embeddings saved to {final_path}")

# corpus_embeddings = beir_sbert.encode_corpus(
#     corpus,
#     batch_size=batch_size,
#     show_progress_bar=True,
#     convert_to_tensor=True
# )
# torch.save(corpus_embeddings, f'datasets/{DATASET}/corpus_embeddings.pt')



path_corpus_embeddings = f'datasets/{DATASET}/corpus_embeddings.pt'

# Verify file existence
if not os.path.exists(path_corpus_embeddings):
    raise FileNotFoundError(f"File '{path_corpus_embeddings}' not found.")

# open up  encoded queries and encoded corpus
# query_embeddings = torch.load(path_query_embeddings)
corpus_embeddings = torch.load(path_corpus_embeddings)



# create model class
model = DRESNE(query_ids, corpus_ids, query_embeddings, corpus_embeddings)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

# pass to retriever.retrieve corpus and query will be empty values, as they should be useless
results = retriever.retrieve(corpus = {}, queries = {})
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

csv_file_path = f'results/{DATASET}/eval_metrics_DRESNE_{DATASET}.csv'
# Write data to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for row in data:
        writer.writerow(row)

# Logging info
    logging.info(f"Metrics are written to {csv_file_path}")
print("Done saving Metrics")






