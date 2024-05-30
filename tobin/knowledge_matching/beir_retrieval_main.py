"""
Summary:
    This script downloads the MSMARCO dataset, encodes the queries and corpus using a pre-trained SBERT model,
    and saves the embeddings without applying any rounding.

Usage:
    python3 get_baseline_no_round_msmarco.py

Dependencies:
    - requests
    - numpy
    - sentence-transformers
    - beir
    - torch

Author:
    Shayla Nguyen

Date:
    2024-05-23
"""


from time import time
from beir import util
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

from knowledge_matching.ExperiementRetrievalExactSearch import DenseRetrievalExactSearchNoEncoding as DRESNE # type: ignore


def download_dataset(DATASET: str):
    """This is a simple function to download the dataset and use the built in loader."""
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(DATASET)
    out_dir = "datasets"
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test") # checking dataloader works
    return data_path


def sample_dataset(DATASET: str, p: float, queries, corpus, qrels):
    """This function samples a subset of the dataset for testing. This may need revision. It also defines a new dataset name for reproducibility."""
    # @Shayla needed here
    subset_of_queries = random.sample(queries.keys(), int(p*len(queries)))
    queries = {qid: queries[qid] for qid in subset_of_queries}
    qrels = {qid: qrels[qid] for qid in subset_of_queries}
    true_documents = set([docid for qid in qrels for docid in qrels[qid]])
    false_documents = set(random.sample(list(set([docid for docid in corpus if docid not in true_documents])), int(p*len(corpus))))
    subset_of_corpus = true_documents | false_documents
    corpus = {docid: corpus[docid] for docid in subset_of_corpus}

    DATASET = f"{DATASET}_sample_{p}"
    return queries, corpus, qrels, DATASET


def encode_queries_and_corpus(DATASET: str, MODELNAME: str, data_path: str, device: str, sample_p: float):
    """This function encodes the queries and corpus and saves the embeddings to the dataset. Optionally this will create a sampled dataset and save it."""

    from params import BEIR_BATCH_SIZE, BEIR_CORPUS_CHUNK_SIZE
    from knowledge_matching.beir_reengineered import NewSentenceBERT

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    beir_sbert = NewSentenceBERT(MODELNAME, device=device)

    if sample_p != 1.0:
        queries, corpus, qrels, DATASET = sample_dataset(DATASET, sample_p, queries, corpus, qrels)

    os.makedirs(f"datasets/{DATASET}", exist_ok=True)

    # encode queries and save for future use
    queries_list = [queries[qid] for qid in queries]
    query_embeddings = beir_sbert.encode_queries(
                queries_list,
                batch_size=BEIR_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_tensor=False,
                convert_to_numpy=False
                )
    
    query_embeddings = [e.cpu() for e in query_embeddings]
    query_embeddings = torch.stack(query_embeddings)
    torch.save(query_embeddings, f'datasets/{DATASET}/query_embeddings.pt')

    # encode corpus and save for future use
    corpus_list = [corpus[cid] for cid in corpus]
    corpus_embeddings = beir_sbert.encode_corpus(
        corpus_list,
        batch_size=BEIR_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=False,
        convert_to_numpy=False        
    )
    corpus_embeddings = [e.cpu() for e in corpus_embeddings]
    corpus_embeddings = torch.stack(corpus_embeddings)
    torch.save(corpus_embeddings, f'datasets/{DATASET}/corpus_embeddings.pt')

    # Now save the queries and corpus to the dataset
    with open(f'datasets/{DATASET}/queries.json', 'w') as f:
        json.dump(queries, f)
    with open(f'datasets/{DATASET}/corpus.json', 'w') as f:
        json.dump(corpus, f)
    with open(f'datasets/{DATASET}/qrels.json', 'w') as f:
        json.dump(qrels, f)

    return DATASET

def check_base_retrieval_metrics(DATASET: str):
    """This function checks the retrieval metrics for the dataset."""

    with open(f'datasets/{DATASET}/queries.json', 'r') as f:
        queries = json.load(f)
    with open(f'datasets/{DATASET}/corpus.json', 'r') as f:
        corpus = json.load(f)
    with open(f'datasets/{DATASET}/qrels.json', 'r') as f:
        qrels = json.load(f)
    
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

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())

    # create model class
    model = DRESNE(query_ids, corpus_ids, query_embeddings, corpus_embeddings)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    # pass to retriever.retrieve corpus and query will be empty values, as they should be useless
    results = retriever.retrieve(corpus = {}, queries = {})
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    # Data to be written to the CSV file
    headers = [f"NDCG@{k}" for k in retriever.k_values] + [f'MAP@{k}' for k in retriever.k_values] + [f'Recall@{k}' for k in retriever.k_values] + [f'P@{k}' for k in retriever.k_values]
    data = [ndcg[f"NDCG@{k}"] for k in retriever.k_values] + \
            [_map[f"MAP@{k}"] for k in retriever.k_values] + \
            [recall[f"Recall@{k}"] for k in retriever.k_values] + \
            [precision[f"P@{k}"] for k in retriever.k_values]
    
    print("Done saving base metrics for ", DATASET)
    return headers, data

import sys; sys.path.append('..')
from tools import get_edges, fakeround, gauss_noise, uniform_noise
def rounding_and_retrieval(DATASET: str, rounding_type: str, round_corpus: bool, round_param: int):
    """This is a copy of the check_base_retrieval_metrics function, but with the addition of rounding."""
    
    with open(f'datasets/{DATASET}/queries.json', 'r') as f:
        queries = json.load(f)
    with open(f'datasets/{DATASET}/corpus.json', 'r') as f:
        corpus = json.load(f)
    with open(f'datasets/{DATASET}/qrels.json', 'r') as f:
        qrels = json.load(f)
    
    path_query_embeddings = f'datasets/{DATASET}/query_embeddings.pt'
    path_corpus_embeddings = f'datasets/{DATASET}/corpus_embeddings.pt'

    query_embeddings = torch.load(path_query_embeddings)
    corpus_embeddings = torch.load(path_corpus_embeddings)

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())

    if rounding_type is None:
        rounding_type = "none"
    else:
        query_embedding_range = [query_embeddings.min().item(), query_embeddings.max().item()]
        corpus_embedding_range = [corpus_embeddings.min().item(), corpus_embeddings.max().item()]
        embedding_range = [min(query_embedding_range + corpus_embedding_range), max(query_embedding_range+corpus_embedding_range)]
        edges, diff = get_edges(round_param, embedding_range)
    
    if rounding_type == "round":
        query_embeddings = fakeround(edges, query_embeddings.to('cpu'))
        if round_corpus:
            corpus_embeddings = fakeround(edges, corpus_embeddings.to('cpu'))

    if rounding_type == "guass":
        query_embeddings = gauss_noise(query_embeddings, diff)
        if round_corpus:
            corpus_embeddings = gauss_noise(corpus_embeddings, diff)

    if rounding_type == "uniform":
        query_embeddings = uniform_noise(query_embeddings, diff)
        if round_corpus:
            corpus_embeddings = uniform_noise(corpus_embeddings, diff)

    model = DRESNE(query_ids, corpus_ids, query_embeddings, corpus_embeddings)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")
    results = retriever.retrieve(corpus = {}, queries = {})
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    headers = [f"NDCG@{k}" for k in retriever.k_values] + [f'MAP@{k}' for k in retriever.k_values] + [f'Recall@{k}' for k in retriever.k_values] + [f'P@{k}' for k in retriever.k_values]
    data = [ndcg[f"NDCG@{k}"] for k in retriever.k_values] + \
            [_map[f"MAP@{k}"] for k in retriever.k_values] + \
            [recall[f"Recall@{k}"] for k in retriever.k_values] + \
            [precision[f"P@{k}"] for k in retriever.k_values]

    return headers, data









