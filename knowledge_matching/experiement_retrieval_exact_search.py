#TODO: refactor to have base class with rounding and noise

# import libraries
from time import time
import os, json, random
import pickle


from beir.retrieval.search import BaseSearch # type: ignore beir/retrieval/search/dense/exact_search.py
from beir.util import cos_sim
import torch # type: ignore
import numpy as np
from typing import Dict
import heapq
import logging
logger = logging.getLogger(__name__)


# ExperiementRetrievalExactSearch is parent class for any model we are using for our experiement that can be used for retrieval
# Abstract class is BaseSearch
class ExperiementRetrievalExactSearch(BaseSearch):
    def __init__(
            self,
            model,
            path_corpus_embeddings: str,
            path_query_embeddings: str,
            **kwargs):
        #model is class should do nothing
        self.model = model
        self.path_corpus_embeddings = path_corpus_embeddings
        self.path_query_embeddings = path_query_embeddings
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}

        logger.info("Load in Encoded Queries and Corpus from Pickle...")
        # Verify file existence
        if not os.path.exists(self.path_corpus_embeddings):
            raise FileNotFoundError(f"File '{self.path_corpus_embeddings}' not found.")
        if not os.path.exists(self.path_query_embeddings):
            raise FileNotFoundError(f"File '{self.path_query_embeddings}' not found.")

        with open(self.path_query_embeddings, "rb") as f:
            self.query_embeddings = pickle.load(f)

        with open(self.path_corpus_embeddings, "rb") as f:
            self.corpus_embeddings = pickle.load(f)



    def add_rounding(self, rounding_decimal: int) -> None:
        # rounding decimal
        if rounding_decimal < 12:
            logger.info("Rounding decimal places of Queries and Corpus...")
            for key, value in self.query_embeddings.items():
                self.query_embeddings[key] = np.round(value, decimals=rounding_decimal)

            for key, value in self.corpus_embeddings.items():
                self.corpus_embeddings[key] = np.round(value, decimals=rounding_decimal)

    def add_noise(self, rounding_decimal: int) -> None:
        logger.info("Adding Noise to Queries and Corpus...")
        for key, value in self.query_embeddings.items():
            self.query_embeddings[key] += np.random.random() / 10**rounding_decimal

        for key, value in self.corpus_embeddings.items():
            self.corpus_embeddings[key] += np.random.random() / 10**rounding_decimal


    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: int,
               score_function: str,
               return_sorted: bool = False,
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids

        query_ids = list(self.query_embeddings.keys())
        self.results = {qid: {} for qid in query_ids}

        # print("Sorting Corpus by document length (Longest first)...")
        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(list(self.corpus_embeddings.keys()), reverse=True)

        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query

        # Convert dictionary values to PyTorch tensors
        corpus_tensors = [torch.tensor(embedding) for embedding in self.corpus_embeddings.values()]
        query_tensors = [torch.tensor(embedding) for embedding in self.query_embeddings.values()]
        # Stack tensors along a new dimension (batch dimension)
        corpus_embeddings_tensor = torch.stack(corpus_tensors)
        query_embeddings_tensor = torch.stack(query_tensors)

        # print("Compute similarites using  cosine-similarity")
        # Compute similarites using  cosine-similarity
        cos_scores = cos_sim(query_embeddings_tensor, corpus_embeddings_tensor)
        cos_scores[torch.isnan(cos_scores)] = -1

        # print("Get top-k values")
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

        # print("build heap")
        for query_itr in range(len(query_embeddings_tensor)):
            query_id = query_ids[query_itr]
            for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                corpus_id = corpus_ids[sub_corpus_id]
                if corpus_id != query_id:
                    if len(result_heaps[query_id]) < top_k:
                        # Push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        print("get results heaps")
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

# ExperiementRetrievalExactSearch is parent class for any model we are using for our experiement that can be used for retrieval
# Abstract class is BaseSearch
class DenseRetrievalExactSearchNoEncoding(BaseSearch):
    def __init__(
            self,
            query_ids: list[str],
            corpus_ids: list[str],
            query_embeddings,
            corpus_embeddings,
            **kwargs):
        self.query_ids = query_ids
        self.corpus_ids = corpus_ids
        self.query_embeddings = query_embeddings
        self.corpus_embeddings = corpus_embeddings
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}

    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: int,
               score_function: str,
               return_sorted: bool = False,
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids

        query_ids = self.query_ids
        print(type(query_ids))
        self.results = {qid: {} for qid in query_ids}
        query_embeddings = self.query_embeddings

        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query

        corpus_ids = self.corpus_ids

        # print("Compute similarites using  cosine-similarity")
        # Compute similarites using  cosine-similarity
        cos_scores = cos_sim(query_embeddings, self.corpus_embeddings)
        cos_scores[torch.isnan(cos_scores)] = -1

        # print("Get top-k values")
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

        # print("build heap")
        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]
            for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                corpus_id = corpus_ids[sub_corpus_id]
                if corpus_id != query_id:
                    if len(result_heaps[query_id]) < top_k:
                        # Push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        print("get results heaps")
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

