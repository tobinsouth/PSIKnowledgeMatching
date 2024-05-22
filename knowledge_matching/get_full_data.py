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
logger = logging.getLogger(__name__)

#### Download nfcorpus.zip dataset and unzip the dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(DATASET)
out_dir = "datasets"
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files:
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
# Save qrels to a JSON file
with open(f"datasets/{DATASET}/qrels_full.json", "w") as f:
    json.dump(qrels, f)


#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

beir_sbert = NewSentenceBERT(sbert_model_name, device=device)



class GenericExactSearch(BaseSearch):

    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
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

        return self.results

model = DRES(beir_sbert, batch_size=256, corpus_chunk_size=512*9999)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)



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
corpus_embeddings_dict = dict(zip(corpus_ids, sub_corpus_embeddings))
query_embeddings_dict = dict(zip(queries.keys(), query_embeddings))
import pickle

with open(f"datasets/{DATASET}/corpus_embeddings_full.pkl", "wb") as f:
    pickle.dump(corpus_embeddings_dict, f)
with open(f"datasets/{DATASET}/query_embeddings_full.pkl", "wb") as f:
    pickle.dump(query_embeddings_dict, f)












with open(f"datasets/{DATASET}/query_embeddings_full.pkl", "rb") as f:
    query_embeddings = pickle.load(f)

with open(f"datasets/{DATASET}/corpus_embeddings_full.pkl", "rb") as f:
    corpus_embeddings = pickle.load(f)


retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus_embeddings, query_embeddings)

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
print( ndcg, _map, recall, precision )


# # ExperiementRetrievalExactSearch is parent class for any model we are using for our experiement that can be used for retrieval
# # Abstract class is BaseSearch
# class ExperiementRetrievalExactSearch(BaseSearch):
#     def __init__(
#             self,
#             model,
#             path_corpus_embeddings: str = "datasets/{DATASET}/corpus_embeddings.pkl",
#             path_query_embeddings: str = "datasets/{DATASET}/query_embeddings.pkl",
#             **kwargs):
#         #model is class should do nothing
#         self.model = model
#         self.path_corpus_embeddings = path_corpus_embeddings
#         self.path_query_embeddings = path_query_embeddings
#         self.show_progress_bar = kwargs.get("show_progress_bar", True)
#         self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
#         self.results = {}

#         logger.info("Load in Encoded Queries and Corpus from Pickle...")
#         # Verify file existence
#         if not os.path.exists(self.path_corpus_embeddings):
#             raise FileNotFoundError(f"File '{self.path_corpus_embeddings}' not found.")
#         if not os.path.exists(self.path_query_embeddings):
#             raise FileNotFoundError(f"File '{self.path_query_embeddings}' not found.")

#         with open(self.path_query_embeddings, "rb") as f:
#             self.query_embeddings = pickle.load(f)

#         with open(self.path_corpus_embeddings, "rb") as f:
#             self.corpus_embeddings = pickle.load(f)



#     def add_rounding(self, rounding_decimal: int) -> None:
#         # rounding decimal
#         if rounding_decimal < 12:
#             logger.info("Rounding decimal places of Queries and Corpus...")
#             for key, value in self.query_embeddings.items():
#                 self.query_embeddings[key] = np.round(value, decimals=rounding_decimal)

#             for key, value in self.corpus_embeddings.items():
#                 self.corpus_embeddings[key] = np.round(value, decimals=rounding_decimal)

#     def add_noise(self, rounding_decimal: int) -> None:
#         logger.info("Adding Noise to Queries and Corpus...")
#         for key, value in self.query_embeddings.items():
#             self.query_embeddings[key] += np.random.random() / 10**rounding_decimal

#         for key, value in self.corpus_embeddings.items():
#             self.corpus_embeddings[key] += np.random.random() / 10**rounding_decimal


#     def search(self,
#                corpus: Dict[str, Dict[str, str]],
#                queries: Dict[str, str],
#                top_k: int,
#                score_function: str,
#                return_sorted: bool = False,
#                **kwargs) -> Dict[str, Dict[str, float]]:
#         # Runs semantic search against the corpus embeddings
#         # Returns a ranked list with the corpus ids

#         query_ids = list(self.query_embeddings.keys())
#         self.results = {qid: {} for qid in query_ids}

#         # print("Sorting Corpus by document length (Longest first)...")
#         logger.info("Sorting Corpus by document length (Longest first)...")
#         corpus_ids = sorted(list(self.corpus_embeddings.keys()), reverse=True)

#         result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query

#         # Convert dictionary values to PyTorch tensors
#         corpus_tensors = [torch.tensor(embedding) for embedding in self.corpus_embeddings.values()]
#         query_tensors = [torch.tensor(embedding) for embedding in self.query_embeddings.values()]
#         # Stack tensors along a new dimension (batch dimension)
#         corpus_embeddings_tensor = torch.stack(corpus_tensors)
#         query_embeddings_tensor = torch.stack(query_tensors)

#         # print("Compute similarites using  cosine-similarity")
#         # Compute similarites using  cosine-similarity
#         cos_scores = cos_sim(query_embeddings_tensor, corpus_embeddings_tensor)
#         cos_scores[torch.isnan(cos_scores)] = -1

#         # print("Get top-k values")
#         # Get top-k values
#         cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
#         cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
#         cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

#         # print("build heap")
#         for query_itr in range(len(query_embeddings_tensor)):
#             query_id = query_ids[query_itr]
#             for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
#                 corpus_id = corpus_ids[sub_corpus_id]
#                 if corpus_id != query_id:
#                     if len(result_heaps[query_id]) < top_k:
#                         # Push item on the heap
#                         heapq.heappush(result_heaps[query_id], (score, corpus_id))
#                     else:
#                         # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
#                         heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

#         print("get results heaps")
#         for qid in result_heaps:
#             for score, corpus_id in result_heaps[qid]:
#                 self.results[qid][corpus_id] = score

#         return self.results


# beir_sbert = NewSentenceBERT(sbert_model_name, device=device)



# model = ExperiementRetrievalExactSearch(
#     beir_sbert,
#     path_corpus_embeddings = f"datasets/{DATASET}/corpus_embeddings_full.pkl",
#     path_query_embeddings= f"datasets/{DATASET}/query_embeddings_full.pkl") # This can go outside the loop

# print("here")
# with open(f"datasets/{DATASET}/qrels_full.json", "r") as f:
#     qrels = json.load(f)

# with open(f"datasets/{DATASET}/query_embeddings_full.pkl", "rb") as f:
#     query_embeddings = pickle.load(f)

# with open(f"datasets/{DATASET}/corpus_embeddings_full.pkl", "rb") as f:
#     corpus_embeddings = pickle.load(f)


# rounding_recalls = []
# rounding_decimals = range(12,-1,-1)
# for decimal in rounding_decimals:
#     print(decimal, type(decimal))
#     model.add_rounding(decimal)
#     print("after add_rounding")
#     retriever = EvaluateRetrieval(model, score_function="cos_sim")
#     print("after EvaluateRetrieval")
#     # print(model.corpus_embeddings, model.query_embeddings)
#     # issue maybe you have to pass the correct corpus and queries in
#     results = retriever.retrieve(corpus_embeddings, query_embeddings)
#     print("after retrieve")
#     ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
#     print("decimal_places:", decimal , ndcg, _map, recall, precision )
#     #recall@100 matters mostr, save to array for plotting
#     rounding_recalls.append(recall)


# noise_recalls = []
# noise_decimals = range(12,-1,-1)
# for decimal in range(12,-1,-1):
#     # print(decimal, type(decimal))
#     model.add_noise(decimal)
#     # model.noise()
#     retriever = EvaluateRetrieval(model, score_function="cos_sim")
#     results = retriever.retrieve(model.corpus_embeddings, model.query_embeddings)
#     ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
#     print("noise decimal:", decimal , ndcg, _map, recall, precision )
#     noise_recalls.append(recall)


# import csv
# # Headers for CSV file
# headers = ['Rounding_Decimals', 'Recall@100', 'Recall@1000']

# # Create and write data to CSV file
# with open('results/rounding_recalls_full.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=headers)
#     writer.writeheader()

#     for rounding_decimal, rounding_recall in zip(rounding_decimals, rounding_recalls):
#         row_data = {
#             'Rounding_Decimals': rounding_decimal,
#             'Recall@100': rounding_recall['Recall@100'],
#             'Recall@1000': rounding_recall['Recall@1000']
#         }
#         writer.writerow(row_data)

# # Headers for CSV file
# headers = ['Rounding_Decimals', 'Recall@100', 'Recall@1000']

# # Create and write data to CSV file
# with open('results/noise_recall_full.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=headers)
#     writer.writeheader()

#     for rounding_decimal, noise_recall in zip(rounding_decimals, noise_recalls):
#         row_data = {
#             'Rounding_Decimals': rounding_decimal,
#             'Recall@100': noise_recall['Recall@100'],
#             'Recall@1000': noise_recall['Recall@1000']
#         }
#         writer.writerow(row_data)