"""This file is going to use BEIR to download a dataset and create embeddings of it."""

# %%
DATASET = "quora"
sbert_model_name = "msmarco-distilbert-base-tas-b"
device = "cpu" # cuda for gpu usage
k_queries = 100
k_documents = 10000

# %%
from time import time
from beir import util
from beir_reengineered import NewSentenceBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import os, json, random
# %%
#### Download nfcorpus.zip dataset and unzip the dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(DATASET)
out_dir = "datasets"
data_path = util.download_and_unzip(url, out_dir)

# %%
#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files:
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# %%
#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

beir_sbert = NewSentenceBERT(sbert_model_name, device=device)
model = DRES(beir_sbert, batch_size=256, corpus_chunk_size=512*9999)

# %%
# Create sub-sample
subset_of_queries = random.sample(queries.keys(), k_queries)
queries = {qid: queries[qid] for qid in subset_of_queries}
qrels = {qid: qrels[qid] for qid in subset_of_queries}
true_documents = set([docid for qid in qrels for docid in qrels[qid]])
false_documents = set(random.sample(list(set([docid for docid in corpus if docid not in true_documents])), k_documents))
subset_of_corpus = true_documents | false_documents
corpus = {docid: corpus[docid] for docid in subset_of_corpus}

# %%
# Encode queries
queries_l = [queries[qid] for qid in queries]
query_embeddings = model.model.encode_queries(
    queries_l,
    batch_size=model.batch_size,
    show_progress_bar=model.show_progress_bar,
    convert_to_tensor=model.convert_to_tensor
).cpu().numpy()
# %%
# Encode documents
corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
corpus_l = [corpus[cid] for cid in corpus_ids]
sub_corpus_embeddings = model.model.encode_corpus(
    corpus_l,
    batch_size=model.batch_size,
    show_progress_bar=model.show_progress_bar,
    convert_to_tensor=model.convert_to_tensor
).cpu().numpy()

# %%
# # Save embeddings # broken
corpus_embeddings_dict = dict(zip(corpus_ids, sub_corpus_embeddings))
query_embeddings_dict = dict(zip(queries.keys(), query_embeddings))
import pickle

with open("datasets/subquora/corpus_embeddings.pkl", "wb") as f:
    pickle.dump(corpus_embeddings_dict, f)
with open("datasets/subquora/query_embeddings.pkl", "wb") as f:
    pickle.dump(query_embeddings_dict, f)

# %%
# Bin the embeddings in chunks
import numpy as np
# Specify the number of decimal points to round to
decimal_points = 2

# Bin corpus embeddings
binned_corpus_embeddings = np.round(sub_corpus_embeddings, decimals=decimal_points)

# Bin query embeddings
binned_query_embeddings = np.round(query_embeddings, decimals=decimal_points)

# %%
# Match on the highest level of the embedding and subdivide the process iteratively
from sklearn.cluster import AgglomerativeClustering

def hierarchical_clustering(embeddings, num_clusters):
    # safe guard
    num_samples = embeddings.shape[0]  # Get the number of samples in the data
    if num_clusters > num_samples:  # Check if the number of clusters is greater than the number of samples
        return {0: embeddings}  # If so, return the entire embeddings as a single cluster

    # Perform hierarchical clustering
    clusterer = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    clusters = clusterer.fit_predict(embeddings)

    # Initialize an empty dictionary to store subclusters
    subclusters = {}

    # Iterate over each cluster
    for cluster_id in range(num_clusters):
        # Get indices of embeddings belonging to the current cluster
        cluster_indices = np.where(clusters == cluster_id)[0]

        # Check if there are more than one embedding in the cluster
        if len(cluster_indices) > 1:
            # Recursively call hierarchical clustering on the subset of embeddings
            subembeddings = embeddings[cluster_indices]
            subclusters[cluster_id] = hierarchical_clustering(subembeddings, num_clusters)
        else:
            # If only one embedding in the cluster, store it directly
            subclusters[cluster_id] = embeddings[cluster_indices[0]]

    return subclusters

# %%
# Specify the number of clusters for the highest level of clustering
num_clusters_initial = 4

# Perform hierarchical clustering on query embeddings
query_clusters = hierarchical_clustering(binned_query_embeddings, num_clusters_initial)

# Perform hierarchical clustering on corpus embeddings
corpus_clusters = hierarchical_clustering(binned_corpus_embeddings, num_clusters_initial)

# Now 'query_clusters' and 'corpus_clusters' contain hierarchical clustering information for query and corpus embeddings respectively.



# %%
# testing different num clusters
query_clusters_15 = hierarchical_clustering(binned_query_embeddings, 15)
corpus_clusters_10025 = hierarchical_clustering(binned_corpus_embeddings, 10025)


# %%
# Evaluating Embeddings by hand
import torch, numpy as np

# for qid, query in query_embeddings_dict.items():
#     # Find cosine sim to all doc
#     distances = torch.cosine_similarity(torch.tensor(query_embeddings[0]), torch.tensor(sub_corpus_embeddings))
#     # Select the most similar doc (or docs) (or just those below a threshold)
#     best_doc = distances.argmin()
#     # Check if the best doc is in the qrels


# Define the number of top documents to consider
k_documents = 10

# Initialize evaluation metrics
total_queries = 0
total_correct = 0

for qid, query in query_embeddings_dict.items():
    # Find cosine similarity to all documents
    query_embedding = torch.tensor(query).unsqueeze(0)  # Convert to tensor and add batch dimension
    distances = torch.cosine_similarity(query_embedding, torch.tensor(sub_corpus_embeddings))

    # Select the most similar document indices
    best_doc_indices = distances.argsort(descending=True)[:k_documents]  # Select top k_documents documents
    # Retrieve corresponding document IDs
    best_docs = [corpus_ids[idx] for idx in best_doc_indices]

    # Check if the best docs are in the qrels for the current query
    if qid in qrels:
        relevant_docs = qrels[qid].keys()
        total_queries += 1

        # Check if any of the relevant documents are in the top k_documents retrieved
        if any(doc_id in relevant_docs for doc_id in best_docs):
            total_correct += 1
        # if set(best_docs).intersection(relevant_docs):
        #     total_correct += 1


# Calculate accuracy
accuracy = total_correct / total_queries if total_queries > 0 else 0
print("Accuracy:", accuracy)
print("total_correct:", total_correct, "total_queries:", total_queries)

# %%




# %%