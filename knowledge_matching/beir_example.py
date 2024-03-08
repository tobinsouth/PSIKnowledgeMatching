"""This file is going to use BEIR to download a dataset and create embeddings of it. In can then evaluate the performance of those embeddings.
 By default, it will use a subset of a standard dataset for speed reasons, but full benchmarks are run for the paper."""


DATASET = "quora"
sbert_model_name = "msmarco-distilbert-base-tas-b"
device = "cuda:2" # cuda for gpu usage
k_queries = 15
k_documents = 10000

from time import time
from beir import util
from beir_reengineered import NewSentenceBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import os, json, random


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

#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

beir_sbert = NewSentenceBERT(sbert_model_name, device=device)
model = DRES(beir_sbert, batch_size=256, corpus_chunk_size=512*9999)

# Embed documents and queries, save them for PSI
subset_of_queries = random.sample(queries.keys(), k_queries)
queries = {qid: queries[qid] for qid in subset_of_queries}
qrels = {qid: qrels[qid] for qid in subset_of_queries}
true_documents = set([docid for qid in qrels for docid in qrels[qid]])
false_documents = set(random.sample(list(set([docid for docid in corpus if docid not in true_documents])), k_documents))
subset_of_corpus = true_documents | false_documents
corpus = {docid: corpus[docid] for docid in subset_of_corpus}

queries_l = [queries[qid] for qid in queries]
query_embeddings = model.model.encode_queries(
            queries_l, batch_size=model.batch_size, show_progress_bar=model.show_progress_bar, convert_to_tensor=model.convert_to_tensor).cpu().numpy()

corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
corpus_l = [corpus[cid] for cid in corpus_ids]
sub_corpus_embeddings = model.model.encode_corpus(
        corpus_l,
        batch_size=model.batch_size,
        show_progress_bar=model.show_progress_bar, 
        convert_to_tensor = model.convert_to_tensor
        ).cpu().numpy()

# Save as new dataset
os.makedirs("datasets/subquora/qrels", exist_ok=True)
with open("datasets/subquora/queries.jsonl", "w") as f:
    f.writelines([json.dumps({"_id": qid, "text": queries[qid], "metadata":{}})+"\n" for qid in queries])
with open("datasets/subquora/corpus.jsonl", "w") as f:
    f.writelines([json.dumps({"_id": docid, "title": corpus[docid].get("title"), "text": corpus[docid].get("text"), "metadata":{}})+"\n" for docid in corpus])
with open("datasets/subquora/qrels/test.tsv", "w") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for qid in qrels:
        for docid in qrels[qid]:
            f.write("{}\t{}\t{}\n".format(qid, docid, qrels[qid][docid]))

# Save embeddings
corpus_embeddings_dict = dict(zip(corpus_ids, sub_corpus_embeddings))
query_embeddings_dict = dict(zip(queries.keys(), query_embeddings))
import pickle
with open("datasets/subquora/corpus_embeddings.pkl", "wb") as f:
    pickle.dump(corpus_embeddings_dict, f)
with open("datasets/subquora/query_embeddings.pkl", "wb") as f:
    pickle.dump(query_embeddings_dict, f)





# retriever = EvaluateRetrieval(model, score_function="dot")

# #### Retrieve dense results (format of results is identical to qrels)
# start_time = time()
# results = retriever.retrieve(corpus, queries)
# end_time = time()
# print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
# #### Evaluate your retrieval using NDCG@k, MAP@K ...

# print("Retriever evaluation for k in: {}".format(retriever.k_values))
# ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

# mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
# recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
# hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

# #### Print top-k documents retrieved ####
# top_k = 10

# query_id, ranking_scores = random.choice(list(results.items()))
# scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
# print("Query : %s\n" % queries[query_id])

# for rank in range(top_k):
#     doc_id = scores_sorted[rank][0]
#     # Format: Rank x: ID [Title] Body
#     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))