# This file will aggregate the results of the retrieval benchmarking


# @ Shayla task
# 1. Load in corpus of queries and documents
# 2. Allow us to choose how we change these documents (noise, round, nothing).
# 3. Run the retrieval on the documents with the chosen changes.
# 4. Save the results.


from params import DATASET, MODELNAME, device

from knowledge_matching.beir_retrieval_main import download_dataset, encode_queries_and_corpus, rounding_and_retrieval, check_base_retrieval_metrics, sample_dataset

data_path= download_dataset(DATASET)
dataset = encode_queries_and_corpus(DATASET, MODELNAME, data_path, device, sample_p=0.3)
print("Finished encoding on", dataset)

base_retrieval_metrics = check_base_retrieval_metrics(dataset)

