# This file will aggregate the results of the retrieval benchmarking


# @ Shayla task
# 1. Load in corpus of queries and documents
# 2. Allow us to choose how we change these documents (noise, round, nothing).
# 3. Run the retrieval on the documents with the chosen changes.
# 4. Save the results.


from params import DATASET, MODELNAME, device
from tqdm import tqdm
import pandas as pd
import os

from knowledge_matching.beir_retrieval_main import download_dataset, encode_queries_and_corpus, rounding_and_retrieval, check_base_retrieval_metrics, sample_dataset

data_path= download_dataset(DATASET)
dataset = encode_queries_and_corpus(DATASET, MODELNAME, data_path, device, sample_p=0.5)
print("Finished encoding on", dataset, data_path)

base_retrieval_metrics = check_base_retrieval_metrics(dataset)
base_retrieval_metrics_df = pd.DataFrame([base_retrieval_metrics[1]], columns=base_retrieval_metrics[0])
print("Base Recall@10:", base_retrieval_metrics_df.loc[0, "Recall@10"])

rounding_results = []
for rounding_type in ["round", "guass", "uniform"]:
    for rounding_p in tqdm([1,2,3,4,5,6,7,8,9,10,12,15,20]):
        for round_corpus in [True, False]:
            headers, data, diff = rounding_and_retrieval(dataset, rounding_type, round_corpus=round_corpus, round_param=rounding_p)
            rounding_results.append([rounding_type, rounding_p, diff, round_corpus] + data )

headers, data, diff = rounding_and_retrieval(dataset, None, round_corpus=False, round_param=None)
base_retrieval_results_2 = ['None', 0, 0., False] + data

rounding_result_df = pd.DataFrame(rounding_results+[base_retrieval_results_2], columns=["rounding_type", "rounding_p", "diff", "round_corpus"]+headers)

os.makedirs("results", exist_ok=True)
rounding_result_df.to_csv(f"results/rounding_results_{dataset}.csv", index=False)






# Now we want to benchmark on set intersections