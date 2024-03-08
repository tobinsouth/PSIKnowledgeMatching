# Matching Data Across Knowledgebases Privately

## Goal:
- Find a question-answering dataset / benchmarking. Run sbert / ada embeddings on this benchmark.
- Embed both the queries and the documents using the same embedding.
- Bin the embeddings in chunks (essentially round them to a decimal point).
- Match on the highest level of the embedding; those the match will be subdivided and the process is repeated.


## Repo
We're gonna use BEIR to do our benchmarking because it has some nice standard approaches. The problem is that it like to run the embeddings in real-time. `beir_reengineered.py` will help us change this to run the embeddings once and then save them.

PSI example shows how we can match things using PSI.