# Matching Data Across Knowledgebases Privately

## Goal:
- Find a question-answering dataset / benchmarking. Run sbert / ada embeddings on this benchmark.
- Embed both the queries and the documents using the same embedding.
- Bin the embeddings in chunks (essentially round them to a decimal point).
- Match on the highest level of the embedding; those the match will be subdivided and the process is repeated.


## Repo
We're gonna use BEIR to do our benchmarking because it has some nice standard approaches. The problem is that it like to run the embeddings in real-time. `beir_reengineered.py` will help us change this to run the embeddings once and then save them.

PSI example shows how we can match things using PSI.

## Automated Benchmarking Pipeline

The pipeline for obtaining rounding benchmarks for any dataset has been automated. The steps to get rounding benchmarks are:

1. **Download and Encode the Dataset:** Encode the queries and corpus (documents) of the dataset and save these embeddings for later use. See `get_baseline_no_round_msmarco.py` for an example.
2. **Round the Embeddings:** Apply different rounding methods as defined in `tobin/tools.py`. Use `round_embeddings.py` for this step.
3. **Calculate Recalls:** Determine the recalls (recall@100 and recall@1000) for all rounding types. This is handled by `get_recalls.py`.
4. **Plot Recalls:** Compare the recalls with the baseline. Refer to `plot_recalls_msmarco.py` for an example using the MSMARCO dataset.

### Running the Pipeline on Any Dataset

To run this pipeline on any dataset, follow these steps:

1. **Set the Dataset:**
   - Modify `DATASET = "msmarco"` at the top of each script to your preferred dataset from [BEIR](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/).

2. **Update System Path:**
   - Change the system path to the location of this repository on your system in each script. For example, `sys.path.append('/u/tsouth/projects/PSIKnowledgeMatching/')`.

3. **Run the Scripts:**
   - Execute each script in your terminal:
     ```sh
     python3 get_baseline_no_round_msmarco.py && python3 round_embeddings.py && python3 get_recalls.py && python3 plot_recalls_msmarco.py
     ```

### Note

- For large datasets like MSMARCO, encoding the corpus can take a long time. It is recommended to periodically save the embeddings. Refer to `get_baseline_no_round_msmarco_checkpoints.py` for an example of how this can be done.

By following these steps, you can efficiently benchmark and analyze the performance of different rounding methods on various datasets.