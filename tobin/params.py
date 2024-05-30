import torch

MODELNAME = "sentence-transformers/all-MiniLM-L6-v2"
USE_INVERTER = False
device = "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"
DATASET = "quora"
BEIR_BATCH_SIZE=256
BEIR_CORPUS_CHUNK_SIZE=512*9999