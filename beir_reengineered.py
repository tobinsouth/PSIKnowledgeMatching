from typing import List, Dict, Union, Tuple
from torch import Tensor
from sentence_transformers import SentenceTransformer
from beir.retrieval import models

class NewSentenceBERT(models.SentenceBERT):
    def __init__(self, model_path, device):
        self.q_model = SentenceTransformer(model_path, device=device)
        self.doc_model = self.q_model
        self.sep = " "