from typing import List, Dict, Union, Tuple
from torch import Tensor
from sentence_transformers import SentenceTransformer
from beir.retrieval import models

class NewSentenceBERT(models.SentenceBERT):
    """This BERT class allows us to explicitly define which model we want to use for the SentenceTransformer and direct it to the right GPU. We can also add support for ada / openai embeddings here."""
    def __init__(self, model_path, device):
        self.q_model = SentenceTransformer(model_path, device=device)
        self.doc_model = self.q_model
        self.sep = " "