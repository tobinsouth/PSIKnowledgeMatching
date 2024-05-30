from typing import List, Dict, Union, Tuple
from torch import Tensor
from sentence_transformers import SentenceTransformer
from beir.retrieval import models
import os
import openai
import torch

class NewSentenceBERT(models.SentenceBERT):
    """This BERT class allows us to explicitly define which model we want to use for the SentenceTransformer and direct it to the right GPU. We can also add support for ada / openai embeddings here."""
    def __init__(self, model_path, device):
        self.q_model = SentenceTransformer(model_path, device=device)
        self.doc_model = self.q_model
        self.sep = " "


class EmbedWithAda():
    """This class allows us to embed documents and queries using Ada embeddings."""
    def __init__(self):
        self.openai_key = os.environ["OPENAI_API_KEY"]

    def embed(self, text: Union[str, List[str]], model: str = "text-ada-001", batch_size: int = 1, convert_to_tensor: bool = False) -> Union[Tensor, List[Tensor]]:
        """Embeds a list of strings using the OpenAI API."""
        embeddings = []
        for i in range(0, len(text), batch_size):
            batch = text[i:i+batch_size]
            response = openai.Embedding.create(
                input=batch,
                model=model
            )
            batch_embeddings = [torch.tensor(embed['embedding']) for embed in response['data']]
            if convert_to_tensor:
                batch_embeddings = torch.stack(batch_embeddings).to(self.device)
            embeddings.extend(batch_embeddings)
        
        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings