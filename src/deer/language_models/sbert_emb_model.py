import time
import asyncio
import torch
from typing import List, Any
from tqdm import tqdm

from deer.base import EmbeddingModel
from sentence_transformers import SentenceTransformer

import numpy as np

class SBERTEmbeddingModel(EmbeddingModel):
    """
    A wrapper for Hugging Face's embedding model API.
    """
    def __init__(self, model_name: str):
        """
        Initialize the sbert embedding model.

        Args:
            model_name (str): The name of the Hugging Face embedding model (e.g., "bert-base-uncased").
        """
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def generate_embedding(self, texts: List[str], batch_size, sleep_time, **kwargs) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the Hugging Face model.

        Args:
            texts (List[str]): A list of input texts.
            batch_size (int): Number of texts to process in each batch.
            sleep_time (int): Sleep time between API requests.
            **kwargs: Additional keyword arguments for the API.

        Returns:
            List[Any]: A list of embeddings for the input texts.
        """
        embeddings = []
        
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing Batches (sleep time: {sleep_time}s)"):
            
            batch_texts = texts[i:i + batch_size]
            
            # Generate embeddings for the batch
            batch_embeddings = self.model.encode(batch_texts)
            
            # Append the batch embeddings to the list
            embeddings.extend(batch_embeddings)
            
        # Return the list of embeddings
        embeddings = [emb.tolist() for emb in embeddings]
            
        return embeddings