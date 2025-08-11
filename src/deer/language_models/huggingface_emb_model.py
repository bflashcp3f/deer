import time
import asyncio
import torch
from typing import List, Any
from tqdm import tqdm

from deer.base import EmbeddingModel
from transformers import AutoTokenizer, AutoModel

import numpy as np

class HFContextualizedTokenEmbeddingModel(EmbeddingModel):
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
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
        for text in tqdm(texts, desc="Embedding texts"):
            # Tokenize the text
            inputs = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
            offset_mapping = inputs.pop("offset_mapping").numpy().squeeze(0)
            inputs = {key: val.to(self.device) for key, val in inputs.items() if key != "offset_mapping"}
            
            # Generate embeddings for the input
            with torch.no_grad():
                outputs = self.model(**inputs)
                subtoken_embeddings = outputs['last_hidden_state'].cpu().numpy().squeeze(0)
                assert len(subtoken_embeddings) == len(offset_mapping)
                embeddings.append(
                    {
                        "offset_mapping": offset_mapping,
                        "subtoken_embeddings": subtoken_embeddings,
                    }
                )
            
            # # Sleep for a short time to avoid hitting the API rate limit
            # time.sleep(sleep_time)
            
        return embeddings


class HFUncontextualizedTokenEmbeddingModel(EmbeddingModel):
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def collect_vocab(self, texts: List[str]) -> List[str]:
        """
        Collect the vocabulary of the input texts.

        Args:
            texts (List[str]): A list of input texts.

        Returns:
            List[str]: A list of unique tokens in the input texts.
        """
        vocab = set()
        for text in texts:
            tokens = text.split()
            vocab.update(tokens)
        return list(vocab)
        
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
        
        vocab = self.collect_vocab(texts)
        
        # Generate embeddings for the vocabulary
        vocab_embeddings = {}
        for token in tqdm(vocab, desc="Embedding vocabulary"):
            inputs = self.tokenizer(token, return_tensors="pt")
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embedding = torch.mean(outputs['last_hidden_state'], dim=1).cpu().numpy().squeeze(0)
                vocab_embeddings[token] = token_embedding
        
        embeddings = []
        
        # Process texts in batches
        for text in tqdm(texts, desc="Prepare token embeddings"):
            token_embeddings = np.array([vocab_embeddings[token] for token in text.split()])
            embeddings.append(token_embeddings)
            
        return embeddings, vocab_embeddings