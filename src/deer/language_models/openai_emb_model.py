import time
import asyncio
from typing import List, Any
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from deer.base import EmbeddingModel

import numpy as np

class OpenAIEmbeddingModel(EmbeddingModel):
    """
    A wrapper for OpenAI's embedding model API.
    """
    def __init__(self, model_name: str, async_mode: bool = True):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name (str): The name of the OpenAI embedding model (e.g., "text-embedding-3-small").
            async_mode (bool): Whether to use asynchronous mode for API requests. Default is True.
        """
        super().__init__(model_name)
        self.async_mode = async_mode

        # Initialize the OpenAI API client
        if async_mode:
            self.client = AsyncOpenAI()
        else:
            self.client = OpenAI()

    async def dispatch_embedding_requests(
        self, 
        list_of_text: List[str], 
        model: str, 
        **kwargs
    ) -> List[List[float]]:
        """
        Asynchronously dispatch embedding requests to OpenAI API.

        Args:
            list_of_text (List[str]): List of input texts for embeddings.
            model (str): Model name.
            **kwargs: Additional keyword arguments for the API.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.
        """
        assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

        # Replace newlines, which can negatively affect performance
        list_of_text = [text.replace("\n", " ") for text in list_of_text]

        try:
            data = (
                await self.client.embeddings.create(input=list_of_text, model=model, **kwargs)
            ).data
            return [d.embedding for d in data]
        except Exception as e:
            raise RuntimeError(f"Failed to dispatch asynchronous requests: {e}")

    def generate_embedding(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        sleep_time: int = 0, 
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts, batching requests as needed.

        Args:
            texts (List[str]): A list of input texts.
            batch_size (int): Number of texts to process in each batch. Default is 32.
            sleep_time (int): Delay (in seconds) between batch requests. Default is 0.
            **kwargs: Additional keyword arguments for the API.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.
        """
        embeddings = []

        # Validate batch size
        if batch_size > 2048:
            raise ValueError("Batch size cannot exceed 2048.")

        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing Batches (sleep time: {sleep_time}s)"):
            batch_texts = texts[i:i + batch_size]

            if self.async_mode:
                # Asynchronous mode
                try:
                    batch_embeddings = asyncio.run(
                        self.dispatch_embedding_requests(batch_texts, model=self.model_name, **kwargs)
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to generate embeddings asynchronously: {e}")
            else:
                # Synchronous mode
                try:
                    # Replace newlines
                    batch_texts = [text.replace("\n", " ") for text in batch_texts]

                    data = self.client.embeddings.create(
                        input=batch_texts, model=self.model_name, **kwargs
                    ).data
                    batch_embeddings = [d.embedding for d in data]
                except Exception as e:
                    raise RuntimeError(f"Failed to generate embeddings synchronously: {e}")

            embeddings.extend(batch_embeddings)
            time.sleep(sleep_time)

        return embeddings

    
class OpenAIUncontextualizedTokenEmbeddingModel(EmbeddingModel):
    """
    A wrapper for OpenAI's embedding model API.
    """
    def __init__(self, model_name: str, async_mode: bool = True):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name (str): The name of the OpenAI embedding model (e.g., "text-embedding-3-small").
            async_mode (bool): Whether to use asynchronous mode for API requests. Default is True.
        """
        super().__init__(model_name)
        self.async_mode = async_mode

        # Initialize the OpenAI API client
        if async_mode:
            self.client = AsyncOpenAI()
        else:
            self.client = OpenAI()

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

    async def dispatch_embedding_requests(
        self, 
        list_of_text: List[str], 
        model: str, 
        **kwargs
    ) -> List[List[float]]:
        """
        Asynchronously dispatch embedding requests to OpenAI API.

        Args:
            list_of_text (List[str]): List of input texts for embeddings.
            model (str): Model name.
            **kwargs: Additional keyword arguments for the API.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.
        """
        assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

        # Replace newlines, which can negatively affect performance
        list_of_text = [text.replace("\n", " ") for text in list_of_text]

        try:
            data = (
                await self.client.embeddings.create(input=list_of_text, model=model, **kwargs)
            ).data
            return [d.embedding for d in data]
        except Exception as e:
            raise RuntimeError(f"Failed to dispatch asynchronous requests: {e}")

    def generate_embedding(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        sleep_time: int = 0, 
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts, batching requests as needed.

        Args:
            texts (List[str]): A list of input texts.
            batch_size (int): Number of texts to process in each batch. Default is 32.
            sleep_time (int): Delay (in seconds) between batch requests. Default is 0.
            **kwargs: Additional keyword arguments for the API.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.
        """
        
        vocab = self.collect_vocab(texts)
        
        # Generate embeddings for the vocabulary
        vocab_embeddings = {}
        for i in tqdm(range(0, len(vocab), batch_size), desc=f"Embedding vocab. (sleep time: {sleep_time}s)"):
            batch_tokens = vocab[i:i + batch_size]
            
            if self.async_mode:
                # Asynchronous mode
                try:
                    batch_embeddings = asyncio.run(
                        self.dispatch_embedding_requests(batch_tokens, model=self.model_name, **kwargs)
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to generate embeddings asynchronously: {e}")
            else:
                # Synchronous mode
                try:
                    batch_embeddings = self.client.embeddings.create(
                        input=batch_tokens, model=self.model_name, **kwargs
                    ).data
                    batch_embeddings = [d.embedding for d in batch_embeddings]
                except Exception as e:
                    raise RuntimeError(f"Failed to generate embeddings synchronously: {e}")
            
            for token, embedding in zip(batch_tokens, batch_embeddings):
                vocab_embeddings[token] = embedding
            
            time.sleep(sleep_time)
        
        embeddings = []

        # Process texts in batches
        for text in tqdm(texts, desc="Prepare token embeddings"):
            token_embeddings = np.array([vocab_embeddings[token] for token in text.split()])
            embeddings.append(token_embeddings)
            
        return embeddings, vocab_embeddings