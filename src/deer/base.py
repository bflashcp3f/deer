import os

from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple, Optional

import numpy as np

class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    Defines the interface and enforces the implementation of core methods.
    """
    def __init__(self, model_name: str):
        """
        Initialize the embedding model with a model name.

        Args:
            model_name (str): The name of the embedding model.
        """
        self.model_name = model_name

    @abstractmethod
    def generate_embedding(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        sleep_time: int = 0, 
        **kwargs
    ) -> List[Any]:
        """
        Abstract method to generate embeddings for a list of texts.

        Args:
            texts (List[str]): A list of input texts.
            batch_size (int): Number of texts to process in each batch. Default is 32.
            sleep_time (int): Delay (in seconds) between batch requests. Default is 0.
            **kwargs: Additional keyword arguments for the API.

        Returns:
            List[Any]: A list of embeddings for the input texts.
        """
        pass


class LMOutput(NamedTuple):
    text: list[str]
    logprobs: Optional[list[np.ndarray]] = None


class LanguageModel(ABC):
    """
    Abstract base class for language models.
    Defines the interface and enforces implementation of core methods.
    """
    def __init__(self, model_name: str):
        """
        Initialize the language model with the given model name.

        Args:
            model_name (str): The name of the language model.
        """
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt_list: List[str],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 100,
        stop: List[str] = None,
        batch_size: int = 1,
        response_format: Any = None,
        logprobs: bool = False,
        seed: int = None,
        sleep_time: int = 0,
    ) -> List[LMOutput]:
        """
        Abstract method to generate responses for a list of prompts.

        Args:
            prompt_list (List[str]): A list of prompts to generate responses for.
            temperature (float): Sampling temperature. Higher values mean more randomness.
            top_p (float): Nucleus sampling probability. Limits to the smallest set of tokens.
            max_tokens (int): Maximum number of tokens in the output.
            stop (List[str]): A list of stop sequences to end generation.
            batch_size (int): Number of prompts to process in each API call.
            response_format (Any): Specifies the format of the model's output.
            logprobs (bool): Whether to return token log probabilities.
            seed (int): Random seed for deterministic behavior.
            sleep_time (int): Delay (in seconds) between batch requests to respect rate limits.

        Returns:
            List[LMOutput]: Generated responses for each prompt in `prompt_list`.
        """
        pass


class Retriever(ABC):
    """
    Abstract base class for retrieval strategies in in-context learning.
    """
    def __init__(self, retrieval_data, num_examples, seed=42):
        """
        Initialize the retriever.

        Args:
            retrieval_data (list): The data from which to retrieve examples.
            num_examples (int): Number of examples to retrieve.
            seed (int): Random seed for reproducibility.
        """
        self.retrieval_data = retrieval_data
        self.num_examples = num_examples
        self.seed = seed
        
    @abstractmethod
    def retrieve(self):
        """
        Abstract method to implement the retrieval logic.

        Returns:
            list: Retrieved examples from the retrieval data.
        """
        pass


class TaskItem(ABC):
    """
    Abstract base class for task items, providing a standard interface for task-specific data items.
    """
    def __init__(self, item: dict):
        """
        Initialize a task item from a dictionary.

        Args:
            item (dict): A dictionary containing the attributes of the item.
        """
        self.split = item.get('split', None)
        self.id = item.get('id', None)

    def to_dict(self):
        """
        Convert the task item back to a dictionary.

        Returns:
            dict: A dictionary representation of the task item.
        """
        return {
            "split": self.split,
            "id": self.id,
        }


class Task(ABC):
    """
    Abstract base class for tasks, providing a structure for task-specific logic.
    """
    def __init__(self, args):
        """
        Initialize a task.

        Args:
            args: Parsed command-line arguments.
        """
        self.args = args
        self.train_data = None
        self.eval_data = None
        
    def get_data_path(self, data_dir, data_split: str) -> str:
        """
        Get the file path for a given data split.

        Args:
            args: Parsed command-line arguments containing data_dir.
            data_split (str): The split name (e.g., "train", "dev", "test").

        Returns:
            str: The file path to the data split.
        """
        return os.path.join(data_dir, f'{data_split}.jsonl')

    @abstractmethod
    def load_data(self):
        """
        Load the task-specific data. Must be implemented by subclasses.
        """
        pass

    def preprocess_data(self, data):
        """
        Preprocess the data. Optionally implemented by subclasses.

        Args:
            data: Raw data to preprocess.
        Returns:
            Preprocessed data.
        """
        return data

    def postprocess_predictions(self, predictions):
        """
        Postprocess predictions. Optionally implemented by subclasses.

        Args:
            predictions: Raw model predictions.
        Returns:
            Postprocessed predictions.
        """
        return predictions