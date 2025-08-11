import os
import json
import nltk
import torch
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from deer.base import Retriever
from deer.language_models.openai_emb_model import OpenAIEmbeddingModel
from deer.language_models.sbert_emb_model import SBERTEmbeddingModel

from nltk.corpus import stopwords
try:
    STOPWORDS = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = stopwords.words('english')

class KATERetriever(Retriever):
    """
    A retrieval strategy that uses sentence embeddings to retrieve examples.
    """
    def __init__(self, demo_data, query_data, args):
        """
        Initialize the retriever.

        Args:
            candidate_data (list): The data from which to retrieve examples.
            query_data (list): The query data.
            args (argparse.Namespace): The arguments containing configuration options.
        """
        self.demo_data = demo_data
        self.query_data = query_data
        self.data_name = args.data_name
        self.icl_demo_num = args.icl_demo_num
        self.eval_split = args.eval_split
        self.eval_num = args.eval_num
        self.train_num = args.train_num
        self.sample_seed = args.sample_seed
        self.emb_model_type = args.emb_model_type
        self.emb_model_name = args.emb_model_name
        self.emb_batch_size = args.emb_batch_size
        self.emb_sleep_time = args.emb_sleep_time
        self.data_dir = f"data/{self.data_name}"
        self.output_dir = args.output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        demo_retrieval_scores_idxs_path = self.get_demo_retrieval_path(self.output_dir, self.eval_split, self.train_num, self.eval_num, self.icl_demo_num, self.sample_seed, self.emb_model_name)
        
        if os.path.exists(demo_retrieval_scores_idxs_path):
            # Load precomputed scores and indices
            print(f"Loading demo retrieval scores and indices from {demo_retrieval_scores_idxs_path}...")
            with open(demo_retrieval_scores_idxs_path, 'r') as f:
                demo_retrieval_data = json.load(f)
                self.demo_selected_scores = demo_retrieval_data['scores']
                self.demo_selected_idxs = demo_retrieval_data['indices']
        else:
            self.demo_selected_scores, self.demo_selected_idxs = self.calculate_demo_relevance_scores(
                demo_data, query_data, self.data_dir, self.emb_model_type, self.emb_model_name, self.emb_batch_size, self.emb_sleep_time, self.icl_demo_num, self.device
            )
            
            if not os.path.exists(os.path.dirname(demo_retrieval_scores_idxs_path)):
                os.makedirs(os.path.dirname(demo_retrieval_scores_idxs_path))
            
            # Save the computed scores and indices
            print(f"Saving demo retrieval scores and indices to {demo_retrieval_scores_idxs_path}")
            with open(demo_retrieval_scores_idxs_path, 'w') as f:
                json.dump({
                    'scores': self.demo_selected_scores,
                    'indices': self.demo_selected_idxs
                }, f)

        self.query_id2idx = {query.id: idx for idx, query in enumerate(query_data)}
        self.query_idx2id = {idx: query.id for idx, query in enumerate(query_data)}
        
    def get_demo_retrieval_path(self, output_dir, eval_split, train_num, eval_num, icl_demo_num, sample_seed, emb_model_name):
        split_name = f"{eval_split}_sample_{eval_num}_seed_{sample_seed}" if eval_num else eval_split
        split_name = f"{split_name}_train_sample_{train_num}_seed_{sample_seed}" if train_num else split_name
        retriever_name = "kate" + (f"_{emb_model_name}" if "/" not in emb_model_name else f"_{emb_model_name.split('/')[-1]}")
        icl_demo_name = f"demo_{icl_demo_num}_retrieval_{retriever_name}"
        return f"{output_dir}/{split_name}/{icl_demo_name}/demo_scores_idxs.json"
        
    def calculate_demo_relevance_scores(self, demo_examples, query_examples, data_dir, emb_model_type, emb_model_name, batch_size, sleep_time, n_shots, device):
        """
        Calculate the relevance scores for the demo examples with respect to the query examples.

        Args:
            demo_examples (list): The demo examples.
            query_examples (list): The query examples.
            emb_model_name (str): Name of the embedding model to use.
            n_shots (int): Number of shots for the demo.
            device (str): Device to use for computation.

        Returns:
            tuple: Relevance scores and indices of selected examples.
        """
        # Load/generate embeddings for the demo examples
        demo_emb_path = os.path.join(data_dir, f"train_emb_{emb_model_name if '/' not in emb_model_name else emb_model_name.split('/')[-1]}.jsonl")
        
        if os.path.exists(demo_emb_path):
            print(f"Loading demo embeddings from {demo_emb_path}...")
            with open(demo_emb_path, 'r') as f:
                embeddings = {json.loads(line)["id"]: json.loads(line)["sentence_embedding"] for line in f}
                for item in demo_examples:
                    item.sentence_embedding = embeddings.get(item.id)
        else:
            print(f"Generating embeddings for demo examples using {emb_model_name}...")
            emb_model = self.load_embedding_model(emb_model_type, emb_model_name, device)
            self.generate_emb_data(demo_examples, emb_model, batch_size, sleep_time)
            
            if device == "cuda" and emb_model_type == "sbert":
                del emb_model
                torch.cuda.empty_cache()
            
            with open(demo_emb_path, 'w') as f:
                for item in demo_examples:
                    f.write(json.dumps({"id": item.id, "sentence_embedding": item.sentence_embedding}) + '\n')
                    
        # Load/generate embeddings for the query examples
        query_emb_path = os.path.join(data_dir, f"test_emb_{emb_model_name if '/' not in emb_model_name else emb_model_name.split('/')[-1]}.jsonl")
        
        if os.path.exists(query_emb_path):
            print(f"Loading query embeddings from {query_emb_path}...")
            with open(query_emb_path, 'r') as f:
                embeddings = {json.loads(line)["id"]: json.loads(line)["sentence_embedding"] for line in f}
                for item in query_examples:
                    item.sentence_embedding = embeddings.get(item.id)
        else:
            print(f"Generating embeddings for query examples using {emb_model_name}...")
            emb_model = self.load_embedding_model(emb_model_type, emb_model_name, device)
            self.generate_emb_data(query_examples, emb_model, batch_size, sleep_time)
            
            if device == "cuda" and emb_model_type == "sbert":
                del emb_model
                torch.cuda.empty_cache()
            
            with open(query_emb_path, 'w') as f:
                for item in query_examples:
                    f.write(json.dumps({"id": item.id, "sentence_embedding": item.sentence_embedding}) + '\n')
            
        demo_data_embs = np.array([item.sentence_embedding for item in demo_examples])
        demo_data_embs_norms = np.linalg.norm(demo_data_embs, axis=1)
        
        demo_selected_scores, demo_selected_idxs = [], []
        for query_item in tqdm(query_examples, desc="Retrieve demo examples"):
            query_emb = np.array(query_item.sentence_embedding)
            dot_product = np.dot(demo_data_embs, query_emb)
            cosine_sim_scores = dot_product / demo_data_embs_norms
            top_idx = np.argsort(cosine_sim_scores)[-n_shots:]
            selected_scores = cosine_sim_scores[top_idx]
            demo_selected_scores.append(selected_scores.tolist())
            demo_selected_idxs.append(top_idx.tolist())
            
        return demo_selected_scores, demo_selected_idxs
            
        
    def load_embedding_model(self, emb_model_type, emb_model_name, device):
        """
        Load the embedding model based on the specified type and name.

        Args:
            emb_model_type (str): Type of the embedding model.
            emb_model_name (str): Name of the embedding model.
            device (str): Device to use for computation.

        Returns:
            EmbeddingModel: The loaded embedding model.
        """
        if emb_model_type == "openai":
            emb_model = OpenAIEmbeddingModel(emb_model_name)
        elif emb_model_type == "sbert":
            emb_model = SBERTEmbeddingModel(emb_model_name)
        else:
            raise ValueError(f"Unsupported embedding model type: {emb_model_type}")
        
        return emb_model

    def generate_emb_data(self, data, emb_model, batch_size, sleep_time):
        """
        Generate embeddings for the given data.

        Args:
            data (list): The data for which to generate embeddings.
            
        Returns:
            list: The data with generated embeddings.
        """
        sentences = [item.sentence for item in data]
        # Generate embeddings in batches
        embeddings = emb_model.generate_embedding(sentences, batch_size, sleep_time)
        
        assert len(embeddings) == len(data), "Number of embeddings should match the number of examples."
        
        for item, embedding in zip(data, embeddings):
            item.sentence_embedding = embedding
        
    def retrieve(self, query_item):
        """
        Retrieve `num_examples` examples with the closest sentence embeddings to the query.

        Args:
            query_item (TaskItem): The query item for which to retrieve examples.
            
        Returns:
            list: Retrieved examples from the retrieval data.
        """
        id = query_item.id
        top_idx = self.demo_selected_idxs[self.query_id2idx[id]]
        assert len(top_idx) == self.icl_demo_num, f"Expected {self.icl_demo_num} examples, but got {len(top_idx)}"
        
        return [self.demo_data[idx] for idx in top_idx]