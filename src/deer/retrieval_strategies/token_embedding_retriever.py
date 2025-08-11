import os
import json
import nltk
import torch
import numpy as np
from itertools import chain
from math import log
from multiprocessing import Pool
from collections import Counter, defaultdict
from tqdm import tqdm
from deer.base import Retriever
from deer.language_models.openai_emb_model import OpenAIEmbeddingModel, OpenAIUncontextualizedTokenEmbeddingModel
from deer.language_models.huggingface_emb_model import HFContextualizedTokenEmbeddingModel, HFUncontextualizedTokenEmbeddingModel

from nltk.corpus import stopwords
try:
    STOPWORDS = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = stopwords.words('english')

STOPWORDS = stopwords.words('english')
    
class DEERRetriever(Retriever):
    """
    A retrieval strategy that uses contextualized token embeddings to retrieve examples.
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
        
        self.alpha_token_match = args.alpha_token_match
        self.alpha_embed_sim = args.alpha_embed_sim
        
        self.context_len = args.context_len
        self.entity_weight = args.entity_weight
        self.context_weight = args.context_weight
        self.other_weight = args.other_weight
        
        entity_token_dict, context_token_dict, other_token_dict, token_dict, \
            entity_token_source_dict, context_token_source_dict, other_token_source_dict, token_source_dict = self.collect_token_distributions(self.demo_data, self.context_len)
        
        token_count_dict = dict([(token, [entity_token_dict[token], context_token_dict[token], other_token_dict[token], count]) for token, count in token_dict.items()])
        token_rate_dict = dict([(token, [entity_token_dict[token]/count, context_token_dict[token]/count, other_token_dict[token]/count]) for token, count in token_dict.items()])
            
        self.token_count_dict = token_count_dict
        self.token_rate_dict = token_rate_dict
        self.entity_token_source_dict = entity_token_source_dict
        self.context_token_source_dict = context_token_source_dict
        self.other_token_source_dict = other_token_source_dict
        self.token_source_dict = token_source_dict
        
        self.data_dict = {}
        for data_item in self.demo_data:
            self.data_dict[data_item.id]= data_item
        
        self.ent_span_dict, self.non_ent_span_dict = self.build_ent_spans(self.demo_data, self.context_len)
        
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
        
    def build_ent_spans(self, retrieval_data, context_len) -> dict:
        """
        Build entity spans from the retrieval data.
        
        Args:
            retrieval_data (list): The data from which to build entity spans.
            context_len (int): Number of context tokens of each entity to consider.
            
        Returns:
            ent_span_dict_all: A dictionary containing entity spans.
            non_ent_span_dict_all: A dictionary containing non-entity spans.
        """
        ent_span_all = []
        ent_span_dict_all, non_ent_span_dict_all = defaultdict(list), defaultdict(list)
        
        for data_item in retrieval_data:
            id = data_item.id
            
            tokens = data_item.tokens
            ent_list = data_item.ent_list
            
            for ent in ent_list:
                ent_idx, ent_name, ent_type = ent['token_idx'], ent['name'], ent['type']
                ent_span = tokens[ent_idx[0]:ent_idx[1]+1]
                if ent_name != " ".join(ent_span):
                    pass
                else:
                    assert ent_name == " ".join(ent_span)
                ent_cxt_span = tokens[max(0, ent_idx[0]-context_len):min(len(tokens), ent_idx[1]+1+context_len)]
                ent_span_all.append({'id': id, 'ent_idx': ent_idx, 'ent_span': ent_span, 'ent_cxt_span': ent_cxt_span, 'ent_type': ent_type})
                
        for ent_span_item in ent_span_all:
            ent_span_dict_all[" ".join(ent_span_item['ent_span'])].append(ent_span_item)
            
        for ent_span_name in ent_span_dict_all.keys():
            ent_span = ent_span_name.split()
            
            non_ent_sen_id_list = []
            for token_idx, token in enumerate(ent_span):
                sen_id_list = [sen_id for sen_id, count in self.context_token_source_dict[token] + self.other_token_source_dict[token]]
                
                if token_idx == 0:
                    non_ent_sen_id_list = sen_id_list
                else:
                    non_ent_sen_id_list = list(set(non_ent_sen_id_list) & set(sen_id_list))
                    
            if not non_ent_sen_id_list:
                non_ent_span_dict_all[ent_span_name] = []
            else:
                non_ent_span_name = ent_span_name
                for non_ent_sen_id in non_ent_sen_id_list:
                    non_ent_sen = self.data_dict[non_ent_sen_id]
                    ent_list_per_sen = non_ent_sen.ent_list
                    tokens_per_sen = non_ent_sen.tokens
                    
                    if non_ent_span_name not in " ".join(tokens_per_sen):
                        continue
                    
                    non_ent_span_start_idx = " ".join(tokens_per_sen)[:" ".join(tokens_per_sen).index(non_ent_span_name)].count(" ")
                    non_ent_span_end_idx = non_ent_span_start_idx + len(non_ent_span_name.split()) - 1
                    
                    if non_ent_span_name == " ".join(tokens_per_sen[non_ent_span_start_idx:non_ent_span_end_idx+1]) and non_ent_span_name not in ent_list_per_sen:
                        
                        non_ent_span_start_idx = -1
                        for token_idx, token in enumerate(tokens_per_sen):
                            if non_ent_span_name.split() == tokens_per_sen[token_idx:token_idx+len(non_ent_span_name.split())]:
                                non_ent_span_start_idx = token_idx
                                break
                            
                        non_ent_span_end_idx = non_ent_span_start_idx + len(non_ent_span_name.split()) - 1
                        assert non_ent_span_end_idx < len(tokens_per_sen)
                        assert non_ent_span_name == " ".join(tokens_per_sen[non_ent_span_start_idx:non_ent_span_end_idx+1])
                        
                        non_ent_span = tokens_per_sen[non_ent_span_start_idx:non_ent_span_end_idx+1]
                        non_ent_cxt_span = tokens_per_sen[max(0, non_ent_span_start_idx-context_len):min(len(tokens_per_sen), non_ent_span_end_idx+1+context_len)]
                        
                        non_ent_span = tokens_per_sen[non_ent_span_start_idx:non_ent_span_end_idx+1]
                        non_ent_cxt_span = tokens_per_sen[max(0, non_ent_span_start_idx-context_len):min(len(tokens_per_sen), non_ent_span_end_idx+1+context_len)]
                        
                        non_ent_span_dict_all[non_ent_span_name].append({'id': non_ent_sen.id, 'ent_idx': (non_ent_span_start_idx, non_ent_span_end_idx), 'ent_span': non_ent_span, 'ent_cxt_span': non_ent_cxt_span, 'ent_type': None})
        
        return ent_span_dict_all, non_ent_span_dict_all

    def collect_token_distributions(self, retrieval_data, context_len):
        """
        Collect token distributions from the retrieval data.
        
        Args:
            retrieval_data (list): The data from which to collect token distributions.
            context_len (int): Number of context tokens of each entity to consider.
        """
        entity_token_dict_all, context_token_dict_all, other_token_dict_all, token_dict_all = Counter(), Counter(), Counter(), Counter()
        entity_token_source_dict_all, context_token_source_dict_all, other_token_source_dict_all, token_source_dict_all = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        
        for data_item in tqdm(retrieval_data, desc="Collecting token distributions"):
            id = data_item.id
            
            tokens = data_item.tokens
            ent_list = data_item.ent_list
            
            entity_token_idx_list = []
            context_token_idx_list = []
            
            # Collect entity tokens
            for ent in ent_list:
                ent_name = ent['name']
                ent_idx = ent['token_idx']
                if ent_name != " ".join(tokens[ent_idx[0]:ent_idx[1]+1]):
                    pass
                entity_token_idx_list += list(range(ent_idx[0], ent_idx[1]+1))
            assert entity_token_idx_list == sorted(entity_token_idx_list) # check if the ent_token_idx_list is sorted
            
            # Collect context tokens
            for ent in ent_list:
                ent_idx = ent['token_idx']
                for context_token_idx in list(range(max(0, ent_idx[0]-context_len), ent_idx[0])) + list(range(ent_idx[1]+1, min(len(tokens), ent_idx[1]+1+context_len))):
                        if context_token_idx not in entity_token_idx_list:
                            context_token_idx_list.append(context_token_idx)
                            
            other_token_idx_list = [idx for idx in range(len(tokens)) if idx not in entity_token_idx_list+context_token_idx_list]
            
            entity_token_list = [tokens[idx] for idx in entity_token_idx_list]
            context_token_list = [tokens[idx] for idx in context_token_idx_list]
            other_token_list = [tokens[idx] for idx in other_token_idx_list]
            
            entity_token_dict, context_token_dict, other_token_dict = Counter(entity_token_list), Counter(context_token_list), Counter(other_token_list)
            token_dict = entity_token_dict + context_token_dict + other_token_dict
            
            for token, token_count in entity_token_dict.items():
                entity_token_source_dict_all[token].append((id, token_count))
                
            for token, token_count in context_token_dict.items():
                context_token_source_dict_all[token].append((id, token_count))
                
            for token, token_count in other_token_dict.items():
               other_token_source_dict_all[token].append((id, token_count))
                
            for token, token_count in token_dict.items():
                token_source_dict_all[token].append((id, token_count))
                
            entity_token_dict_all += entity_token_dict
            context_token_dict_all += context_token_dict
            other_token_dict_all += other_token_dict
            token_dict_all += token_dict
        
        return entity_token_dict_all, context_token_dict_all, other_token_dict_all, token_dict_all, \
            entity_token_source_dict_all, context_token_source_dict_all, other_token_source_dict_all, token_source_dict_all

    def generate_retrieval_data_embs(self):
        """
        Generate sentence embeddings for the retrieval data.
        """
        retrieval_data_embs = []
        for data_item in tqdm(self.retrieval_data, desc="Formulating sentence embeddings"):
            tokens = data_item.tokens
            sentence = data_item.sentence
            assert " ".join(tokens) == sentence
            token_embeddings = data_item.token_embeddings
            
            token_scores = [self.entity_weight*self.token_rate_dict[token][0] + self.context_weight*self.token_rate_dict[token][1] + self.other_weight*self.token_rate_dict[token][2] for token in tokens]

            if np.sum(token_scores) == 0:
                print("Warning: token_scores sum to zero, using uniform weights or skipping.")
                sentence_embedding = np.mean(token_embeddings, axis=0)  # Fallback to mean
            else:
                sentence_embedding = np.average(token_embeddings, axis=0, weights=token_scores)

            retrieval_data_embs.append(sentence_embedding)
            
        self.retrieval_data_embs = np.array(retrieval_data_embs)
        self.retrieval_data_embs_norms = np.linalg.norm(self.retrieval_data_embs, axis=1)

    def get_demo_retrieval_path(self, output_dir, eval_split, train_num, eval_num, icl_demo_num, sample_seed, emb_model_name):
        split_name = f"{eval_split}_sample_{eval_num}_seed_{sample_seed}" if eval_num else eval_split
        split_name = f"{split_name}_train_sample_{train_num}_seed_{sample_seed}" if train_num else split_name
        retriever_name = "deer" + (f"_{emb_model_name}" if "/" not in emb_model_name else f"_{emb_model_name.split('/')[-1]}") + f"_{self.context_len}_{self.entity_weight}_{self.context_weight}_{self.other_weight}"
        retriever_name += f"_match_{self.alpha_token_match}_embed_{self.alpha_embed_sim}"
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
        demo_emb_path = os.path.join(data_dir, f"train_emb_vocab_uncxt_{emb_model_name if '/' not in emb_model_name else emb_model_name.split('/')[-1]}.jsonl")
        
        if os.path.exists(demo_emb_path):
            print(f"Loading demo embeddings from {demo_emb_path}...")
            with open(demo_emb_path, 'r') as f:
                vocab_emb = json.load(f)
            vocab_emb = {token: np.array(embedding) for token, embedding in vocab_emb.items()}
            
            for item in demo_examples:
                tokens = item.tokens
                sentence = item.sentence
                assert " ".join(tokens) == sentence
                token_embeddings = np.array([vocab_emb[token] for token in tokens if token in vocab_emb])
                
                token_scores = [self.entity_weight*self.token_rate_dict[token][0] + self.context_weight*self.token_rate_dict[token][1] + self.other_weight*self.token_rate_dict[token][2] if token in self.token_rate_dict else 1.0 for token in tokens]
                
                sentence_embedding = np.average(token_embeddings, axis=0, weights=token_scores)
                item.sentence_embedding = sentence_embedding
        else:
            print(f"Generating embeddings for demo examples using {emb_model_name}...")
            emb_model = self.load_embedding_model(emb_model_type, emb_model_name)
            vocab_emb = self.generate_emb_data(demo_examples, emb_model, batch_size, sleep_time)
            
            if device == "cuda" and emb_model_type == "huggingface":
                del emb_model
                torch.cuda.empty_cache()
            
            # Save the embeddings to a file
            with open(demo_emb_path, "w", encoding="utf-8") as f:
                vocab_emb = {token: (embedding.tolist() if isinstance(embedding, np.ndarray) else embedding) for token, embedding in vocab_emb.items()}
                json.dump(vocab_emb, f, ensure_ascii=False)
                    
        # Load/generate embeddings for the query examples
        query_emb_path = os.path.join(data_dir, f"test_emb_vocab_uncxt_{emb_model_name if '/' not in emb_model_name else emb_model_name.split('/')[-1]}.jsonl")
        
        if os.path.exists(query_emb_path):
            print(f"Loading query embeddings from {query_emb_path}...")
            with open(query_emb_path, 'r') as f:
                vocab_emb = json.load(f)
            vocab_emb = {token: np.array(embedding) for token, embedding in vocab_emb.items()}
            
            for item in query_examples:
                tokens = item.tokens
                sentence = item.sentence
                assert " ".join(tokens) == sentence
                token_embeddings = np.array([vocab_emb[token] for token in tokens if token in vocab_emb])
                
                token_scores = [self.entity_weight*self.token_rate_dict[token][0] + self.context_weight*self.token_rate_dict[token][1] + self.other_weight*self.token_rate_dict[token][2] if token in self.token_rate_dict else 1.0 for token in tokens]
                
                sentence_embedding = np.average(token_embeddings, axis=0, weights=token_scores)
                item.sentence_embedding = sentence_embedding
        else:
            print(f"Generating embeddings for query examples using {emb_model_name}...")
            emb_model = self.load_embedding_model(emb_model_type, emb_model_name)
            vocab_emb = self.generate_emb_data(query_examples, emb_model, batch_size, sleep_time)
            
            if device == "cuda" and emb_model_type == "huggingface":
                del emb_model
                torch.cuda.empty_cache()
            
            # Save the embeddings to a file
            with open(query_emb_path, "w", encoding="utf-8") as f:
                vocab_emb = {token: (embedding.tolist() if isinstance(embedding, np.ndarray) else embedding) for token, embedding in vocab_emb.items()}
                json.dump(vocab_emb, f, ensure_ascii=False)
            
        demo_data_embs = np.array([item.sentence_embedding for item in demo_examples])
        demo_data_embs_norms = np.linalg.norm(demo_data_embs, axis=1)
        
        demo_selected_scores, demo_selected_idxs = [], []
        for query_item in tqdm(query_examples, desc="Retrieve demo examples"):
            tokens = query_item.tokens
            
            # Calculate embedding-level similarity scores
            query_emb = np.array(query_item.sentence_embedding)
            dot_product = np.dot(demo_data_embs, query_emb)
            cosine_sim_scores = dot_product / demo_data_embs_norms
            
            # Calculate token-match similarity scores
            unseen_tokens = [token for token in tokens if token not in self.token_count_dict or self.token_count_dict[token][3] == 0]
            seen_tokens = [token for token in tokens if token not in unseen_tokens and token not in STOPWORDS]
            
            retrieval_example_ids = list(set([item_id for token in seen_tokens for item_id, _ in self.token_source_dict[token]]))
            retrieval_examples = [self.data_dict[item_id] for item_id in retrieval_example_ids]
            
            retrieval_example_token_match_scores = {}
            for each_example in retrieval_examples:
                each_retrieval_tokens = each_example.tokens
                
                overlap_tokens = set(each_retrieval_tokens) & set(seen_tokens)
                ent_score = sum([self.token_rate_dict[token][0] for token in overlap_tokens])
                context_score = sum([self.token_rate_dict[token][1] for token in overlap_tokens])
                other_score = sum([self.token_rate_dict[token][2] for token in overlap_tokens])
                item_score = ent_score*self.entity_weight + context_score*self.context_weight + other_score*self.other_weight
                
                retrieval_example_token_match_scores[each_example.id] = item_score
                
            aggregated_scores = []
            for example_idx in range(len(self.demo_data)):
                example_id = self.demo_data[example_idx].id
                aggregated_score = self.alpha_token_match*(retrieval_example_token_match_scores[example_id] if example_id in retrieval_example_token_match_scores else 0.0) + self.alpha_embed_sim*cosine_sim_scores[example_idx]
            
                aggregated_scores.append(aggregated_score)
            
            aggregated_scores = np.array(aggregated_scores)
            top_idx = np.argsort(aggregated_scores)[-n_shots:]
            selected_scores = aggregated_scores[top_idx]
            demo_selected_scores.append(selected_scores.tolist())
            demo_selected_idxs.append(top_idx.tolist())
            
        return demo_selected_scores, demo_selected_idxs

    def load_embedding_model(self, emb_model_type, emb_model_name):
        """
        Load the embedding model based on the specified type and name.

        Args:
            emb_model_type (str): Type of the embedding model.
            emb_model_name (str): Name of the embedding model.

        Returns:
            EmbeddingModel: The loaded embedding model.
        """
        if emb_model_type == "openai":
            emb_model = OpenAIUncontextualizedTokenEmbeddingModel(emb_model_name)
        elif emb_model_type == "huggingface":
            emb_model = HFUncontextualizedTokenEmbeddingModel(emb_model_name)
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
        embeddings_vocab = emb_model.generate_embedding(sentences, batch_size, sleep_time)
        embeddings, vocab_emb = embeddings_vocab
        
        assert len(embeddings) == len(data), "Number of embeddings should match the number of examples."
        for item, embedding in zip(data, embeddings):
            tokens = item.tokens
            sentence = item.sentence
            assert " ".join(tokens) == sentence
            assert len(tokens) == len(embedding), f"Number of tokens ({len(tokens)}) should match the number of embeddings ({len(embedding)})"
            token_embeddings = embedding
            
            token_scores = [self.entity_weight*self.token_rate_dict[token][0] + self.context_weight*self.token_rate_dict[token][1] + self.other_weight*self.token_rate_dict[token][2] if token in self.token_rate_dict else 1.0 for token in tokens]
            
            sentence_embedding = np.average(token_embeddings, axis=0, weights=token_scores)
            item.sentence_embedding = sentence_embedding
            
        return vocab_emb
    
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