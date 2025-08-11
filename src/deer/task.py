import os
import re
import json
import copy
import random
import string
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import defaultdict, Counter
from deer.base import Task, TaskItem, LMOutput
from deer.language_models.openai_emb_model import OpenAIEmbeddingModel, OpenAIUncontextualizedTokenEmbeddingModel
from deer.language_models.sbert_emb_model import SBERTEmbeddingModel
from deer.language_models.huggingface_emb_model import HFContextualizedTokenEmbeddingModel, HFUncontextualizedTokenEmbeddingModel
from deer.retrieval_strategies.sentence_embedding_retriever import KATERetriever
from deer.retrieval_strategies.token_embedding_retriever import DEERRetriever

PROMPT_TEMPLATE = None
STOPWORDS = stopwords.words('english')
ARTICLES = ['a', 'an', 'the']

class NERItem(TaskItem):
    """
    Represents a Named Entity Recognition (NER) data item.
    """

    def __init__(self, item: dict, context_len: int):
        """
        Initialize an NERItem.

        Args:
            item (dict): Dictionary containing attributes of the NER item.
            context_len (int): Number of context tokens of each entity to consider in demonstration retrieval.
        """
        super().__init__(item)
        self.tokens = item.get('tokens')
        self.ent_list = item.get('ent_list')
        self.sentence = item.get('sentence', ' '.join(self.tokens) if self.tokens else None)
        self.sentence_embedding = item.get('sentence_embedding', None)
        self.subtoken_embeddings = item.get('subtoken_embeddings', None)
        self.offset_mapping = item.get('offset_mapping', None)
        self.token_embeddings = item.get('token_embeddings', None)

        self.icl_example_ids = item.get('icl_example_ids', None)
        self.icl_prompt = item.get('icl_prompt', None)
        self.llm_response = item.get('llm_response', None)
        self.ent_list_pred = item.get('ent_list_pred', None)
        self.ent_list_pred_prior = item.get('ent_list_pred_prior', None)
        self.reflect_tokens = item.get('reflect_tokens', None)
        self.reflect_prompt = item.get('reflect_prompt', None)
        self.boundary_tokens = item.get('boundary_tokens', None)
        
        # Build a dictionary for each token under each role
        self.entity_span_dict, self.context_span_dict, self.other_span_dict = self.build_span_dict(context_len)

    def build_span_dict(self, context_len):
        """
        Build a dictionary for each token under each role.
        
        Args:
            context_len (int): Number of context tokens of each entity to consider in demonstration retrieval.

        Returns:
            tuple: Tuple of dictionaries for each token under each role.
        """
        entity_span_dict, context_span_dict, other_span_dict = defaultdict(list), defaultdict(list), defaultdict(list)
        entity_token_idx_list, context_token_idx_list = [], []
        
        for ent in self.ent_list:
            ent_idx, ent_name, ent_type = ent['token_idx'], ent['name'], ent['type']
            ent_span = self.tokens[ent_idx[0]:ent_idx[1]+1]
            if ent_name != " ".join(ent_span):
                continue
            else:
                assert ent_name == " ".join(ent_span)
            ent_cxt_span = self.tokens[max(0, ent_idx[0]-context_len):min(len(self.tokens), ent_idx[1]+1+context_len)]
            
            for ent_token_idx in range(ent_idx[0], ent_idx[1]+1):
                ent_token = self.tokens[ent_token_idx]
                entity_span_dict[ent_token].append({'id': self.id, 'token_idx': ent_token_idx, 'token': ent_token, 'ent_type': ent_type, 'ent_span': ent_span, 'ent_cxt_span': ent_cxt_span})
            entity_token_idx_list += list(range(ent_idx[0], ent_idx[1]+1))
        assert entity_token_idx_list == sorted(entity_token_idx_list) # check if the entity_token_idx_list is sorted
        
        for ent in self.ent_list:
            ent_idx, ent_name, ent_type = ent['token_idx'], ent['name'], ent['type']
            ent_span = self.tokens[ent_idx[0]:ent_idx[1]+1]
            if ent_name != " ".join(ent_span):
                continue
            else:
                assert ent_name == " ".join(ent_span)
            ent_cxt_span = self.tokens[max(0, ent_idx[0]-context_len):min(len(self.tokens), ent_idx[1]+1+context_len)]
            
            for cxt_token_idx in list(range(max(0, ent_idx[0]-context_len), ent_idx[0])) + list(range(ent_idx[1]+1, min(len(self.tokens), ent_idx[1]+1+context_len))):
                if cxt_token_idx not in entity_token_idx_list:
                    cxt_token = self.tokens[cxt_token_idx]
                    context_span_dict[cxt_token].append({'id': self.id, 'token_idx': cxt_token_idx, 'token': cxt_token, 'ent_type': ent_type, 'ent_span': ent_span, 'ent_cxt_span': ent_cxt_span})
                    context_token_idx_list.append(cxt_token_idx)
                        
        other_token_idx_list = [idx for idx in range(len(self.tokens)) if idx not in entity_token_idx_list+context_token_idx_list]
        for other_token_idx in other_token_idx_list:
            other_token = self.tokens[other_token_idx]
            other_span = self.tokens[max(0, other_token_idx-context_len):min(len(self.tokens), other_token_idx+1+context_len)]
            other_span_dict[other_token].append({'id': self.id, 'token_idx': other_token_idx, 'token': other_token, 'ent_type': None, 'ent_span': None, 'ent_cxt_span': other_span, 'ent_span_processed': None})
                
        return entity_span_dict, context_span_dict, other_span_dict

    def to_dict(self):
        """
        Convert the NERItem to a dictionary.

        Returns:
            dict: Dictionary representation of the NERItem.
        """
        base_dict = super().to_dict()
        base_dict.update({
            "tokens": self.tokens,
            "ent_list": self.ent_list,
            "sentence": self.sentence,
            "icl_example_ids": self.icl_example_ids,
            "icl_prompt": self.icl_prompt,
            "llm_response": self.llm_response,
            "ent_list_pred": self.ent_list_pred,
            "ent_list_pred_prior": self.ent_list_pred_prior,
            "reflect_tokens": self.reflect_tokens,
            "reflect_prompt": self.reflect_prompt,
            "boundary_tokens": self.boundary_tokens,
        })
        return base_dict

    def build_icl_examples_str(self, prompt_template_name: str, icl_examples: list) -> str:
        """
        Create a string representation of ICL examples.

        Args:
            prompt_template_name (str): Name of the ICL prompt template.
            icl_examples (list): List of ICL examples.

        Returns:
            str: String representation of the ICL examples.
        """
        examples = []
        for item in icl_examples:
            if prompt_template_name == 'icl_json_format':
                ent_output = {
                    'named entities': [
                        {'name': ent['name'], 'type': ent['type']} for ent in item.ent_list
                    ]
                }
                examples.append(f"Input: {item.sentence}\nOutput: {json.dumps(ent_output)}")
            elif prompt_template_name == 'icl_tagging_format':
                assert item.sentence == " ".join(item.tokens)
                assert sorted([ent['token_idx'][0] for ent in item.ent_list]) == [ent['token_idx'][0] for ent in item.ent_list], f"Token indices are not sorted: {item.ent_list}"
                token_index_start = 0
                tokens_processed = []
                for ent in item.ent_list:
                    ent_index_start, ent_index_end = ent['token_idx']
                    # Check if there are tokens before the entity
                    if ent_index_start > token_index_start:
                        tokens_processed.append(" ".join(item.tokens[token_index_start:ent_index_start]))

                    # Check if the entity name matches the tokens
                    ent_name, ent_type = ent['name'], ent['type']
                    if ent_name != " ".join(item.tokens[ent_index_start:ent_index_end+1]):
                        tokens_processed.append(" ".join(item.tokens[ent_index_start:ent_index_end+1]))
                    else:
                        tokens_processed.append(f"[ {ent_name} | {ent_type} ]")
                    token_index_start = ent_index_end + 1
                if token_index_start < len(item.tokens):
                    tokens_processed.append(" ".join(item.tokens[token_index_start:]))
                examples.append(f"Input: {item.sentence}\nOutput: {' '.join(tokens_processed)}")
            elif prompt_template_name == 'icl_codeie_format':
                example_template = """def named_entity_recognition(input_text):\n\t# extract named entities from the input_text\n\tinput_text = "{{input_sentence}}"\n\tentity_list = []\n\t{{ent_output}}""".expandtabs(4)
                assert item.sentence == " ".join(item.tokens)
                assert sorted([ent['token_idx'][0] for ent in item.ent_list]) == [ent['token_idx'][0] for ent in item.ent_list], f"Token indices are not sorted: {item.ent_list}"
                ent_output = "\n\t".join([f'entity_list.append({{"text": "{ent["name"]}", "type": "{ent["type"]}"}})' for ent in item.ent_list]).expandtabs(4)
                examples.append(example_template.replace("{{input_sentence}}", item.sentence).replace("{{ent_output}}", ent_output).strip())
            else:
                raise ValueError(f"Invalid ICL prompt template: {prompt_template_name}")
        return "\n\n".join(examples)

    def build_icl_prompt(self, prompt_template_name: str, prompt_template: str, icl_examples: list):
        """
        Build an ICL prompt using a template and examples.

        Args:
            prompt_template_name (str): Name of the ICL prompt template.
            prompt_template (str): Template for the prompt.
            icl_examples (list): List of ICL examples.
        """
        self.icl_example_ids = [item.id for item in icl_examples]
        icl_examples_str = self.build_icl_examples_str(prompt_template_name, icl_examples)

        assert "{{input_sentence}}" in prompt_template, "Template missing {{input_sentence}} placeholder."
        assert "{{icl_examples}}" in prompt_template, "Template missing {{icl_examples}} placeholder."

        self.icl_prompt = prompt_template.replace("{{input_sentence}}", self.sentence).replace("{{icl_examples}}", icl_examples_str)

    def build_reflection_prompt(self, prompt_template_name: str, prompt_template: str, icl_examples_str: str):
        """
        Build a reflection prompt using a template and examples.

        Args:
            prompt_template (str): Template for the prompt.
            icl_examples_str (str): String representation of ICL examples.
        """
        if prompt_template_name in ['reflect_unseen', 'reflect_fn']:
            assert "{{input_text}}" in prompt_template, "Template missing {{input_text}} placeholder."
            assert "{{icl_examples}}" in prompt_template, "Template missing {{icl_examples}} placeholder."

            self.reflect_prompt = prompt_template.replace("{{input_text}}", self.sentence).replace("{{icl_examples}}", icl_examples_str)
        elif prompt_template_name == 'reflect_boundary':
            assert "{{input_text}}" in prompt_template, "Template missing {{input_text}} placeholder."
            assert "{{predicted_entity}}" in prompt_template, "Template missing {{predicted_entity}} placeholder."
            assert "{{icl_examples}}" in prompt_template, "Template missing {{icl_examples}} placeholder."
            assert len(self.ent_list_pred) == 1
            predicted_entity = copy.deepcopy(self.ent_list_pred[0])
            predicted_entity.pop('token_idx')

            self.reflect_prompt = prompt_template.replace("{{input_text}}", self.sentence).replace("{{predicted_entity}}", json.dumps(predicted_entity)).replace("{{icl_examples}}", icl_examples_str)
        else:
            raise ValueError(f"Invalid reflection prompt template: {prompt_template_name}")

    def process_miss_duplicated_entities(self, ent_list):
        """
        Process and resolve missing or duplicated entities.

        Args:
            ent_list (list): List of entities.

        Returns:
            list: Processed list of entities.
        """
        ent_names_dict = defaultdict(list)
        for ent in ent_list:
            ent_names_dict[ent['name']].append(ent)

        ent_list_processed, ent_token_index_list = [], []
        ent_list_processed = ent_list.copy()
        for ent in ent_list_processed:
            ent_token_index_list += list(range(ent['token_idx'][0], ent['token_idx'][1]+1))

        for name in sorted(ent_names_dict.keys(), key=lambda x: len(x.split()), reverse=True):
            ent_per_name = ent_names_dict[name]

            if len(set([ent['type'] for ent in ent_per_name])) != 1:
                print(f"Entities: {ent_per_name}")
                print(f"Different entity types for the same entity name: {set([ent['type'] for ent in ent_per_name])}")
            ent_type_most_common = Counter([ent['type'] for ent in ent_per_name]).most_common(1)[0][0]

            name_len = len(name.split())

            for token_idx in range(len(self.tokens) - name_len + 1):
                if name == ' '.join(self.tokens[token_idx:token_idx+name_len]) and all([token_idx not in ent_token_index_list for token_idx in range(token_idx, token_idx+name_len)]):
                    ent_list_processed.append({'name': name, 'type': ent_type_most_common, 'token_idx': [token_idx, token_idx+name_len-1]})
                    ent_token_index_list += list(range(token_idx, token_idx+name_len))

        ent_list_processed = sorted(ent_list_processed, key=lambda x: x['token_idx'][0])

        return ent_list_processed

    def get_entity_index(self, ent_list_pred):
        """
        Get token indices for predicted entities.

        Args:
            ent_list_pred (list): List of predicted entities.

        Returns:
            list: List of entities with token indices.
        """
        ent_list_pred_indexed = []
        token_idx_start = 0
        
        # Make sure the predicted entities are in the sentence
        ent_list_pred = [ent for ent in ent_list_pred if ent['name'] in self.sentence]

        for ent_pred in ent_list_pred:
            ent_pred_index = ent_pred.copy()
            ent_name = ent_pred['name']
            ent_len = len(ent_name.split())
            for token_idx in range(token_idx_start, len(self.tokens)):
                if ent_name == ' '.join(self.tokens[token_idx:token_idx+ent_len]):
                    ent_pred_index['token_idx'] = [token_idx, token_idx+ent_len-1]
                    token_idx_start = token_idx+ent_len
                    break
                
            if 'token_idx' not in ent_pred_index and all([(ent_indexed['name'] not in ent_name and ent_name not in ent_indexed['name']) for ent_indexed in ent_list_pred_indexed]) and ent_name in ' '.join(self.tokens[0:token_idx_start]):
                for token_idx in range(token_idx_start):
                    if ent_name == ' '.join(self.tokens[token_idx:token_idx+ent_len]):
                        ent_pred_index['token_idx'] = [token_idx, token_idx+ent_len-1]
                        break

            if 'token_idx' not in ent_pred_index:
                print(f"Entity '{ent_name}' not found in the sentence '{self.sentence}', full prediction '{ent_list_pred}'.")
            else:
                ent_list_pred_indexed.append(ent_pred_index)

        return ent_list_pred_indexed

    def update_results(self, prompt_template_name: str, llm_response: LMOutput):
        """
        Update the NER item with results from the LLM.

        Args:
            prompt_template_name: Name of the prompt template.
            llm_response: LMOutput object having two attributes: text and logprobs
        """
        if prompt_template_name == 'icl_json_format':
            try:
                json_pattern = re.compile(r'(\{.*\})', re.DOTALL)
                response_searched = json_pattern.search(llm_response.text).group(1)
                ent_list_pred = json.loads(response_searched)["named entities"]
            except Exception as e:
                print(f"Error {e} when processing response: {llm_response.text}")
                ent_list_pred = []
        elif prompt_template_name == 'icl_tagging_format':
            try:
                ent_list_pred = []
                token_pattern = re.compile(r'\[ (.*?) \| (.*?) \]')
                for token_match in token_pattern.finditer(llm_response.text):
                    ent_name, ent_type = token_match.group(1), token_match.group(2)
                    ent_list_pred.append({'name': ent_name, 'type': ent_type})
            except Exception as e:
                print(f"Error {e} when processing response: {llm_response.text}")
                ent_list_pred = []
        elif prompt_template_name == 'icl_codeie_format':
            try:
                ent_list_pred = []
                ent_list_pred_pattern = re.compile(r'entity_list\.append\(\{"text": "(.*?)", "type": "(.*?)"\}\)', re.DOTALL)
                for ent_match in ent_list_pred_pattern.finditer(llm_response.text):
                    ent_name, ent_type = ent_match.group(1), ent_match.group(2)
                    ent_list_pred.append({'name': ent_name, 'type': ent_type})
            except Exception as e:
                print(f"Error {e} when processing response: {llm_response.text}")
                ent_list_pred = []
        else:
            raise ValueError(f"Invalid ICL prompt template: {prompt_template_name}")
            
        ent_list_pred = self.get_entity_index(ent_list_pred)
        ent_list_pred = self.process_miss_duplicated_entities(ent_list_pred)
        self.ent_list_pred = ent_list_pred
        self.llm_response = llm_response.text

    def update_reflection_results(self, prompt_template_name: str, llm_response: LMOutput):
        """
        Update the NER item with reflection results from the LLM.

        Args:
            prompt_template_name (str): Name of the reflection prompt template.
            llm_response: LMOutput object having two attributes: text and logprobs
        """
        self.llm_response = llm_response.text
        
        if prompt_template_name in ['reflect_unseen', 'reflect_fn']:
            try: 
                json_pattern = re.compile(r'(\{.*\})', re.DOTALL)
                response_searched = json_pattern.search(llm_response.text).group(1)
                if "Final predicted entities" in response_searched:
                    response_searched = response_searched.split("Final predicted entities")[1]
                    response_searched = json_pattern.search(response_searched).group(1)
                ent_list_pred = json.loads(response_searched)["named entities"]
            except Exception as e:
                ent_list_pred = []
        elif prompt_template_name == 'reflect_boundary':
            try: 
                json_pattern = re.compile(r'(\{.*\})', re.DOTALL)
                response_searched = json_pattern.search(llm_response.text).group(1)
                if response_searched.count('{') > 1 or response_searched.count('}') > 1:
                    if "Updated Predicted Entity" in llm_response.text:
                        response_searched = llm_response.text.split("Updated Predicted Entity")[1]
                        response_searched = json_pattern.search(response_searched).group(1)    
                ent_list_pred = [json.loads(response_searched)]
            except Exception as e:
                ent_list_pred = []
        else:
            raise ValueError(f"Invalid reflection prompt template: {prompt_template_name}")
        
        ent_list_pred_processed = []
        for ent in ent_list_pred:
            if 'name' not in ent or 'type' not in ent or not ent['name'].strip():
                continue
            elif ent['name'] in self.sentence:
                ent_list_pred_processed.append({'name': ent['name'].strip(), 'type': ent['type']})
            elif ent['name'].lower().replace(' ', '') in self.sentence.lower().replace(' ', ''):
                    
                def index_spans(A, B):
                    A_lower = A.lower()
                    B_lower = B.lower()
                    
                    i, j = 0, 0
                    start = -1
                    while i < len(A_lower) and j < len(B_lower):
                        # print(f"A: {A_lower[i]}, B: {B_lower[j]}")
                        if A_lower[i] == B_lower[j] and B_lower[j:].replace(' ', '')[:len(A_lower.replace(' ', ''))] == A_lower.replace(' ', ''):
                            if start == -1:
                                start = j
                            i += 1
                            j += 1
                        elif start != -1 and A_lower[i] == B_lower[j]:
                            i += 1
                            j += 1
                        else:
                            j += 1
                    if i == len(A_lower):
                        # Extract and return the substring from B using original indices
                        return B[start:j]
                    else:
                        raise ValueError(f"Entity '{A}' not found in the sentence '{B}'.")
                    
                try:
                    ent_name_mapped = index_spans(ent['name'], self.sentence)
                    ent_list_pred_processed.append({'name': ent_name_mapped.strip(), 'type': ent['type']})
                except Exception as e:
                    print(f"Error: {e}")
            else:
                
                # Remove letters that do not exist in the sentence
                ent_name_processed = ''.join([char for char in ent['name'] if char in self.sentence])
                
                if ent_name_processed.strip() and ent_name_processed in self.sentence and ent_name_processed not in string.punctuation:
                    ent_list_pred_processed.append({'name': ent_name_processed.strip(), 'type': ent['type']})
                    continue
                    
                print(f"Entity '{ent['name']}' not found in the sentence '{self.sentence}'.")
                
        output_processed = ent_list_pred_processed
        if prompt_template_name == 'reflect_fn' or prompt_template_name == 'reflect_unseen':
            self.ent_list_pred_prior = self.ent_list_pred.copy()
            
            if output_processed and (self.reflect_tokens['fn_tokens'] or self.reflect_tokens['unseen_tokens']):
                
                fn_tokens = self.reflect_tokens['fn_tokens']
                unseen_tokens = self.reflect_tokens['unseen_tokens']
                
                ent_processed, ent_processed_index_list = [], []
                ent_processed = self.ent_list_pred.copy()
                for ent in ent_processed:
                    ent_processed_index_list += list(range(ent['token_idx'][0], ent['token_idx'][1]+1))
                
                # Make sure added or removed entities are about reflection tokens
                for ent in output_processed:
                    tokens_relevant = [token_item for token_item in fn_tokens+unseen_tokens if token_item['token'] in ent['name'].split()]
                    tokens_relevant = sorted(tokens_relevant, key=lambda x: x['token_index'])
                    if tokens_relevant:
                        for token_relevant in tokens_relevant:
                            token_relevant_index = token_relevant['token_index']
                            token_relevant_index_within_ent = ent['name'].split().index(token_relevant['token'])
                            ent_index_start = token_relevant_index - token_relevant_index_within_ent
                            ent_index_end = ent_index_start + len(ent['name'].split()) - 1
                            
                            # Make sure the entity index is correct and entity does not overlap with other entities
                            if self.tokens[ent_index_start:ent_index_end+1] == ent['name'].split() and all([token_idx not in ent_processed_index_list for token_idx in range(ent_index_start, ent_index_end+1)]):
                                ent_processed.append({'name': ent['name'].strip(), 'type': ent['type'], 'token_idx': [ent_index_start, ent_index_end]})
                                ent_processed_index_list += list(range(ent_index_start, ent_index_end+1))
                                print(f"{self.id}: Added entity '{ent['name']}' contains reflection token '{tokens_relevant}'.")
                                print()
                    else:
                        print(f"{self.id}: irrelevant entity '{ent['name']}' added!!!")  
                        print()
                        
                # Add fn_spans
                if prompt_template_name == 'reflect_fn' and self.reflect_tokens['fn_spans']:
                    for fn_span in self.reflect_tokens['fn_spans']:
                        
                        fn_span_name, fn_span_type, fn_span_index = fn_span['name'], fn_span['type'], fn_span['token_idx']
                        
                        if any(token_index in ent_processed_index_list for token_index in range(fn_span_index[0], fn_span_index[1]+1)):
                            continue
                        
                        ent_processed.append({'name': fn_span_name, 'type': fn_span_type, 'token_idx': fn_span_index})
                    
                ent_processed = self.process_miss_duplicated_entities(ent_processed)
                self.ent_list_pred = ent_processed            
        elif prompt_template_name == 'reflect_boundary':
            assert len(self.ent_list_pred) == 1
            ent_name, ent_token_index = self.ent_list_pred[0]['name'], self.ent_list_pred[0]['token_idx']
            ent_list_pred_other = [item for item in self.ent_list_pred_prior if item not in self.ent_list_pred]
            
            # Collect the token indices of other entities to make sure the new entity does not overlap with other existing entities
            ent_token_index_list_other = []
            for ent in ent_list_pred_other:
                ent_token_index_list_other += list(range(ent['token_idx'][0], ent['token_idx'][1]+1))
            
            # Save the unprocessed entity
            self.ent_list_pred_prior = self.ent_list_pred.copy()
            
            boundary_tokens = [item['token'] for item in self.boundary_tokens]
            if not output_processed:
                # Make sure the deleted entity is about boundary tokens
                if all([token not in boundary_tokens for token in ent_name.split()]):
                    print(f"{self.id}: No boundary words {boundary_tokens} in the deleted entity {ent_name}.")
                else:
                    self.ent_list_pred = output_processed
            else:
                assert len(output_processed) == 1
                ent_name_new = output_processed[0]['name']
                ent_type_new = output_processed[0]['type']
                
                # Make sure the new entity is about boundary tokens
                if ent_name_new != ent_name:
                    # Make sure the boundary changes are about boundary tokens
                    if all([token not in boundary_tokens for token in ent_name.split()]) and all([token not in boundary_tokens for token in ent_name_new.split()]):
                        print(f"{self.id}: No boundary words {boundary_tokens} in the deleted entity {ent_name} and the new entity {ent_name_new}.")
                    else:
                        # Index the new entity
                        ent_token_index_new_start = -1
                        for token_idx in range(len(self.tokens)):
                            # The new entity should overlap with the old entity
                            if ent_name_new == ' '.join(self.tokens[token_idx:token_idx+len(ent_name_new.split())]) and set(range(token_idx, token_idx+len(ent_name_new.split()))) & set(range(ent_token_index[0], ent_token_index[1]+1)):
                                ent_token_index_new_start = token_idx
                                break
                        ent_token_index_new_end = ent_token_index_new_start + len(ent_name_new.split()) - 1
                        
                        # Make sure the new entity does not overlap with other existing entities
                        if ent_token_index_new_start != -1 and all([token_idx not in ent_token_index_list_other for token_idx in range(ent_token_index_new_start, ent_token_index_new_end+1)]):
                            ent_list_pred_new = [{'name': ent_name_new, 'type': ent_type_new, 'token_idx': [ent_token_index_new_start, ent_token_index_new_end]}]
                            self.ent_list_pred = ent_list_pred_new
        else:
            raise ValueError(f"Invalid reflection prompt template: {prompt_template_name}")

class NER(Task):
    """
    Represents the Named Entity Recognition (NER) task.
    """

    def __init__(self, args):
        """
        Initialize the NER task.

        Args:
            args: Configuration and arguments for the task.
        """
        super().__init__(args)
        self.train_data = []
        self.eval_data = []

        self.data_name = args.data_name
        self.data_dir = f"data/{self.data_name}"
        self.output_dir = args.output_dir
        self.train_num = args.train_num
        self.eval_num = args.eval_num
        self.eval_split = args.eval_split
        self.sample_seed = args.sample_seed
        
        self.emb_model_type = args.emb_model_type
        self.emb_model_name = args.emb_model_name
        self.emb_batch_size = args.emb_batch_size
        self.emb_sleep_time = args.emb_sleep_time
        
        # ICL inference arguments
        self.icl_inference = args.icl_inference
        self.model_type = args.model_type
        self.model_name = args.model_name
        self.icl_demo_num = args.icl_demo_num
        self.icl_span_demo_num = args.icl_span_demo_num
        self.icl_demo_retrieval_method = args.icl_demo_retrieval_method
        self.alpha_token_match = args.alpha_token_match
        self.alpha_embed_sim = args.alpha_embed_sim
        self.entity_weight = args.entity_weight
        self.context_weight = args.context_weight
        self.other_weight = args.other_weight
        self.context_len = args.context_len
        self.entity_bound_unseen = args.entity_bound_unseen
        self.context_bound_unseen = args.context_bound_unseen
        self.entity_bound_fn = args.entity_bound_fn
        self.process_abbrev = args.process_abbrev
        self.ignore_rare = args.ignore_rare
        self.ignore_article = args.ignore_article
        self.include_unseen_boundary = args.include_unseen_boundary
        self.filter_single_token_fp = args.filter_single_token_fp
        self.retrieval_seed = args.retrieval_seed
        self.sleep_time = args.sleep_time
        self.prompt_template_name = args.prompt_template_name
        self.prior_prompt_template_name = args.prior_prompt_template_name
        self.prompt_template = PROMPT_TEMPLATE
        self.retriever = None
        self.eval_data_reflect = None
        self.eval_data_keep = None
        
        self.load_data()

    def load_data(self):
        """
        Load training and evaluation data for the NER task.
        """
        train_path = self.get_data_path(self.data_dir, "train")
        
        # Load output data for reflection
        if self.icl_inference:
            eval_path = self.get_data_path(self.data_dir, self.eval_split)
        else:
            eval_path = self.get_results_path(self.prior_prompt_template_name)
        
        if os.path.exists(train_path):
            print(f"Loading training data from data path: {train_path}...")
            with open(train_path, "r") as f:
                self.train_data = [NERItem(json.loads(line), self.context_len) for line in f]
            if self.train_num:
                self.train_data = self.sample_random_examples(self.train_data, self.train_num, self.sample_seed)
        else:
            raise FileNotFoundError(f"Train data not found: {train_path}")

        if os.path.exists(eval_path):
            print(f"Loading evaluation data from: {eval_path}...")
            with open(eval_path, "r") as f:
                self.eval_data = [NERItem(json.loads(line), self.context_len) for line in f]
            if self.eval_num:
                self.eval_data = self.sample_random_examples(self.eval_data, self.eval_num, self.sample_seed)
        else:
            raise FileNotFoundError(f"Evaluation data not found: {eval_path}")
        
        print(f"Loaded {len(self.train_data)} training and {len(self.eval_data)} evaluation examples.")

    def generate_reflection_data(self):
        """
        Generate data for reflection.
        """
        reflect_data, keep_data = [], []
        if self.prompt_template_name in ['reflect_unseen', 'reflect_fn']:
            for data_item in self.eval_data:
                reflect_tokens = self.get_reflection_tokens_spans(data_item, entity_bound_unseen=self.entity_bound_unseen, context_bound_unseen=self.context_bound_unseen, entity_bound_fn=self.entity_bound_fn, context_len=self.context_len)
                data_item_processed = copy.deepcopy(data_item)
                data_item_processed.reflect_tokens = reflect_tokens
                
                if self.prompt_template_name == 'reflect_unseen' and reflect_tokens['unseen_tokens']:
                    reflect_data.append(data_item_processed)
                elif self.prompt_template_name == 'reflect_fn' and reflect_tokens['fn_tokens']:
                    reflect_data.append(data_item_processed)
                else:
                    keep_data.append(data_item_processed)
        elif self.prompt_template_name == 'reflect_boundary':
            for data_item in self.eval_data:
                if data_item.ent_list_pred:
                    ent_list_pred = data_item.ent_list_pred.copy()    
                    for ent_pred in ent_list_pred:
                        data_item_processed = copy.deepcopy(data_item)
                        data_item_processed.ent_list_pred = [ent_pred]
                        data_item_processed.ent_list_pred_prior = ent_list_pred
                        boundary_tokens = self.get_boundary_tokens(data_item_processed)
                        data_item_processed.boundary_tokens = boundary_tokens
                        if not boundary_tokens:
                            data_item_processed.ent_list_pred_prior = [ent_pred]
                        
                        if boundary_tokens:
                            reflect_data.append(data_item_processed)
                        else:
                            keep_data.append(data_item_processed)
                else:
                    keep_data.append(data_item)
        else:
            raise ValueError(f"Invalid reflection prompt template: {self.prompt_template_name}")

        print(f"{len(reflect_data)} out of {len(self.eval_data)} examples in the valuation are selected for reflection.")
        self.eval_data_reflect, self.eval_data_keep = reflect_data, keep_data

    def get_boundary_tokens(self, data_item, entity_bound=0.5, context_len=2, rare_threshold=5):
        """
        Get boundary tokens for reflection.
        
        Args:
            data_item (NERItem): NER item for reflection.
            entity_bound (float): Threshold for recognizing entity token.
            context_len (int): Length of the context span.
            
        Returns:
            dict: Token and spans for reflection.
        """
        tokens = data_item.tokens
        ent_list_pred = data_item.ent_list_pred
        assert len(ent_list_pred) == 1
        ent_pred = ent_list_pred[0]
        ent_name = ent_list_pred[0]['name']
        
        ent_count, non_ent_count = len(self.retriever.ent_span_dict.get(ent_name, [])), len(self.retriever.non_ent_span_dict.get(ent_name, []))
        if (ent_count > 0 and non_ent_count == 0) or ent_count > 2*non_ent_count:
            return []
        
        ent_list_pred_prior = data_item.ent_list_pred_prior
        ent_list_pred_other = [item for item in ent_list_pred_prior if item != ent_pred]
        
        ent_token_index_list_other = []
        for ent in ent_list_pred_other:
            ent_token_index_list_other += list(range(ent['token_idx'][0], ent['token_idx'][1]+1))
        
        ent_word_idx_start, ent_word_idx_end = ent_pred['token_idx']
        assert ent_name == " ".join(tokens[ent_word_idx_start:ent_word_idx_end+1])
        
        boundary_token_idxes = list(range(max(0, ent_word_idx_start-1), min(len(tokens), ent_word_idx_start+1))) + list(range(ent_word_idx_end, min(len(tokens), ent_word_idx_end+1+1)))
        
        boundary_token_idxes = [token_idx for token_idx in boundary_token_idxes if token_idx not in ent_token_index_list_other]
        
        fp_tokens_idxes = [each_token['token_index'] for each_token in data_item.reflect_tokens['fp_tokens'] if each_token['token_index'] >= ent_word_idx_start and each_token['token_index'] <= ent_word_idx_end]
        
        boundary_token_idxes = boundary_token_idxes + fp_tokens_idxes
        boundary_token_idxes = sorted(list(set(boundary_token_idxes)))
        
        boundary_tokens = []
        for token_idx in boundary_token_idxes:
            token = tokens[token_idx]
            
            context_span_idxes = list(range(max(0, token_idx-context_len), min(len(tokens), token_idx+1+context_len)))
            context_span = [tokens[idx] for idx in context_span_idxes]
            
            if token in string.punctuation:
                continue
            else:
                if self.ignore_article and token.lower() in ARTICLES:
                    if token_idx < ent_word_idx_start or token_idx > ent_word_idx_end:
                        boundary_tokens.append({'token': token, 'span': context_span, 'token_index': token_idx})    
                    continue
            
            if token in self.retriever.token_count_dict:
                entity_token_count, context_token_count, other_token_count, token_count = self.retriever.token_count_dict[token]
                entity_token_rate, context_token_rate, other_token_rate = self.retriever.token_rate_dict[token]
                entity_context_rate = entity_token_count/(entity_token_count+context_token_count) if entity_token_count+context_token_count>0 else 0
                assert entity_context_rate >= entity_token_rate
                
                if token_idx < ent_word_idx_start or token_idx > ent_word_idx_end:
                    if (entity_context_rate > entity_bound or entity_token_rate > entity_bound):
                        boundary_tokens.append({'token': token, 'span': context_span, 'token_index': token_idx})
                else:
                    if entity_context_rate < entity_bound:
                        boundary_tokens.append({'token': token, 'span': context_span, 'token_index': token_idx})
            else:
                if self.include_unseen_boundary:
                    boundary_tokens.append({'token': token, 'span': context_span, 'token_index': token_idx})
            
        if self.ignore_rare and all((boundary_token_item['token'] not in self.retriever.token_count_dict or self.retriever.token_count_dict[boundary_token_item['token']][-1] < rare_threshold) for boundary_token_item in boundary_tokens):
            boundary_tokens = []
                    
        return boundary_tokens

    def get_reflection_tokens_spans(self, data_item, entity_bound_unseen, context_bound_unseen, entity_bound_fn, context_len, rare_threshold=5):
        """
        Get token and spans for reflection.

        Args:
            data_item (NERItem): NER item for reflection.
            entity_bound_unseen (float): Threshold for recognizing unseen entity token through potential entity tokens in the context.
            context_bound_unseen (float): Threshold for recognizing unseen entity token through potential context tokens.
            entity_bound_fn (float): Threshold for recognizing false negative entity token.
            rare_threshold (int): Threshold for recognizing rare tokens.

        Returns:
            dict: Token and spans for reflection.
        """
        tokens = data_item.tokens
        sentence_str = " ".join(tokens)
        output_entities = sorted(data_item.ent_list_pred, key=lambda x: x['token_idx'][0])
        
        entity_token_index_list = []
        for ent in output_entities:
            entity_token_index_list += list(range(ent['token_idx'][0], ent['token_idx'][1]+1))
        
        fp_tokens, fn_tokens, unseen_tokens = [], [], []
        fn_spans_potential_tokens = []
        for token_index, token in enumerate(tokens):
            
            if self.process_abbrev == "only" and not token[0].isupper():
                continue
            
            context_span_idxes = list(range(max(0, token_index-context_len), min(len(tokens), token_index+1+context_len)))
            context_span = [tokens[idx] for idx in context_span_idxes]
            
            if token in self.retriever.token_count_dict and self.retriever.token_count_dict[token][-1] > 0:
                entity_token_count, context_token_count, other_token_count, token_count = self.retriever.token_count_dict[token]
                entity_token_rate, context_token_rate, other_token_rate = self.retriever.token_rate_dict[token]
                
                if (entity_token_rate > entity_bound_fn and token_count > rare_threshold) and token_index not in entity_token_index_list:
                    fn_tokens.append({'token': token, 'span': context_span, 'token_index': token_index})
                    
                elif self.process_abbrev == "unseen" and token.isupper() and ((token_count > rare_threshold and entity_token_rate > entity_bound_unseen) or token_count <= rare_threshold) and token_index not in entity_token_index_list:
                    unseen_tokens.append({'token': token, 'span': context_span, 'token_index': token_index})
                    
                if entity_token_rate == 0 and token_count > rare_threshold and token_index in entity_token_index_list:
                    fp_tokens.append({'token': token, 'span': context_span, 'token_index': token_index})
                
                if entity_token_count > 1 and token_index not in entity_token_index_list and token not in string.punctuation and token not in STOPWORDS:
                    fn_spans_potential_tokens.append({'token': token, 'span': context_span, 'token_index': token_index})
            else:
                if token_index not in entity_token_index_list and not token.isdigit():
                    
                    # Detect whether there are entity or context tokens in the context span
                    ent_tokens, context_tokens = [], []
                    for cxt_token, cxt_token_idx in zip(context_span, context_span_idxes):
                        
                        if cxt_token not in self.retriever.token_rate_dict:
                            continue
                        entity_rate, context_rate, other_rate = self.retriever.token_rate_dict[cxt_token]
                        entity_count, context_count, other_count, token_count = self.retriever.token_count_dict[cxt_token]
                        
                        if cxt_token in STOPWORDS or cxt_token[0] in string.punctuation:
                            continue
                        
                        if entity_rate >= entity_bound_unseen:
                            ent_tokens.append(cxt_token)
                        elif context_rate >= context_bound_unseen:
                            context_tokens.append(cxt_token)
                    
                    if ent_tokens+context_tokens:
                        unseen_tokens.append({'token': token, 'span': context_span, 'token_index': token_index})

        fn_spans, fn_spans_index_list = [], []
        for token_item in fn_spans_potential_tokens:
            
            token, token_index = token_item['token'], token_item['token_index']

            # Include all potential entities with the starting token
            ent_spans = list(set([ent['name'] for sen_id, _ in self.retriever.entity_token_source_dict[token] for ent in self.retriever.data_dict[sen_id].ent_list if ent['name'].split()[0]==token and ent['name'] in sentence_str and len(self.retriever.ent_span_dict[ent['name']]) > 1 and len(self.retriever.ent_span_dict[ent['name']]) > 2*len(self.retriever.non_ent_span_dict[ent['name']])]))
            
            # Make sure it is not part of the existing entities or fn_spans
            ent_spans = [ent_span_name for ent_span_name in ent_spans if all(t_idx not in entity_token_index_list+fn_spans_index_list for t_idx in range(token_index, token_index+len(ent_span_name.split())))]
            
            if len(ent_spans) >= 1:
                # Take the longest span
                ent_span_name = sorted(ent_spans, key=lambda x: len(x.split()), reverse=True)[0]
                
                if ent_span_name != " ".join(tokens[token_index:token_index+len(ent_span_name.split())]):
                    continue
                
                if len(ent_span_name.split()) > 1:
                    
                    ent_spans_selected = self.retriever.ent_span_dict[ent_span_name]
                    ent_types = [item['ent_type'] for item in ent_spans_selected]
                    
                    # Take the most frequent entity type
                    ent_type = max(set(ent_types), key=ent_types.count)
                    
                    fn_spans.append({'name': ent_span_name, 'type': ent_type, 'token_idx': [token_index, token_index+len(ent_span_name.split())-1]})
                    fn_spans_index_list += list(range(token_index, token_index+len(ent_span_name.split())))
                    
        # Remove the fn_tokens that are part of the fn_spans
        fn_tokens = [fn_token for fn_token in fn_tokens if fn_token['token_index'] not in fn_spans_index_list]
                    
        return {'fp_tokens': fp_tokens, 'fn_tokens': fn_tokens, 'unseen_tokens': unseen_tokens, 'fn_spans': fn_spans}

    def generate_reflection_prompts(self):
        """
        Generate prompts for reflection.
        """
        for item in tqdm(self.eval_data_reflect, desc="Generating reflection prompts"):
            if self.prompt_template_name == 'reflect_unseen':
                icl_examples_str = self.build_icl_examples_str_reflect_unseen(item.reflect_tokens['unseen_tokens'], entity_bound=self.entity_bound_unseen, context_bound=self.context_bound_unseen, num_examples=self.icl_span_demo_num)
                item.build_reflection_prompt(self.prompt_template_name, self.prompt_template, icl_examples_str)
            elif self.prompt_template_name == 'reflect_fn':
                icl_examples_str = self.build_icl_examples_str_reflect_fn(item.reflect_tokens['fn_tokens'], num_examples=self.icl_span_demo_num)
                item.build_reflection_prompt(self.prompt_template_name, self.prompt_template, icl_examples_str)
            elif self.prompt_template_name == 'reflect_boundary':
                icl_examples_str = self.build_icl_examples_str_reflect_boundary(item.boundary_tokens, item, num_examples=self.icl_span_demo_num)
                item.build_reflection_prompt(self.prompt_template_name, self.prompt_template, icl_examples_str)
            else:
                raise ValueError(f"Invalid reflection prompt template: {self.prompt_template_name}")

    def get_reflection_prompt(self) -> list:
        """
        Retrieve reflection prompts for the eval_data_reflect.
        
        Returns:
            list: List of reflection prompts.
        """
        return [item.reflect_prompt for item in self.eval_data_reflect]

    def build_icl_examples_str_reflect_unseen(self, unseen_tokens, entity_bound, context_bound, num_examples):
        """
        Build ICL examples string for reflection on unseen tokens.
        
        Args:
            unseen_tokens (list): List of unseen tokens.
            entity_bound (float): Threshold for recognizing unseen entity token through potential entity tokens in the context.
            context_bound (float): Threshold for recognizing unseen entity token through potential context tokens.
            num_examples (int): Number of examples to generate.
            
        Returns:
            str: ICL examples string for reflection.
        """
        unseen_tokens_examples_str_list = [f"<candidate_tokens>\n{[token_item['token'] for token_item in unseen_tokens]}\n</candidate_tokens>"]
        for token_item in unseen_tokens:
            token, span = token_item['token'], token_item['span']
            context = [t for t in span if t != token]
            
            entity_tokens, context_tokens = [], []
            for cxt_token in context:
                
                if cxt_token not in self.retriever.token_rate_dict:
                    continue
                entity_rate, context_rate, other_rate = self.retriever.token_rate_dict[cxt_token]
                
                if cxt_token in STOPWORDS or cxt_token[0] in string.punctuation:
                    continue
                
                if entity_rate >= entity_bound:
                    entity_tokens.append(cxt_token)
                elif context_rate >= context_bound:
                    context_tokens.append(cxt_token)
            
            entity_context_examples_str_list = []
            if entity_tokens:
                entity_examples_str_list = []
                for ent_token in entity_tokens:
                    sen_selected_ent = []
                    for sen_id, _ in self.retriever.entity_token_source_dict[ent_token]:
                        sen_selected = [self.retriever.data_dict[sen_id]] if ent_token in self.retriever.data_dict[sen_id].tokens and ent_token in self.retriever.data_dict[sen_id].entity_span_dict else []
                        sen_selected_ent += [ent for each_sen in sen_selected for ent in each_sen.entity_span_dict[ent_token]]
                    selected_ent_sorted = sorted(sen_selected_ent, key=lambda x: len(set(x['ent_cxt_span']) & set(span)), reverse=True)[:num_examples]
                    
                    examples_str_list = []
                    for ent_span in selected_ent_sorted:
                        assert ent_token in ent_span['ent_span']
                        ent_output = {'name': ' '.join(ent_span['ent_span']), 'type': ent_span['ent_type']} if ent_span['ent_span'] else {}
                        examples_str_list.append(f"Input: {' '.join(ent_span['ent_cxt_span'])}\nOutput: {json.dumps(ent_output)}")
                    examples_str_list = list(set(examples_str_list))
                    examples_str = '\n\n'.join(examples_str_list)
                    entity_examples_str = f"<ent_token>\n{ent_token}\n</ent_token>\n<examples>\n{examples_str}\n</examples>"
                    entity_examples_str_list.append(entity_examples_str)
                entity_examples_str = '\n\n'.join(entity_examples_str_list)
                entity_examples_str = f"<potential_entity_tokens_around>\n{entity_tokens}\n</potential_entity_tokens_around>\n{entity_examples_str}"
                entity_context_examples_str_list.append(entity_examples_str)
            
            if context_tokens:
                context_examples_str_list = []
                for context_token in context_tokens:
                    sen_selected_context = []
                    for sen_id, _ in self.retriever.context_token_source_dict[context_token]:
                        sen_selected = [self.retriever.data_dict[sen_id]] if context_token in self.retriever.data_dict[sen_id].tokens and context_token in self.retriever.data_dict[sen_id].context_span_dict else []
                        sen_selected_context += [ent for each_sen in sen_selected for ent in each_sen.context_span_dict[context_token]]
                    selected_context_span_sorted = sorted(sen_selected_context, key=lambda x: len(set(x['ent_cxt_span']) & set(span)), reverse=True)[:num_examples]
                    
                    examples_str_list = []
                    for context_span in selected_context_span_sorted:
                        assert context_token in context_span['ent_cxt_span']
                        ent_output = {'name': ' '.join(context_span['ent_span']), 'type': context_span['ent_type']} if context_span['ent_span'] else {}
                        examples_str_list.append(f"Input: {' '.join(context_span['ent_cxt_span'])}\nOutput: {json.dumps(ent_output)}")
                    examples_str_list = list(set(examples_str_list))
                    examples_str = '\n\n'.join(examples_str_list)
                    context_examples_str = f"<context_token>\n{context_token}\n</context_token>\n<examples>\n{examples_str}\n</examples>"
                    context_examples_str_list.append(context_examples_str)
                context_examples_str = '\n\n'.join(context_examples_str_list)
                context_examples_str = f"<potential_context_tokens_around>\n{context_tokens}\n</potential_context_tokens_around>\n{context_examples_str}"
                entity_context_examples_str_list.append(context_examples_str)
            
            entity_context_examples_str = '\n\n'.join(entity_context_examples_str_list)
            unseen_tokens_examples_str_list.append(f"<candidate_token>\n{token}\n</candidate_token>\n{entity_context_examples_str}")
            
        unseen_tokens_examples_str = '\n\n'.join(unseen_tokens_examples_str_list)
        return unseen_tokens_examples_str

    def build_icl_examples_str_reflect_fn(self, fn_tokens, num_examples):
        """
        Build ICL examples string for reflection on false negative tokens.
        
        Args:
            fn_tokens (list): List of false negative tokens.
            num_examples (int): Number of examples to generate.
            
        Returns:
            str: ICL examples string for reflection.
        """
        fn_tokens_examples_str_list = [f"<candidate_tokens>\n{[token_item['token'] for token_item in fn_tokens]}\n</candidate_tokens>"]
        for token_item in fn_tokens:
            token, span = token_item['token'], token_item['span']
            
            entity_token_count, context_token_count, other_token_count, token_count = self.retriever.token_count_dict[token]
            token_stat = {"num_occurrences_as_entity": entity_token_count, "num_occurrences_as_context_tokens": context_token_count, "num_occurrences_as_other_tokens": other_token_count, "entity_vs_context_count": f"{entity_token_count} vs {context_token_count}", "entity_vs_non_entity_count": f"{entity_token_count} vs {context_token_count+other_token_count}"}
            
            # Select entity-related examples
            sen_selected_ent = []
            for sen_id, _ in self.retriever.entity_token_source_dict[token]:
                sen_selected = [self.retriever.data_dict[sen_id]] if token in self.retriever.data_dict[sen_id].tokens and token in self.retriever.data_dict[sen_id].entity_span_dict else []
                sen_selected_ent += [ent for each_sen in sen_selected for ent in each_sen.entity_span_dict[token]]
            selected_entity_span_sorted = sorted(sen_selected_ent, key=lambda x: len(set(x['ent_cxt_span']) & set(span)), reverse=True)[:num_examples]
            
            entity_examples_str_list = []
            for ent_span in selected_entity_span_sorted:
                assert token in ent_span['ent_span']
                ent_output = {'name': ' '.join(ent_span['ent_span']), 'type': ent_span['ent_type']} if ent_span['ent_span'] else {}
                entity_examples_str_list.append(f"Input: {' '.join(ent_span['ent_cxt_span'])}\nOutput: {json.dumps(ent_output)}")
            entity_examples_str_list = list(set(entity_examples_str_list))
            entity_examples_str = '\n\n'.join(entity_examples_str_list)
            
            # Select context-related examples
            sen_selected_context = []
            for sen_id, _ in self.retriever.context_token_source_dict[token]:
                sen_selected = [self.retriever.data_dict[sen_id]] if token in self.retriever.data_dict[sen_id].tokens and token in self.retriever.data_dict[sen_id].context_span_dict else []
                sen_selected_context += [ent for each_sen in sen_selected for ent in each_sen.context_span_dict[token]]
            selected_context_span_sorted = sorted(sen_selected_context, key=lambda x: len(set(x['ent_cxt_span']) & set(span)), reverse=True)[:num_examples]
            
            context_examples_str_list = []
            for context_span in selected_context_span_sorted:
                assert token in context_span['ent_cxt_span']
                ent_output = {'name': ' '.join(context_span['ent_span']), 'type': context_span['ent_type']} if context_span['ent_span'] else {}
                context_examples_str_list.append(f"Input: {' '.join(context_span['ent_cxt_span'])}\nOutput: {json.dumps(ent_output)}")
            context_examples_str_list = list(set(context_examples_str_list))
            context_examples_str = '\n\n'.join(context_examples_str_list)

            # Select other-related examples
            sen_selected_other = []
            for sen_id, _ in self.retriever.other_token_source_dict[token]:
                sen_selected = [self.retriever.data_dict[sen_id]] if token in self.retriever.data_dict[sen_id].tokens and token in self.retriever.data_dict[sen_id].other_span_dict else []
                sen_selected_other += [ent for each_sen in sen_selected for ent in each_sen.other_span_dict[token]]
            selected_other_span_sorted = sorted(sen_selected_other, key=lambda x: len(set(x['ent_cxt_span']) & set(span)), reverse=True)[:num_examples]
            
            other_examples_str_list = []
            for other_span in selected_other_span_sorted:
                assert token in other_span['ent_cxt_span']
                ent_output = {'name': ' '.join(other_span['ent_span']), 'type': other_span['ent_type']} if other_span['ent_span'] else {}
                other_examples_str_list.append(f"Input: {' '.join(other_span['ent_cxt_span'])}\nOutput: {json.dumps(ent_output)}")
            other_examples_str_list = list(set(other_examples_str_list))
            other_examples_str = '\n\n'.join(other_examples_str_list)
            
            examples_str_per_token = f"<candidate_token>\n{token}\n</candidate_token>\n<token_stat>\n{json.dumps(token_stat)}\n</token_stat>\n<examples>\nPositive Examples (part of entity):\n{entity_examples_str}\n\nHard Negative Examples (context tokens):\n{context_examples_str}\n\nNegative Examples (other tokens, not entity nor context):\n{other_examples_str}\n</examples>"
            fn_tokens_examples_str_list.append(examples_str_per_token)
            
        fn_tokens_examples_str = '\n\n'.join(fn_tokens_examples_str_list)
        return fn_tokens_examples_str

    def build_icl_examples_str_reflect_boundary(self, boundary_tokens, data_item, num_examples, rare_threshold=5):
        """
        Build ICL examples string for reflection on boundary tokens.
        
        Args:
            boundary_tokens (list): List of boundary tokens.
            data_item (NERItem): NER item for reflection.
            num_examples (int): Number of examples to generate.
            rare_threshold (int): Threshold for recognizing rare tokens.
            
        Returns:
            str: ICL examples string for reflection.
        """
        boundary_tokens_examples_str_list = [f"<boundary_tokens>\n{[token_item['token'] for token_item in boundary_tokens]}\n</boundary_tokens>"]
        for token_item in boundary_tokens:
            token, span = token_item['token'], token_item['span']
            
            entity_token_count, context_token_count, other_token_count, token_count = self.retriever.token_count_dict.get(token, [0, 0, 0, 0])
            toten_status = "part of the entity" if token in data_item.ent_list_pred[0]['name'].split() else "part of the context"
            
            if token_count >= rare_threshold:
                token_stat = {"num_occurrences_as_entity": entity_token_count, "num_occurrences_as_context_tokens": context_token_count, "num_occurrences_as_other_tokens": other_token_count, "entity_vs_context_count": f"{entity_token_count} vs {context_token_count}", "entity_vs_non_entity_count": f"{entity_token_count} vs {context_token_count+other_token_count}"}
        
                # Select entity-related examples
                sen_selected_ent = []
                for sen_id, _ in self.retriever.entity_token_source_dict[token]:
                    sen_selected = [self.retriever.data_dict[sen_id]] if token in self.retriever.data_dict[sen_id].tokens and token in self.retriever.data_dict[sen_id].entity_span_dict else []
                    sen_selected_ent += [ent for each_sen in sen_selected for ent in each_sen.entity_span_dict[token]]
                selected_entity_span_sorted = sorted(sen_selected_ent, key=lambda x: len(set(x['ent_cxt_span']) & set(span)), reverse=True)[:num_examples]
                
                entity_examples_str_list = []
                for ent_span in selected_entity_span_sorted:
                    assert token in ent_span['ent_span']
                    ent_output = {'name': ' '.join(ent_span['ent_span']), 'type': ent_span['ent_type']} if ent_span['ent_span'] else {}
                    entity_examples_str_list.append(f"Input: {' '.join(ent_span['ent_cxt_span'])}\nOutput: {json.dumps(ent_output)}")
                entity_examples_str_list = list(set(entity_examples_str_list))
                ent_examples_str = '\n\n'.join(entity_examples_str_list)
                
                # Select context-related examples
                sen_selected_context = []
                for sen_id, _ in self.retriever.context_token_source_dict[token]:
                    sen_selected = [self.retriever.data_dict[sen_id]] if token in self.retriever.data_dict[sen_id].tokens and token in self.retriever.data_dict[sen_id].context_span_dict else []
                    sen_selected_context += [ent for each_sen in sen_selected for ent in each_sen.context_span_dict[token]]
                selected_context_span_sorted = sorted(sen_selected_context, key=lambda x: len(set(x['ent_cxt_span']) & set(span)), reverse=True)[:num_examples]
                
                context_examples_str_list = []
                for context_span in selected_context_span_sorted:
                    assert token in context_span['ent_cxt_span']
                    ent_output = {'name': ' '.join(context_span['ent_span']), 'type': context_span['ent_type']} if context_span['ent_span'] else {}
                    context_examples_str_list.append(f"Input: {' '.join(context_span['ent_cxt_span'])}\nOutput: {json.dumps(ent_output)}")
                context_examples_str_list = list(set(context_examples_str_list))
                context_examples_str = '\n\n'.join(context_examples_str_list)

                # Select other-related examples
                sen_selected_other = []
                for sen_id, _ in self.retriever.other_token_source_dict[token]:
                    sen_selected = [self.retriever.data_dict[sen_id]] if token in self.retriever.data_dict[sen_id].tokens and token in self.retriever.data_dict[sen_id].other_span_dict else []
                    sen_selected_other += [ent for each_sen in sen_selected for ent in each_sen.other_span_dict[token]]
                selected_other_span_sorted = sorted(sen_selected_other, key=lambda x: len(set(x['ent_cxt_span']) & set(span)), reverse=True)[:num_examples]
                
                other_examples_str_list = []
                for other_span in selected_other_span_sorted:
                    assert token in other_span['ent_cxt_span']
                    ent_output = {'name': ' '.join(other_span['ent_span']), 'type': other_span['ent_type']} if other_span['ent_span'] else {}
                    other_examples_str_list.append(f"Input: {' '.join(other_span['ent_cxt_span'])}\nOutput: {json.dumps(ent_output)}")
                other_examples_str_list = list(set(other_examples_str_list))
                other_examples_str = '\n\n'.join(other_examples_str_list)
                
                examples_str_per_token = f"<boundary_token>\n{token}\n</boundary_token>\n<status>\n{toten_status}\n</status>\n<token_stat>\n{json.dumps(token_stat)}\n</token_stat>\n<examples>\nPositive Examples (part of entity):\n{ent_examples_str}\n\nHard Negative Examples (context tokens):\n{context_examples_str}\n\nNegative Examples (regular tokens, neither entity nor context):\n{other_examples_str}\n</examples>"
                boundary_tokens_examples_str_list.append(examples_str_per_token)
                
            else:
                boundary_tokens_examples_str_list.append(f"<boundary_token>\n{token}\n</boundary_token>\n<status>\n{toten_status}\n</status>\n<examples>\nThis is a token rarely seen or not seen in the training data, so there is no example.\n</examples>")
            
        boundary_tokens_examples_str = '\n\n'.join(boundary_tokens_examples_str_list)
        return boundary_tokens_examples_str

    def load_embeddings(self, data, emb_path):
        """
        Load or generate embeddings.
        """
        if self.icl_demo_retrieval_method == "kate" and any(item.sentence_embedding is None for item in data) and os.path.exists(emb_path):
            with open(emb_path, "r") as f:
                embeddings = {json.loads(line)["id"]: json.loads(line)["sentence_embedding"] for line in f}
                for item in data:
                    item.sentence_embedding = embeddings.get(item.id)
        elif self.icl_demo_retrieval_method == "deer" and os.path.exists(emb_path):
            with open(emb_path, "r", encoding="utf-8") as f:
                vocab_emb = json.load(f)
                vocab_emb = {token: np.array(embedding) for token, embedding in vocab_emb.items()}
                for item in data:
                    item.token_embeddings = np.array([vocab_emb[token] for token in item.tokens])
        elif self.icl_demo_retrieval_method in ["kate", "deer"]:
            self.generate_emb_data(data, batch_size=self.emb_batch_size, sleep_time=self.emb_sleep_time, emb_path=emb_path)
        else:
            raise ValueError(f"Unsupported retrieval method for loading embeddings: {self.icl_demo_retrieval_method}")

    def generate_emb_data(self, data, batch_size, sleep_time, emb_path):
        """
        Generate embeddings for the given data and save to the specified path.

        Args:
            data (list[NERItem]): The data for which embeddings will be generated.
            batch_size (int): Number of examples per batch for embedding generation.
            sleep_time (int): Time to wait between API requests (in seconds).
            emb_path (str): Path to save the generated embeddings.

        Returns:
            dict: A dictionary mapping example IDs to their generated embeddings.
        """
        embeddings = {}
        sentences = [item.sentence for item in data]

        # Generate embeddings in batches
        all_embeddings = self.emb_model.generate_embedding(sentences, batch_size, sleep_time)

        print(f"Generating embeddings and saving to {emb_path}...")
        if self.icl_demo_retrieval_method == "kate":
            assert len(all_embeddings) == len(data), "Number of embeddings should match the number of examples."
            with open(emb_path, "w") as f:
                for item, embedding in zip(data, all_embeddings):
                    # Assign embedding to the item and store in the dictionary
                    item.sentence_embedding = embedding
                    # Save to the embedding file
                    f.write(json.dumps({"id": item.id, "sentence_embedding": embedding}) + "\n")
                    embeddings[item.id] = embedding
        elif self.icl_demo_retrieval_method == "deer":
            all_embeddings, vocab_emb = all_embeddings
            assert len(all_embeddings) == len(data), "Number of embeddings should match the number of examples."
            for item, embedding in zip(data, all_embeddings):
                item.token_embeddings = embedding
                embeddings[item.id] = embedding
            
            with open(emb_path, "w", encoding="utf-8") as f:
                vocab_emb = {token: (embedding.tolist() if isinstance(embedding, np.ndarray) else embedding) for token, embedding in vocab_emb.items()}
                json.dump(vocab_emb, f, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported embedding method: {self.icl_demo_retrieval_method}")

        return embeddings

    def load_retriever(self):
        """
        Load the retriever for ICL inference.
        """
         
        # Initialize shot retrieval
        print(f"Initializing {self.icl_demo_retrieval_method} retriever...")
        if self.icl_demo_retrieval_method == "kate":
            self.retriever = KATERetriever(self.train_data, self.eval_data, self.args)
        elif self.icl_demo_retrieval_method == "deer":
            self.retriever = DEERRetriever(self.train_data, self.eval_data, self.args)
        else:
            raise ValueError(f"Unsupported retrieval method: {self.icl_demo_retrieval_method}")

    def generate_icl_prompts(self):
        """
        Generate prompts for In-Context Learning (ICL) inference.
        """
        for item in tqdm(self.eval_data, desc="Generating ICL prompts"):
            icl_examples = self.retriever.retrieve(item)
            assert len(icl_examples) == self.icl_demo_num, f"Number of retrieved examples ({len(icl_examples)}) does not match the specified number ({self.icl_demo_num})."
            item.build_icl_prompt(self.prompt_template_name, self.prompt_template, icl_examples)

    def sample_random_examples(self, data: list, sample_num: int, seed: int) -> list:
        """
        Randomly sample examples from a dataset.

        Args:
            data (list): Dataset to sample from.
            sample_num (int): Number of examples to sample.
            seed (int): Random seed for reproducibility.

        Returns:
            list: Randomly sampled examples.
        """
        random.seed(seed)
        shuffle_idx = list(range(len(data)))
        random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[:sample_num]
        return [data[idx] for idx in shuffle_idx]

    def get_icl_prompt(self) -> list:
        """
        Retrieve ICL prompts for all evaluation data.

        Returns:
            list: List of ICL prompts.
        """
        return [item.icl_prompt for item in self.eval_data]

    def update_icl_results(self, responses: list):
        """
        Update evaluation items with LLM responses.

        Args:
            responses (list): List of LLM responses.
        """
        for item, response in zip(self.eval_data, responses):
            item.update_results(self.prompt_template_name, response)

    def update_reflection_results(self, responses: list):
        """
        Update evaluation items with LLM responses for reflection.

        Args:
            responses (list): List of LLM responses for reflection.
        """
        for item, response in zip(self.eval_data_reflect, responses):
            item.update_reflection_results(self.prompt_template_name, response)
            
        # Apply single-token false positive filtering if enabled for boundary reflection
        # Process BOTH reflect_data and keep_data to catch all single-token FPs
        if self.prompt_template_name == 'reflect_boundary' and self.filter_single_token_fp:
            all_items = self.eval_data_reflect + self.eval_data_keep
            for item in all_items:
                if item.ent_list_pred:
                    # For items with multiple entities, check each one
                    ent_list_filtered = []
                    for ent in item.ent_list_pred:
                        ent_name = ent['name']
                        
                        # Check if it's a single-token entity
                        if len(ent_name.split()) == 1:
                            if ent_name in self.retriever.token_count_dict and self.retriever.token_count_dict[ent_name][-1] > 0:
                                ent_token_count = self.retriever.token_count_dict[ent_name][0]
                                ent_count = len(self.retriever.ent_span_dict.get(ent_name, []))
                                non_ent_count = len(self.retriever.non_ent_span_dict.get(ent_name, []))
                                rare_threshold = 5
                                
                                # Skip if never appears as standalone entity and is common
                                if ent_count == 0 and (ent_token_count > rare_threshold or non_ent_count > rare_threshold):
                                    continue  # Don't add to filtered list
                        
                        ent_list_filtered.append(ent)
                    
                    # Update with filtered list
                    if item.ent_list_pred != ent_list_filtered:
                        item.ent_list_pred_prior = item.ent_list_pred
                        item.ent_list_pred = ent_list_filtered

    def get_results_path(self, prompt_template_name) -> str:
        """
        Get the file path for storing ICL inference results.

        Returns:
            str: Path to the ICL inference results file.
        """
        
        split_name = f"{self.eval_split}_sample_{self.eval_num}_seed_{self.sample_seed}" if self.eval_num else self.eval_split
        split_name = f"{split_name}_train_sample_{self.train_num}_seed_{self.sample_seed}" if self.train_num else split_name
        retriever_name = self.icl_demo_retrieval_method
        if self.icl_demo_retrieval_method == "deer":
            retriever_name += (f"_{self.emb_model_name}" if "/" not in self.emb_model_name else f"_{self.emb_model_name.split('/')[-1]}") + f"_{self.context_len}_{self.entity_weight}_{self.context_weight}_{self.other_weight}"
            retriever_name += f"_match_{self.alpha_token_match}_embed_{self.alpha_embed_sim}"
        elif self.icl_demo_retrieval_method == "kate":
            retriever_name += f"_{self.emb_model_name}" if "/" not in self.emb_model_name else f"_{self.emb_model_name.split('/')[-1]}"
        else:
            raise ValueError(f"Unsupported retrieval method: {self.icl_demo_retrieval_method}")
        icl_demo_name = f"demo_{self.icl_demo_num}_retrieval_{retriever_name}"
        
        if prompt_template_name in ['icl_json_format', 'icl_tagging_format', 'icl_codeie_format']:
            prompt_template_name_updated = prompt_template_name
        elif prompt_template_name == 'reflect_unseen':
            prompt_template_name_updated = f"{prompt_template_name}_{self.icl_span_demo_num}_{self.entity_bound_unseen}_{self.context_bound_unseen}"
        elif prompt_template_name in ['reflect_fn', 'reflect_boundary']:
            prompt_template_name_updated = f"{prompt_template_name}_{self.icl_span_demo_num}_{self.entity_bound_unseen}_{self.context_bound_unseen}_{self.entity_bound_fn}"
        else:
            raise ValueError(f"Invalid prompt template name: {prompt_template_name}")
        
        return os.path.join(
            self.output_dir,
            split_name,
            icl_demo_name,
            self.model_name,
            prompt_template_name_updated,
            "results.jsonl"
        )
    
    def save_results(self):
        """
        Save ICL inference results to a file.
        """
        result_path = self.get_results_path(self.prompt_template_name)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        print(f"Saving results to {result_path}...")
        with open(result_path, "w") as f:
            if self.icl_inference:
                for item in self.eval_data:
                    f.write(json.dumps(item.to_dict()) + "\n")
            else:
                for item in self.eval_data_reflect+self.eval_data_keep:
                    f.write(json.dumps(item.to_dict()) + "\n")
        