import copy

from tqdm import tqdm
from collections import Counter, defaultdict


def get_results(gold_data, pred_data, entity_types, verbose=False, return_f1=False):
    """
    pred_data: list of dict
    """
    TP, FP, FN = 0, 0, 0

    for pred_item in sorted(pred_data, key=lambda x: int(str(x['id']))):
        
        gold_items = [item for item in gold_data if str(item['id']) == str(pred_item['id'])]
        assert len(gold_items) == 1
        gold_item = gold_items[0]
        
        gold_entities = gold_item['ent_list']

        sen_id = pred_item['id']
        sen_str = pred_item['sentence']
        pred_entities = pred_item['ent_list_pred']
        
        pred_entities = [item for item in pred_entities if item['name'] in sen_str]

        gold_entities = [each_ent for each_ent in gold_entities if each_ent['type'] in entity_types]
        pred_entities = [each_ent for each_ent in pred_entities if each_ent['type'] in entity_types]
        
        gold_entities = ['::'.join([each_ent['name'], each_ent['type'], str(each_ent['token_idx'])]) for each_ent in gold_entities]
        pred_entities = ['::'.join([each_ent['name'], each_ent['type'], str(each_ent['token_idx'])]) for each_ent in pred_entities]
        
        TP_ent = list((Counter(gold_entities) & Counter(pred_entities)).elements())
        FP_ent = list((Counter(pred_entities) - Counter(gold_entities)).elements())
        FN_ent = list((Counter(gold_entities) - Counter(pred_entities)).elements())
        
        TP += len(TP_ent)
        FP += len(FP_ent)
        FN += len(FN_ent)

        if verbose and ((Counter(pred_entities) - Counter(gold_entities)).keys() or (Counter(gold_entities) - Counter(pred_entities)).keys()):
            print(f"Sentence {sen_id}: ")
            print(f"True Positive: {TP_ent}")
            print(f"False Positive: {FP_ent}")
            print(f"False Negative: {FN_ent}")
            print()

    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print(f"Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1*100:.1f}\n")
    
    if return_f1:
        return f1

def convert_span_to_sent(pred_data):
    pred_data_sent_dict = defaultdict(list)
    for each_example in pred_data:
        sent_id = str(each_example['id']).split('_')[0]
        pred_data_sent_dict[sent_id].append(each_example)
        
    pred_data_sent = []
    for sent_id in sorted(pred_data_sent_dict.keys()):
        
        spans = sorted(pred_data_sent_dict[sent_id], key=lambda x: x['id'])
        
        sen_str = " ".join(spans[0]['tokens'])
        ent_list = spans[0]['ent_list']
        tokens = spans[0]['tokens']
        
        ent_list_pred = [ent for each_span in spans for ent in each_span['ent_list_pred']]
        pred_data_sent.append(
            {
                'id': sent_id, 
                'tokens': tokens,
                'ent_list_pred': ent_list_pred, 
                'ent_list': ent_list,
                'sentence': sen_str,
            }
        )
    
    return pred_data_sent

def get_results_boundary(gold_data, pred_data, entity_types, verbose=False):
        
    pred_data_dict_boundary = defaultdict(list)
    for boundary_example in pred_data:
        pred_data_dict_boundary[boundary_example['id']].append(boundary_example)
        
    pred_data_processed = []
    for span_id in sorted(pred_data_dict_boundary.keys()):
        example_updated = copy.deepcopy(pred_data_dict_boundary[span_id][0])
        
        boundary_example_ents, boundary_example_ents_prior = [], []
        for boundary_example in pred_data_dict_boundary[span_id]:
            boundary_example_ent = boundary_example['ent_list_pred']
            boundary_example_ent_prior = boundary_example['ent_list_pred_prior']
            assert len(boundary_example_ent) <= 1
            assert boundary_example_ent_prior is None or len(boundary_example_ent_prior) <= 1
            
            boundary_example_ents += boundary_example_ent
        
        example_updated['ent_list_pred'] = boundary_example_ents
        pred_data_processed.append(example_updated)
        
    pred_data_sent = convert_span_to_sent(pred_data_processed)
    gold_data_sampled = [item for item in gold_data if str(item['id']) in [each_example['id'] for each_example in pred_data_sent]]

    get_results(gold_data_sampled, pred_data_sent, entity_types, verbose=verbose)

