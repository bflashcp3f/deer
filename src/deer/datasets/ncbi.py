from deer.task import NER

ICL_JSON_FORMAT = '''Here is the JSON template for named entity recognition:
{"named entities": [{"name": "ent_name_1", "type": "ent_type_1"}, ..., {"name": "ent_name_n", "type": "ent_type_n"}]}

Please identify "Disease" entities (exact text spans), following the JSON template above, and output the JSON object. If no named entities identified, output {"named entities": []}.

{{icl_examples}}

Input: {{input_sentence}}
Output: '''

ICL_TAGGING_FORMAT = '''Please identify "Disease" entities (tagging exact text spans in the input), following the format in provided examples. If no entities are identified, return the input text unchanged.

{{icl_examples}}

Input: {{input_sentence}}
Output: '''

ICL_CODEIE_FORMAT = '''Please identify "Disease" entities, following the Python function format in provided examples. If no entities are identified, do not return anything.

{{icl_examples}}

def named_entity_recognition(input_text):
    # extract named entities from the input_text
    input_text = "{{input_sentence}}"'''

REFLECT_UNSEEN = """<input_text>
{{input_text}}
</input_text>

{{icl_examples}}

Given the provided text from a disease-related PubMed article, evaluate each candidate token along with the surrounding tokens to determine if it should be categorized as (part of) a new Disease entity. **Use provided examples, if available, for reference.** If it should be a Disease entity, extract the exact text span in the sentence, including any spaces, in JSON format. Note that abbreviations and full disease names are separate entities. If no changes are made, return {"named entities": []}.

JSON Template:
{"named entities": [{"name": "ent_name_1", "type": "ent_type_1"}, ..., {"name": "ent_name_n", "type": "ent_type_n"}]}

Output Format:

Candidate Token: token_1
Contextual Meaning: ...
Relation to Examples Provided: ...
Rationale: ...
Updates: ... (add a new entity/no change)

Candidate Token: token_n
Contextual Meaning: ...
Relation to Examples Provided: ...
Rationale: ...
Updates: ... (add a new entity/no change)

Final predicted entities for the input text (JSON format):"""

REFLECT_FN = '''<input_text>
{{input_text}}
</input_text>

{{icl_examples}}

Please follow the instructions below:
1. Evaluate each candidate token listed above by carefully considering both the positive and negative examples provided. Pay particular attention to the overall statistical data on whether tokens are included or excluded from the entity. Hard negative examples highlight tokens that are not part of the entity but are located near it.
2. Review both sets carefully. In many cases, a token may be identified as part of an entity in positive examples but not in negative ones, likely due to inconsistencies in the annotation process. if positive and (hard) negative examples seem similar, **base your decision on statistical data**, such as the frequency of the span being recognized as an entity versus its context, particularly when the data is clear-cut (e.g., one frequency is significantly higher).
3. If the token has not been seen or is rarely seen in the training data, use your best judgment to determine whether it should be considered as part of or the entire name of a **specific** Disease entity.
4. If any modifications are necessary, provide the updated entities by **extracting the exact text span in the sentence**, including any spaces and no outside tokens added, in JSON format. Note that abbreviations and full disease names are separate entities. If no changes are required, return {"named entities": []}.

JSON Template:
{"named entities": [{"name": "ent_name_1", "type": "ent_type_1"}, ..., {"name": "ent_name_n", "type": "ent_type_n"}]}

Output Format:

Candidate Token: token_1
Training Data Stats: ... entity ... context ... regular ...
Contextual Meaning: ...
Relation to Examples Provided: ... positive examples ... negative examples ...
Rationale: ...
Updates: ... (add a new entity/no change)

Candidate Token: token_n
Training Data Stats: ... entity ... context ... regular ...
Contextual Meaning: ...
Relation to Examples Provided: ... positive examples ... negative examples ...
Rationale: ...
Updates: ... (add a new entity/no change)

Final predicted entities for the input text (JSON format):'''

REFLECT_BOUNDARY = '''<input_text>
{{input_text}}
</input_text>

<predicted_entity>
{{predicted_entity}}
</predicted_entity>

{{icl_examples}}

Please follow the instructions below:
1. Calibrate the boundary of the predicted entity by evaluating each boundary token listed above in relation to the predicted entity. Consider both the provided positive and negative examples, with particular attention to the context surrounding the boundary token and overall statistical data on inclusion or exclusion from the entity. Hard negative examples highlight tokens that are not part of the entity but are located near it.
2. Review both sets carefully. In many cases, a token may be identified as part of an entity in positive examples but not in negative ones, likely due to inconsistencies in the annotation process. if positive and (hard) negative examples seem similar, **base your decision on statistical data**, such as the frequency of the span being recognized as an entity versus its context, particularly when the data is clear-cut (e.g., one frequency is significantly higher).
3. If the token has not been seen or is rarely seen in the training data, use your best judgment to determine whether it should be considered as part of or the entire name of a **specific** Disease entity.
4. Determine whether any modifications, such as adding or removing boundary tokens, are necessary. If changes are required, provide the updated entity by **extracting the exact text span in the sentence**, including any spaces and no outside tokens added, in JSON format. Note that abbreviations and full disease names are separate entities. If no tokens are added to or removed from the predicted entity, output the original entity. If all original tokens are removed, output {}.

JSON Template:
{"name": "ent_name", "type": "ent_type"}

Output Format:

Boundary Token: token_1
Training Data Stats: ... entity ... context ... regular ...
Contextual Meaning: ...
Rationale: ... positive examples ... negative examples ... data stats ...

Boundary Token: token_n
Training Data Stats: ... entity ... context ... regular ...
Contextual Meaning: ...
Rationale: ... positive examples ... negative examples ... data stats ...

Updated Predicted Entity (JSON format):'''

class NCBI(NER):
    """
    A task class for handling the NCBI Disease Named Entity Recognition (NER) dataset.
    """
    def __init__(self, args):
        """
        Initialize the NCBI task with dataset and configuration.

        Args:
            args: Parsed command-line arguments containing task configuration.
        """
        super().__init__(args)
        if args.prompt_template_name == "icl_json_format":
            self.prompt_template = ICL_JSON_FORMAT
        elif args.prompt_template_name == "icl_tagging_format":
            self.prompt_template = ICL_TAGGING_FORMAT
        elif args.prompt_template_name == "icl_codeie_format":
            self.prompt_template = ICL_CODEIE_FORMAT
        elif args.prompt_template_name == "reflect_unseen":
            self.prompt_template = REFLECT_UNSEEN
        elif args.prompt_template_name == "reflect_fn":
            self.prompt_template = REFLECT_FN
        elif args.prompt_template_name == "reflect_boundary":
            self.prompt_template = REFLECT_BOUNDARY
        else:
            raise ValueError(f"Invalid ICL prompt template: {args.prompt_template_name}")