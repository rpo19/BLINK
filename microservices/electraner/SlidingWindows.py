# %%
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import List
import numpy as np

class SlidingWindowsPipeline:

    def __init__(self, model, tokenizer, grouped_entities=True, ignore_subwords=True, cache_dir="D:\\Sgmon\\.cache"):

        self.model = model
        self.tokenizer = tokenizer
        self.grouped_entities = grouped_entities
        self.ignore_subwords = ignore_subwords

    def __call__(self, text: str, ignore_subword=False):
        strided = self.tokenizer(text, max_length=512, stride=128, truncation=True, padding=True,
                                 # return_token_type_ids=True,
                                 return_overflowing_tokens=True,
                                 return_offsets_mapping=True,
                                 # return_special_tokens_mask=True,
                                 return_tensors="pt")
        trfout = self.model(input_ids=strided['input_ids'],
                            # token_type_ids=strided['token_type_ids'],
                            attention_mask=strided['attention_mask'])
        output_tensors = trfout.logits.detach()
        predictions = [torch.argmax(x, dim=1) for x in output_tensors]
        entities_list = []
        for i, labels in enumerate(predictions):
            entities = []
            for y, label in enumerate(labels):
                start_ind, end_ind = strided[i].offsets[y]
                word_ref = text[start_ind: end_ind]
                word = self.tokenizer.convert_ids_to_tokens(strided[i].ids[y])
                if start_ind == 0 and end_ind == 0 and (word == '[CLS]' or word == '[SEP]'):
                    # skip CLS and SEP
                    print('skip', word)
                    continue
                is_subword = len(word_ref) != len(word)
                if (strided[i].ids[y]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
                entity = {
                    "word": word,
                    "score": torch.softmax(output_tensors[i][y], dim=0)[predictions[i][y].item()].item(),
                    "entity": self.model.config.id2label[predictions[i][y].item()],
                    "index": i * len(labels.tolist()) + (y + 1),
                    "start": start_ind,
                    "end": end_ind,
                    'is_subword': is_subword
                }
                entities += [entity]
            if self.grouped_entities:
                entities_list += [self.group_entities(entities)]
            else:
                entities_list += [entities]

        return self.remove_outside(entities_list)

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:

            is_last_idx = entity["index"] == last_idx
            is_subword = self.ignore_subwords and entity["is_subword"]
            if not entity_group_disagg:
                entity_group_disagg += [entity]
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated
            # entity group. The split is meant to account for the "B" and "I" suffixes Shouldn't merge if both
            # entities are B-type
            if (
                    (
                            entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                            and entity["entity"].split("-")[0] != "B"
                    )
                    and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ) or is_subword:
                # Modify subword type to be previous_type
                if is_subword:
                    entity["entity"] = entity_group_disagg[-1]["entity"].split("-")[-1]
                    entity["score"] = np.nan  # set ignored scores to nan and use np.nanmean

                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]

        return entity_groups

    def remove_outside(self, entities_list):
        final_entities = []
        for x in entities_list:
            entities = []
            for y in x:
                if self.grouped_entities:
                    if not y['entity_group'] == 'O':
                        entities.append(y)
                else:
                    if not y['entity'] == 'O':
                        entities.append(y)
            final_entities.append(entities)
        return final_entities


def get_unique_entities(entities_lists: List):
    total_entities = [entity for entities in entities_lists for entity in entities]
    seen = set()
    new_l = []
    for entity in total_entities:
        copy = entity.copy()
        copy.pop('score')
        t = tuple(copy.items())
        if t not in seen:
            seen.add(t)
            new_l.append(entity)
    return sorted(new_l, key=lambda k: k['start'])

# # %%
# from transformers import AutoModelForTokenClassification, AutoTokenizer
# # %%
# model = AutoModelForTokenClassification.from_pretrained('/home/rpo/Scaricati/electra')
# tokenizer = AutoTokenizer.from_pretrained('dbmdz/electra-base-italian-xxl-cased-discriminator')
# # %%
# sw = SlidingWindowsPipeline(model, tokenizer)

# # %%
# sw2 = SlidingWindowsPipeline(model, tokenizer, grouped_entities=False)
# # %%
# # %%
# res = sw(text)
# # %%
# res2 = sw2(text)
# # %%
