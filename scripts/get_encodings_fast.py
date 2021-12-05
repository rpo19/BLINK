import sys
sys.path.append('.')
import json
from tqdm import tqdm
import os
import pickle
from torch.utils.data import DataLoader, SequentialSampler
from blink.biencoder.biencoder import load_biencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import torch
import numpy as np

if len(sys.argv) < 3:
    print('Usage: {} input_mentions.jsonl output_encodings.jsonl'.format(sys.argv[0]))
    sys.exit(1)

input_mentions = sys.argv[1]
save_encodings_path = sys.argv[2]

biencoder_config = "./models/biencoder_wiki_large.json"
biencoder_model = "./models/biencoder_wiki_large.bin"
top_k = 100

def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader

def __load_test(test_filename, kb2id, wikipedia_id2local_id, logger, consider_all=False):
    test_samples = []
    with open(test_filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            record = json.loads(line)
            record["label"] = str(record["label_id"])

            # for tac kbp we should use a separate knowledge source to get the entity id (label_id)
            if kb2id and len(kb2id) > 0:
                if record["label"] in kb2id:
                    record["label_id"] = kb2id[record["label"]]
                else:
                    if consider_all:
                        # NIL
                        record["label_id"] = -1
                    else:
                        continue

            # check that each entity id (label_id) is in the entity collection
            elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
                try:
                    key = int(record["label"].strip())
                    if key in wikipedia_id2local_id:
                        record["label_id"] = wikipedia_id2local_id[key]
                    else:
                        if consider_all:
                            # NIL
                            record["label_id"] = -1
                        else:
                            continue
                except:
                    if consider_all:
                        # NIL
                        record["label_id"] = -1
                    else:
                        continue

            # LOWERCASE EVERYTHING !
            record["context_left"] = record["context_left"].lower()
            record["context_right"] = record["context_right"].lower()
            record["mention"] = record["mention"].lower()
            test_samples.append(record)

    if logger:
        logger.info("{}/{} samples considered".format(len(test_samples), len(lines)))
    return test_samples

def __map_test_entities(test_entities_path, title2id, logger):
    # load the 732859 tac_kbp_ref_know_base entities
    kb2id = {}
    missing_pages = 0
    n = 0
    with open(test_entities_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            if entity["title"] not in title2id:
                missing_pages += 1
            else:
                kb2id[entity["entity_id"]] = title2id[entity["title"]]
            n += 1
    if logger:
        logger.info("missing {}/{} pages".format(missing_pages, n))
    return kb2id

def _get_test_samples(
    test_filename, test_entities_path, title2id, wikipedia_id2local_id, logger, consider_all=False
):
    kb2id = None
    if test_entities_path:
        kb2id = __map_test_entities(test_entities_path, title2id, logger)
    test_samples = __load_test(test_filename, kb2id, wikipedia_id2local_id, logger, consider_all=consider_all)
    return test_samples



def _run_biencoder_only_encodings(biencoder, dataloader, top_k=100, indexer=None, save_encodings=True):
    biencoder.model.eval()
    labels = []
    #nns = []
    #all_scores = []
    encodings = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            context_encoding = biencoder.encode_context(context_input).numpy()
            context_encoding = np.ascontiguousarray(context_encoding)
            if save_encodings:
                encodings.extend([e.tolist() for e in context_encoding])

        labels.extend(label_ids.data.numpy())
    return labels, encodings

###
# load biencoder model
print('Loading bi encoder...')
with open(biencoder_config) as json_file:
    biencoder_params = json.load(json_file)
    biencoder_params["path_to_model"] = biencoder_model
biencoder = load_biencoder(biencoder_params)

# load id2title
id2title_path = './data/id2title.pickle'
print('Loading id2title from {}'.format(id2title_path))
if os.path.isfile(id2title_path):
    with open(id2title_path, 'rb') as fd:
        id2title = pickle.load(fd)
else:
    raise Exception('{} not found! Generate it with `python blink/main_dense.py --save-id2title`.'.format(id2title_path))

title2id = {v:k for k,v in id2title.items()}

# load wikipedia_id2local_id
wikipedia_id2local_id_path = './data/wikipedia_id2local_id.pickle'
print('Loading wikipedia_id2local_id from {}'.format(wikipedia_id2local_id_path))
if os.path.isfile(wikipedia_id2local_id_path):
    with open(wikipedia_id2local_id_path, 'rb') as fd:
        wikipedia_id2local_id = pickle.load(fd)
else:
    raise Exception('{} not found! Generate it with `python blink/main_dense.py --save-wikipedia-id2local-id`.'.format(wikipedia_id2local_id_path))

local_id2wikipedia_id = {v:k for k,v in wikipedia_id2local_id.items()}

print('Starting...')
samples = _get_test_samples(
                    input_mentions,
                    None,
                    title2id,
                    wikipedia_id2local_id,
                    None,
                    consider_all= True
                )

dataloader = _process_biencoder_dataloader(
    samples, biencoder.tokenizer, biencoder_params
)

# run biencoder

labels, encodings = _run_biencoder_only_encodings(
    biencoder, dataloader, top_k, save_encodings=True
)

print('Saving encodings...')
with open(save_encodings_path, 'w') as fd:
    for _enc, _lab in zip(encodings, labels):
        assert len(_lab) == 1
        _lab = int(_lab[0])
        current = {
            "encoding": _enc,
            "label": _lab,
            "wikipedia_id": 0 if local_id2wikipedia_id is None or _lab not in local_id2wikipedia_id else local_id2wikipedia_id[_lab],
            "title": id2title[_lab] if _lab in id2title else "**NOTFOUND**"
        }
        json.dump(current, fd)
        fd.write('\n')
