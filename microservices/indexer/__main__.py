import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
from typing import List
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import json

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

class Input(BaseModel):
    encodings: List[str]
    top_k: int

indexes = []
rw_index = None

title2id = {}
id2title = {}
id2text = {}
wikipedia_id2local_id = {}
local_id2wikipedia_id = {}
id2url = {}

app = FastAPI()

@app.post('/api/indexer/search')
async def search(input_: Input):
    encodings = input_.encodings
    top_k = input_.top_k
    encodings = [vector_decode(e) for e in encodings]
    all_candidates_4_sample_n = [[]] * len(encodings)
    for index in indexes:
        indexer = index['indexer']
        scores, candidates = indexer.search_knn(encodings, top_k)
        for _scores, _cands in zip(scores, candidates):
            # for each samples
            n = 0
            for _score, _cand in zip(_scores, _cands):
                all_candidates_4_sample_n[n].append({
                        'raw_score': float(_score),
                        'id': int(_cand),
                        'indexer': index['name'],
                        'score': float(_score)
                    })
    return all_candidates_4_sample_n

@app.post('/api/indexer/add')
async def add():
    # TODO implement add new entity to rw_index
    pass

def load_models(args):
    assert args.index is not None, 'Error! Index is required.'
    for index in args.index.split(','):
        index_type, index_path, name, rorw = index.split(':')
        print('Loading {} index from {}, mode: {}...'.format(index_type, index_path, rorw))
        if index_type == "flat":
            indexer = DenseFlatIndexer(1)
        elif index_type == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)
        indexes.append({
            'indexer': indexer,
            'name': name,
            'path': index_path
            })

        if rorw == 'rw':
            assert rw_index is None, 'Error! Only one rw index is accepted.'
            rw_index = len(indexes) - 1 # last added

    # load all the 5903527 entities
    # TODO move entitis in a db
    print('Loading entities')
    local_idx = 0
    with open(args.entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1

    for k,v in wikipedia_id2local_id.items():
        local_id2wikipedia_id[v] = k

    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--index", type=str, default=None, help="comma separate list of paths to load indexes [type:path:name:ro/rw] (e.g: hnsw:index.pkl:wiki:ro,flat:index2.pkl:local:rw)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30300", help="port to listen at",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    args = parser.parse_args()

    print('Loading indexes...')
    load_models(args)
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
