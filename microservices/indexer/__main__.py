import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
from typing import List
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import json
import redis

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

# db 0 --> id2title
id2title = None
# db 1 --> id2text
id2text = None
# db 2 --> local_id2wikipedia_id
local_id2wikipedia_id = None
# db 3 --> id2encoding
id2encoding = None

def id2url(id):
    wiki_id = local_id2wikipedia_id.get(id)
    return "https://en.wikipedia.org/wiki?curid={}".format(wiki_id.decode())

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
        for _scores, _cands, _enc in zip(scores, candidates, encodings):
            # for each samples
            n = 0
            for _score, _cand in zip(_scores, _cands):
                raw_score = float(_score)
                _cand = int(_cand)
                # compute dot product if hnsfw
                if isinstance(indexer, DenseHNSWFlatIndexer):
                    entity_enc = id2encoding.get(_cand)
                    entity_enc = vector_decode(entity_enc)
                    _score = np.dot(_enc, entity_enc)
                all_candidates_4_sample_n[n].append({
                        'raw_score': raw_score,
                        'id': _cand,
                        'title': id2title.get(_cand).decode(),
                        'url': id2url(_cand),
                        'indexer': index['name'],
                        'score': float(_score)
                    })
    # sort
    for _sample in all_candidates_4_sample_n:
        _sample.sort(key=lambda x: x['score'], reverse=True)
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
        "--redis-host", type=str, default="127.0.0.1", help="redis host",
    )
    parser.add_argument(
        "--redis-port", type=int, default="6379", help="redis port",
    )

    args = parser.parse_args()

    print('Loading indexes...')
    load_models(args)
    print('Loading complete.')

    # db 0 --> id2title
    id2title = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    # db 1 --> id2text
    id2text = redis.Redis(host=args.redis_host, port=args.redis_port, db=1)
    # db 2 --> local_id2wikipedia_id
    local_id2wikipedia_id = redis.Redis(host=args.redis_host, port=args.redis_port, db=2)
    # db 3 --> id2encoding
    id2encoding = redis.Redis(host=args.redis_host, port=args.redis_port, db=3)



    uvicorn.run(app, host = args.host, port = args.port)
