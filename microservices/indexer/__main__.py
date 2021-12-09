import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
from typing import List
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer

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
app = FastAPI()

@app.post('/api/indexer/search')
async def search(input_: Input):
    encodings = input_.encodings
    top_k = input_.top_k
    encodings = [vector_decode(e) for e in encodings]
    all_candidates = []
    for index in indexes:
        indexer = index['indexer']
        scores, candidates = indexer.search_knn(encodings, top_k)
        for _scores, _cands in zip(scores, candidates):
            all_candidates.append({
                'raw_scores': _scores.tolist(),
                'candidates': _cands.tolist(),
                'indexer': index['name'],
                'scores': _scores.tolist() # is hnsw compute dot-prod # entity encs are required
            })
    return all_candidates


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

    args = parser.parse_args()

    print('Loading indexes...')
    load_models(args)
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
