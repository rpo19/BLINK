import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
from typing import List
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import json
import psycopg

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

def id2url(wikipedia_id):
    return "https://en.wikipedia.org/wiki?curid={}".format(wikipedia_id)

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
                    # query with embedding
                    with dbconnection.cursor() as cur:
                        cur.execute("""
                        SELECT
                            title, wikipedia_id, embedding
                        FROM
                            entities
                        WHERE
                            id = %s AND
                            indexer = %s;
                        """, (_cand, index['indexid']))

                        title, wikipedia_id, embedding = cur.fetchone()

                    embedding = vector_decode(embedding)
                    _score = np.dot(_enc, embedding)
                else:
                    # simpler query
                    with dbconnection.cursor() as cur:
                        cur.execute("""
                        SELECT
                            title, wikipedia_id
                        FROM
                            entities
                        WHERE
                            id = %s AND
                            indexer = %s;
                        """, (_cand, index['indexid']))

                        title, wikipedia_id = cur.fetchone()

                all_candidates_4_sample_n[n].append({
                        'raw_score': raw_score,
                        'id': _cand,
                        'title': title,
                        'url': id2url(wikipedia_id),
                        'indexer': index['indexid'],
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
        index_type, index_path, indexid, rorw = index.split(':')
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
            'indexid': int(indexid),
            'path': index_path
            })

        if rorw == 'rw':
            assert rw_index is None, 'Error! Only one rw index is accepted.'
            rw_index = len(indexes) - 1 # last added

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--index", type=str, default=None, help="comma separate list of paths to load indexes [type:path:indexid:ro/rw] (e.g: hnsw:index.pkl:0:ro,flat:index2.pkl:1:rw)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30301", help="port to listen at",
    )
    parser.add_argument(
        "--postgres", type=str, default=None, help="postgres url (e.g. postgres://user:password@localhost:5432/database)",
    )

    args = parser.parse_args()

    assert args.postgres is not None, 'Error. postgres url is required.'
    dbconnection = psycopg.connect(args.postgres)

    print('Loading indexes...')
    load_models(args)
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
    dbconnection.close()
