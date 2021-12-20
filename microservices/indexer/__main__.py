import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
from typing import List, Optional
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import json
import psycopg
import os

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
    if wikipedia_id > 0:
        return "https://en.wikipedia.org/wiki?curid={}".format(wikipedia_id)
    else:
        return ""

app = FastAPI()

@app.post('/api/indexer/search')
async def search(input_: Input):
    encodings = input_.encodings
    top_k = input_.top_k
    encodings = np.array([vector_decode(e) for e in encodings])
    all_candidates_4_sample_n = []
    for i in range(len(encodings)):
        all_candidates_4_sample_n.append([])
    for index in indexes:
        indexer = index['indexer']
        scores, candidates = indexer.search_knn(encodings, top_k)
        n = 0
        for _scores, _cands, _enc in zip(scores, candidates, encodings):
            # for each samples
            for _score, _cand in zip(_scores, _cands):
                raw_score = float(_score)
                _cand = int(_cand)
                if _cand == -1:
                    # -1 means no other candidates found
                    break
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
            n += 1
    # sort
    for _sample in all_candidates_4_sample_n:
        _sample.sort(key=lambda x: x['score'], reverse=True)
    return all_candidates_4_sample_n

class Item(BaseModel):
    encoding: str
    wikipedia_id: Optional[int]
    title: str
    descr: Optional[str]

@app.post('/api/indexer/add')
async def add(items: List[Item]):
    if rw_index is None:
        raise HTTPException(status_code=404, detail="No rw index!")

    # input: embeddings --> faiss
    # --> postgres
    # wikipedia_id ?
    # title
    # descr ?
    # embedding

    indexer = indexes[rw_index]['indexer']
    indexid = indexes[rw_index]['indexid']
    indexpath = indexes[rw_index]['path']

    # add to index
    embeddings = [vector_decode(e.encoding) for e in items]
    embeddings = np.stack(embeddings).astype('float32')
    indexer.index_data(embeddings)
    ids = list(range(indexer.index.ntotal - embeddings.shape[0], indexer.index.ntotal))
    # save index
    print(f'Saving index {indexid} to disk...')
    indexer.serialize(indexpath)

    # add to postgres
    with dbconnection.cursor() as cursor:
        with cursor.copy("COPY entities (id, indexer, wikipedia_id, title, descr, embedding) FROM STDIN") as copy:
            for id, item in zip(ids, items):
                wikipedia_id = -1 if item.wikipedia_id is None else item.wikipedia_id
                copy.write_row((id, indexid, wikipedia_id, item.title, item.descr, item.encoding))
    dbconnection.commit()

    return {
        'ids': ids,
        'indexer': indexid
    }

def load_models(args):
    assert args.index is not None, 'Error! Index is required.'
    for index in args.index.split(','):
        index_type, index_path, indexid, rorw = index.split(':')
        print('Loading {} index from {}, mode: {}...'.format(index_type, index_path, rorw))
        if os.path.isfile(index_path):
            if index_type == "flat":
                indexer = DenseFlatIndexer(1)
            elif index_type == "hnsw":
                indexer = DenseHNSWFlatIndexer(1)
            else:
                raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
            indexer.deserialize_from(index_path)
        else:
            if index_type == "flat":
                indexer = DenseFlatIndexer(1024)
            elif index_type == "hnsw":
                indexer = DenseHNSWFlatIndexer(1024)
            else:
                raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexes.append({
            'indexer': indexer,
            'indexid': int(indexid),
            'path': index_path
            })

        global rw_index
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
