import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from blink.main_dense import load_crossencoder, prepare_crossencoder_data, _process_crossencoder_dataloader, _run_crossencoder
from blink.crossencoder.train_cross import modify
from typing import List
import json
import psycopg

class Id2Title(object):
    def __init__(self, dbconnection, tablename):
        self.dbconnection = dbconnection
        self.tablename = tablename
    def __getitem__(self, arg):
        # args is a tuple (id, indexer)
        with self.dbconnection.cursor() as cur:
            cur.execute("""
            SELECT
                title
            FROM
                entities
            WHERE
                id = %s AND
                indexer = %s;
            """, (int(arg[0]), int(arg[1])))
            title = cur.fetchone()
        return title

class Id2Text(object):
    def __init__(self, dbconnection, tablename):
        self.dbconnection = dbconnection
        self.tablename = tablename
    def __getitem__(self, arg):
        # args is a tuple (id, indexer)
        with self.dbconnection.cursor() as cur:
            cur.execute("""
            SELECT
                descr
            FROM
                entities
            WHERE
                id = %s AND
                indexer = %s;
            """, (int(arg[0]), int(arg[1])))
            text = cur.fetchone()
        return text

class Mention(BaseModel):
    label:str
    label_id:int
    context_left: str
    context_right:str
    mention: str
    start_pos:int
    end_pos: int
    sent_idx:int

class Candidate(BaseModel):
    id: int
    title: str
    url: str
    indexer: int
    score: float

class Item(BaseModel):
    samples: List[Mention]
    candidates: List[List[Candidate]]

app = FastAPI()

@app.post('/api/blink/crossencoder')
async def run(item: Item):
    samples = item.samples
    samples = [dict(s) for s in samples]

    candidates = item.candidates
    # converting candidates in tuples
    nns = []
    for cands in candidates:
        nn = []
        for _cand in cands:
            nn.append((_cand.id, _cand.indexer))
        nns.append(nn)

    labels = [-1] * len(samples)
    keep_all = True
    logger = None

    # prepare crossencoder data
    context_input, candidate_input, label_input = prepare_crossencoder_data(
        crossencoder.tokenizer, samples, labels, nns, id2title, id2text, keep_all,
    )

    context_input = modify(
        context_input, candidate_input, crossencoder_params["max_seq_length"]
    )

    dataloader = _process_crossencoder_dataloader(
        context_input, label_input, crossencoder_params
    )

    # run crossencoder and get accuracy
    _, index_array, unsorted_scores = _run_crossencoder(
        crossencoder,
        dataloader,
        logger,
        context_len=crossencoder_params["max_context_length"],
    )

    for sample, _candidates, _nns, _scores in zip(samples, candidates, nns, unsorted_scores):
        for _cand, _nn, _score in zip(_candidates, _nns, _scores):
            assert _cand.id == _nn[0]
            assert _cand.indexer == _nn[1]
            _cand.score = _score

        _candidates.sort(key=lambda x: x.score, reverse=True)

    return candidates

def load_models(args):
    # load crossencoder model
    with open(args.crossencoder_config) as json_file:
        crossencoder_params = json.load(json_file)
        crossencoder_params["path_to_model"] = args.crossencoder_model
    crossencoder = load_crossencoder(crossencoder_params)
    return crossencoder, crossencoder_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # crossencoder
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="Path to the crossencoder model.",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="Path to the crossencoder configuration.",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30302", help="port to listen at",
    )
    parser.add_argument(
        "--postgres", type=str, default=None, help="postgres url (e.g. postgres://user:password@localhost:5432/database)",
    )

    args = parser.parse_args()

    assert args.postgres is not None, 'Error. postgres url is required.'
    dbconnection = psycopg.connect(args.postgres)

    id2title = Id2Title(dbconnection=dbconnection, tablename='entities')
    id2text = Id2Text(dbconnection=dbconnection, tablename='entities')

    print('Loading crossencoder...')
    crossencoder, crossencoder_params = load_models(args)
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
