import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import argparse
import requests
import numpy as np
import os
from gatenlp import Document

###
ner = '/api/ner'
biencoder = '/api/blink/biencoder' # mention # entity
biencoder_mention = f'{biencoder}/mention/doc'
biencoder_entity = f'{biencoder}/entity'
crossencoder = '/api/blink/crossencoder'
indexer = '/api/indexer' # search # add
indexer_search = f'{indexer}/search/doc'
indexer_add = f'{indexer}/add/doc'
indexer_reset = f'{indexer}/reset/rw'
nilpredictor = '/api/nilprediction/doc'
nilcluster = '/api/nilcluster/doc'
mongo = '/api/mongo/document'
###

class Input(BaseModel):
    text: str
    doc_id: Optional[int]
    populate: bool = False # whether to add the new entities to the kb or not
    save: bool = False # whether to save to db or not

app = FastAPI()

@app.post('/api/pipeline')
async def run(input: Input):

    doc = Document(input.text)
    if input.doc_id:
        doc.features['id'] = input.doc_id

    res_ner = requests.post(args.baseurl + ner, json=doc.to_dict())
    if not res_ner.ok:
        raise Exception('NER error')
    doc = Document.from_dict(res_ner.json())

    res_biencoder = requests.post(args.baseurl + biencoder_mention, json=doc.to_dict())
    if not res_biencoder.ok:
        raise Exception('Biencoder errror')
    doc = Document.from_dict(res_biencoder.json())

    res_indexer = requests.post(args.baseurl + indexer_search, json=doc.to_dict())
    if not res_indexer.ok:
        raise Exception('Indexer error')
    doc = Document.from_dict(res_indexer.json())

    res_nilprediction = requests.post(args.baseurl + nilpredictor, json=doc.to_dict())
    if not res_nilprediction.ok:
        raise Exception('NIL prediction error')
    doc = Document.from_dict(res_nilprediction.json())

    res_clustering = requests.post(args.baseurl + nilcluster, json=doc.to_dict())
    if not res_clustering.ok:
        raise Exception('Clustering error')
    doc = Document.from_dict(res_clustering.json())

    # TODO add new entities to the KB
    if input.populate:
        # get clusters
        res_populate = requests.post(args.baseurl + indexer_add, json=doc.to_dict())
        if not res_populate.ok:
            raise Exception('Population error')
        doc = Document.from_dict(res_populate.json())

    # TODO save annotations in mongodb

    if not 'pipeline' in doc.features:
        doc.features['pipeline'] = []
    doc.features['pipeline'].append('pipeline')

    return doc.to_dict()

if __name__ == '__main__':

    user = os.environ.get('AUTH_USER', None)
    password = os.environ.get('AUTH_PASSWORD', None)

    if user:
        auth =(user, password)
    else:
        auth = None

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )

    parser.add_argument(
        "--port", type=int, default="30310", help="port to listen at",
    )

    parser.add_argument(
        "--api-baseurl", type=str, default=None, help="Baseurl to call all the APIs", dest='baseurl', required=True
    )
    args = parser.parse_args()

    uvicorn.run(app, host = args.host, port = args.port)