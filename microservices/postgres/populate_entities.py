import json
import argparse
import numpy as np
import torch
import base64
import pandas as pd
import sys
import psycopg
import io
from tqdm import tqdm
import gzip
import itertools

max_title_len = 100
chunksize = 500

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

def ents_generator(args):
    indexer = args.indexer

    if args.entity_catalogue.endswith('gz'):
        fin = gzip.open(args.entity_catalogue, "rt")
    else:
        fin = open(args.entity_catalogue, "r")

    for local_idx in itertools.count():
        line = fin.readline()
        # if line is empty
        # end of file is reached
        if not line:
            break

        entity = json.loads(line)

        split = entity["idx"].split("curid=")
        if len(split) > 1:
            wikipedia_id = int(split[-1].strip())
        else:
            wikipedia_id = int(entity["idx"].strip())

        title = entity["title"]
        title = title[:max_title_len]
        text = entity["text"]

        yield local_idx, wikipedia_id, indexer, title, text

    fin.close()

def load_models(args):
    return torch.load(args.entity_encoding)

def populate(entity_encodings, connection, table_name):
    assert entity_encodings[0].numpy().dtype == 'float32'

    total = entity_encodings.shape[0]

    with connection.cursor() as cursor:
        with cursor.copy("COPY {} (id, indexer, wikipedia_id, title, descr, embedding) FROM STDIN".format(table_name)) as copy:
            for (id, wikipedia_id, indexer, title, text), tensor in tqdm(zip(ents_generator(args), entity_encodings), total=total):
                embedding = vector_encode(tensor.numpy())
                copy.write_row((id, indexer, wikipedia_id, title, text, embedding))
    connection.commit()
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--postgres", type=str, default="", help="postgresql url (e.g. postgresql://username:password@localhost:5432/mydatabase)",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--table-name",
        dest="table_name",
        type=str,
        default=None,  # ALL WIKIPEDIA!
        help="Postgres table name.",
    )
    parser.add_argument(
        "--indexer",
        dest="indexer",
        type=int,
        default=0,
        help="Indexer id.",
    )

    args = parser.parse_args()

    assert args.table_name is not None, 'Error: table-name is required!'

    connection = psycopg.connect(args.postgres)

    print('Loading entities...')
    entity_encodings = load_models(args)

    print('Populating postgres...')
    populate(entity_encodings, connection, args.table_name)

    connection.close()

