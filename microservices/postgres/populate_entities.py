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

max_title_len = 100
chunksize = 500

title2id = {}
id2title = {}
id2text = {}
wikipedia_id2local_id = {}
local_id2wikipedia_id = {}
id2url = {}

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

def load_models(args):
    local_idx = 0
    if args.entity_catalogue.endswith('gz'):
        fin = gzip.open(args.entity_catalogue, "rt")
    else:
        fin = open(args.entity_catalogue, "r")

    lines = fin.readlines()
    fin.close()

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

    return torch.load(args.entity_encoding)

def populate(entity_encodings, connection, table_name):
    assert entity_encodings[0].numpy().dtype == 'float32'

    global wikipedia_id2local_id
    global id2url
    global title2id
    global id2title
    global local_id2wikipedia_id
    global id2text
    del wikipedia_id2local_id
    del id2url
    del title2id

    print('Loading into dataframe...')
    df = pd.DataFrame(columns=['id', 'indexer', 'wikipedia_id', 'title', 'descr', 'embedding'])

    df['id'] = pd.Series(id2title.keys(), dtype=int)
    df['indexer'] = pd.Series([0] * len(id2title), dtype=int) # indexer = 0
    df['title'] = pd.Series(id2title.values(), dtype=str)
    del id2title

    df['wikipedia_id'] = pd.Series(local_id2wikipedia_id.values(), dtype=int)
    del local_id2wikipedia_id

    df['descr'] = pd.Series(id2text.values(), dtype=str)
    del id2text

    # embedding are string encoded
    df['embedding'] = pd.Series(map(lambda tensor: vector_encode(tensor.numpy()), entity_encodings), dtype=str)
    del entity_encodings

    print('Truncating titles to {} characters strings'.format(max_title_len))
    df['title'] = df['title'].apply(lambda x: x[0:max_title_len])

    print('Saving to postgres...')
    print('Shape', df.shape)

    with connection.cursor() as cursor:
        with cursor.copy("COPY {} (id, indexer, wikipedia_id, title, descr, embedding) FROM STDIN".format(table_name)) as copy:
            max_i = df.shape[0]
            for i, record in tqdm(df.iterrows(), total=df.shape[0]):
                copy.write_row(tuple(record.values))
    connection.commit()

    #df.to_sql(table_name, engine, if_exists='append', method='multi', index=False, chunksize=chunksize)
    print('Done.')



    # ...



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

    args = parser.parse_args()

    assert args.table_name is not None, 'Error: table-name is required!'


    connection = psycopg.connect(args.postgres)

    print('Loading entities...')
    entity_encodings = load_models(args)

    print('Populating postgres...')
    populate(entity_encodings, connection, args.table_name)

    connection.close()

