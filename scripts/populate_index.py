import argparse
import psycopg
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import os
import pandas as pd
from tqdm import tqdm
import requests
import numpy as np
import base64
from tqdm import trange
import gzip

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

def populate(args, data):

    # input: embeddings --> faiss
    # --> postgres
    # wikipedia_id ?
    # title
    # descr ?
    # embedding

    # get vector size from the embeddings
    vector_size = vector_decode(data.iloc[0]['encoding']).shape[0]

    if args.indexer_type == 'flat':
        indexer = DenseFlatIndexer(vector_size)
    elif args.indexer_type == 'hnsw':
        indexer = DenseHNSWFlatIndexer(vector_size)
    else:
        raise Exception('Unknown index type {}.'.format(args.indexer_type))

    indexid = args.indexer_id
    indexpath = args.output

    # add to index
    print('Decoding embeddings...')
    embeddings = [vector_decode(e.encoding) for i, e in tqdm(data.iterrows(), total=data.shape[0])]
    embeddings = np.stack(embeddings).astype('float32')
    print('Indexing data...')
    indexer.index_data(embeddings)
    ids = list(range(indexer.index.ntotal - embeddings.shape[0], indexer.index.ntotal))
    # save index
    print(f'Saving index {indexid} to disk...')
    indexer.serialize(indexpath)

    # add to postgres
    print('Populating db...')
    with dbconnection.cursor() as cursor:
        with cursor.copy("COPY entities (id, indexer, wikipedia_id, title, descr, embedding, wikidata_qid, redirects_to) FROM STDIN") as copy:
            for id, (i, row) in tqdm(zip(ids, data.iterrows()), total=data.shape[0]):
                wikipedia_id = -1 if row[args.id_key] is None or isnan(row[args.id_key]) else row[args.id_key]
                wikidata_qid = -1 if row[args.qid_key] is None or isnan(row[args.qid_key]) else row[args.qid_key]
                redirects_to = -1 if row[args.rd_key] is None or isnan(row[args.rd_key]) else row[args.rd_key]
                title = row[args.title_key]
                if len(title) > 100:
                    print('Found big title:', i, title)
                    title = title[:100]
                copy.write_row((id, indexid, wikipedia_id, title, row[args.descr_key], row['encoding'], wikidata_qid, redirects_to))
    dbconnection.commit()
    print('Done.')

def load_dataset(path):
    path2, ext = os.path.splitext(path)
    if ext == '.gz':
        kwargs = {'compression': 'gzip'}
        _, ext = os.path.splitext(path2)
    if ext == '.jsonl':
        df = pd.read_json(path, lines=True, **kwargs)
    elif ext == '.pickle':
        df = pd.read_pickle(path, **kwargs)
    else:
        raise Exception('Error. Not implemented for {} datasets.'.format(ext))
    return df

def pandas_batch(df, batchsize):
    for i in range(0, df.shape[0], batchsize):
        yield df.iloc[i:i+batchsize]

def biencoder_get_encodings(args, df):
    encodings = []
    print('Getting encoding from biencoder...')
    total = int(df.shape[0] / args.batchsize)
    for i, batch in tqdm(enumerate(pandas_batch(df, args.batchsize)), total=total):
        body = batch.apply(lambda x: {'title': x[args.title_key], 'descr': x[args.descr_key]}, axis=1).tolist()
        response = requests.post(args.biencoder, json=body)
        if not response.ok:
            raise Exception('Error from biencoder at batch {}'.format(i))
        encodings.extend(response.json()['encodings'])
    return encodings

def main(args):
    # load datasets
    datasets = args.input.split(',')
    df_list = list(map(load_dataset, datasets))
    data = pd.concat(df_list)
    del df_list
    ## remove duplicates
    data = data.drop_duplicates(subset=[args.id_key], keep='first')

    ## remove unneeded fields
    data = data[[args.id_key, args.title_key, args.descr_key, args.qid_key, args.rd_key]].copy()

    if args.skip_empty_descr:
        data = data.query("{} != ''".format(args.descr_key))

    # get encodings
    data['encoding'] = biencoder_get_encodings(args, data)

    populate(args, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--output", type=str, default=None, help="Output path of the index",
    )
    parser.add_argument(
        "--index-id", type=int, default=-1, help="Indexer id", dest='indexer_id'
    )
    parser.add_argument(
        "--index-type", type=str, default='flat', help="Indexer type (flat or hnsw)", dest='indexer_type'
    )
    parser.add_argument(
        "--postgres", type=str, default=None, help="postgres url (e.g. postgres://user:password@localhost:5432/database)",
    )
    # parser.add_argument(
    #     "--vector-size", type=int, default="1024", help="The size of the vectors", dest="vector_size",
    # )
    parser.add_argument(
        "--biencoder", type=str, default='http://localhost:30300/api/blink/biencoder/entity', help='biencoder url.'
    )
    parser.add_argument(
        "--input", type=str, default=None, help='Input datasets paths (comma separated).'
    )
    parser.add_argument(
        "--id-key", type=str, default='label_id', help='Id key.', dest="id_key"
        # id
    )
    parser.add_argument(
        "--qid-key", type=str, default='qid', help='Wikidata QId key.', dest="qid_key"
        # id
    )
    parser.add_argument(
        "--rd-key", type=str, default='rd_to', help='Redirect key.', dest="rd_key"
        # id
    )
    parser.add_argument(
        "--title-key", type=str, default='label_title', help='Title key.', dest="title_key"
        # title
    )
    parser.add_argument(
        "--descr-key", type=str, default='label', help='Description key.', dest="descr_key"
        # parsed
    )
    parser.add_argument(
        "--skip-empty-descr", default=False, action='store_true', help='Skip entities with no description', dest='skip_emtpy_descr'
    )
    parser.add_argument(
        "--batchsize", type=int, default="200", help="Batchsize for biencoder requests",
    )

    args = parser.parse_args()

    assert args.input is not None, 'Error. input is required.'
    assert args.output is not None, 'Error. output is required.'
    assert args.postgres is not None, 'Error. postgres url is required.'
    assert args.indexer_id != -1, 'Error. index-id required.'
    assert args.indexer_type == 'flat' or args.indexer_type == 'hnsw', 'Error. Unknown index type {}.'.format(args.indexer_type)
    dbconnection = psycopg.connect(args.postgres)

    main(args)

    dbconnection.close()
