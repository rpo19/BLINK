import argparse
import requests
from pprint import pprint
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os

def pandas_batch(df, batchsize):
    for i in range(0, df.shape[0], batchsize):
        yield df.iloc[i:i+batchsize]

def load_dataset(path):
    _, ext = os.path.splitext(path)
    if ext == '.jsonl':
        df = pd.read_json(path, lines=True)
    elif ext == '.pickle':
        df = pd.read_pickle(path)
    else:
        raise Exception('Error. Not implemented for {} datasets.'.format(ext))
    return df

def biencoder_get_mentions_encodings(args, df):
    encodings = []
    print('Encoding mentions with biencoder...')
    total = int(df.shape[0] / args.batchsize)
    for i, batch in tqdm(enumerate(pandas_batch(df, args.batchsize)), total=total):
        body = batch.apply(lambda x: {
                'label': '',
                'label_id': -1,
                'context_left': x[args.context_left_key],
                'mention': x[args.mention_key],
                'context_right': x[args.context_right_key],
                'start_pos': len(x[args.context_left_key]),
                'end_pos': len(x[args.context_left_key]) + len(x[args.mention_key]),
                'sent_idx': 0
            }, axis=1).tolist()
        response = requests.post(args.biencoder, json=body)
        if not response.ok:
            raise Exception('Error from biencoder at batch {}'.format(i))
        encodings.extend(response.json()['encodings'])
    return encodings

def indexer_search(args, df):
    top_cands = []
    print('Indexer search...')
    total = int(df.shape[0] / args.batchsize)
    for i, batch in tqdm(enumerate(pandas_batch(df, args.batchsize)), total=total):
        body = {
            'encodings': batch['encoding'].tolist(),
            'top_k': args.top_k
        }
        response = requests.post(args.indexer, json=body)
        if not response.ok:
            raise Exception('Error from indexer at batch {}'.format(i))
        top_cands.extend(response.json())
    return top_cands

def get_found_at(args, df, top_candidates):
    all_found_at = []
    print('Evaluation...')
    assert len(top_candidates) == df.shape[0]
    for (i,row), cands in tqdm(zip(df.iterrows(), top_candidates), total=df.shape[0]):
        correct_id = row[args.id_key]
        _found_at = -1
        for i,c in enumerate(cands):
            if c[args.indexer_wiki_id] == correct_id:
                _found_at = i
                break
        all_found_at.append(_found_at)

    return all_found_at

def calc_recall(args, all_found_at):
    levels = np.array([int(l)-1 for l in args.recall_levels.split(',')])
    recall_at = np.zeros(len(levels), dtype=int)
    for _found_at in all_found_at:
        recall_at += np.logical_and(_found_at >= 0, _found_at <= levels).astype(int)
    recall_at = recall_at / len(all_found_at)
    return recall_at

def main(args):
    data = load_dataset(args.input)

    data['encoding'] = biencoder_get_mentions_encodings(args, data)

    top_candidates = indexer_search(args, data)

    all_found_at = get_found_at(args, data, top_candidates)

    recall_at = calc_recall(args, all_found_at)

    levels = ['recall@{}'.format(i) for i in args.recall_levels.split(',')]
    recall_at = [float(r) for r in recall_at]
    recall_dict = dict(zip(levels, recall_at))

    print('----- Results -----')
    pprint(recall_dict)
    with open(args.output, 'w') as fd:
        json.dump(recall_dict, fd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--output", type=str, default=None, help="Output path of the evaluation report",
    )
    parser.add_argument(
        "--biencoder", type=str, default='http://localhost:30300/api/blink/biencoder/mention', help='biencoder url.'
    )
    parser.add_argument(
        "--indexer", type=str, default='http://localhost:30301/api/indexer/search', help='indexer url.'
    )
    parser.add_argument(
        "--input", type=str, default=None, help='Input dataset path.'
    )
    parser.add_argument(
        "--id-key", type=str, default='label_id', help='Id key.', dest="id_key"
        # id
    )
    parser.add_argument(
        "--mention-key", type=str, default='mention', help='Mention key.', dest="mention_key"
    )
    parser.add_argument(
        "--context-left-key", type=str, default='context_left', help='Context left key.', dest="context_left_key"
    )
    parser.add_argument(
        "--context-right-key", type=str, default='context_right', help='Context right key.', dest="context_right_key"
    )
    parser.add_argument(
        "--indexer-id-key", type=str, default='wikipedia_id', help='Id key from indexer.', dest="indexer_wiki_id"
    )
    parser.add_argument(
        "--recall-levels", type=str, default='1,3,5,10,20,50,100', help='Recall@k levels. (first is 1)', dest="recall_levels"
    )
    parser.add_argument(
        "--batchsize", type=int, default=400, help="Batchsize for biencoder requests",
    )
    parser.add_argument(
        "--top-k", type=int, default=100, help="top k", dest='top_k'
    )

    args = parser.parse_args()

    main(args)
