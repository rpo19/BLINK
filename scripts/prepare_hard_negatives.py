
import blink.candidate_ranking.utils as utils
import argparse
from tqdm import trange, tqdm
import os
import pandas as pd
import requests
import gzip
import json

def load_dataset(path):
    path2, ext = os.path.splitext(path)
    kwargs = {}
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

def main(params):
    input_samples = utils.read_dataset("train", params["data_path"], compression='gzip',
        max=params['max_dataset'], sample=params['sample_dataset'], seed=params['sample_dataset_seed'])

    batch_size = params['batch_size']

    print('Loading entities')

    entities = load_dataset(params['entities_path'])

    print('loaded')

    entities = entities[['id', 'title', 'parsed']].set_index('id', drop=True).copy()

    samples_with_hard_negatives = []

    x = 0
    for i in trange(0, len(input_samples), batch_size):
        batch = input_samples[i:i+batch_size]

        #   "href":
        #   "mention":
        #   "context_left":
        #   "context_right":
        #   "descr":
        #   "label":

        res_biencoder = requests.post(params['biencoder_url'], json=batch)
        assert res_biencoder.ok

        body = {
            'encodings': res_biencoder.json()['encodings'],
            'top_k': 2 # to ensure there is also a negative
        }
        res_indexer = requests.post(params['indexer_url'], json=body)
        # TODO do not risk to compromise the entire training for a single failure here
        assert res_indexer.ok

        candidates = res_indexer.json()

        # find the negative
        for sample, cands in zip(batch, candidates):
            negative_label = None
            for c in cands:
                if sample['label'] != c['wikipedia_id']:
                    negative_label = c['wikipedia_id']
                    # the first negative is the hardest one
                    break

            assert negative_label is not None

            negative_title, negative_descr = entities.loc[negative_label, ['title', 'parsed']].values

            negative_sample = {
                'href': negative_title,
                'mention': sample['mention'],
                'context_left': sample['context_left'],
                'context_right': sample['context_right'],
                'descr': negative_descr,
                'label': negative_label,
                'neg': 1
            }

            sample['neg'] = 0

            samples_with_hard_negatives.append(sample)
            samples_with_hard_negatives.append(negative_sample)



        x+=len(batch)


    with gzip.open(params['output_path'], 'wt') as fd:
        print('Saving to file', params['output_path'])
        for sample in tqdm(samples_with_hard_negatives):
                json.dump(sample, fd)
                fd.write('\n')

    assert x == len(input_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_training_args()
    # parser.add_eval_args()
    parser.add_argument(
        "--max-dataset", default=None, type=int, dest='max_dataset',
        help="Limit the dataset to this size."
    )
    parser.add_argument(
        "--batch-size", default=500, type=int, dest='batch_size',
        help="Batch size for creating hard negs dataset."
    )
    parser.add_argument(
        "--sample-dataset", default=None, type=int, dest='sample_dataset',
        help="Sample the dataset to this size."
    )
    parser.add_argument(
        "--sample-dataset-seed", default=None, type=int, dest='sample_dataset_seed',
        help="Sample with this seed."
    )
    parser.add_argument(
        #TODO int for how many hard negatives?
        "--hard-negatives", action="store_true", help="Whether to use hard-negatives.",
        dest='hard_negatives', default=False
    )
    parser.add_argument(
        "--biencoder-url",
        default=None,
        required=True,
        type=str,
        help="The url of the biencoder.",
        dest='biencoder_url'
    )
    parser.add_argument(
        "--indexer-url",
        default=None,
        required=True,
        type=str,
        help="The url of the indexer from where to extract hard negatives.",
        dest='indexer_url'
    )
    parser.add_argument(
        "--entities-path",
        required=True,
        default=None,
        type=str,
        help="Path of the entities from which to get descriptions.",
        dest='entities_path'
    )
    parser.add_argument(
        "--data_path",
        default="data/zeshel",
        type=str,
        help="The path to the train data.",
    )
    parser.add_argument(
        "--output_path",
        default="data/zeshel",
        required=True,
        type=str,
        help="The file in which to write the dataset with hard negatives.",
    )

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)