import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import argparse
import requests
import numpy as np

###
ner = '/api/ner'
biencoder = '/api/blink/biencoder' # mention # entity
biencoder_mention = f'{biencoder}/mention'
biencoder_entity = f'{biencoder}/entity'
crossencoder = '/api/blink/crossencoder'
indexer = '/api/indexer' # search # add
indexer_search = f'{indexer}/search'
indexer_add = f'{indexer}/add'
indexer_reset = f'{indexer}/reset/rw'
nilpredictor = '/api/nilprediction'
nilcluster = '/api/nilcluster'
###

nil_threshold = 0.5

### TODO make configurable
context_right = 'context_right'
context_left = 'context_left'
mention = 'mention'
###

class Input(BaseModel):
    text: str
    doc_id: Optional[int]
    populate: bool = False # whether to add the new entities to the kb or not
    save: bool = False # whether to save to db or not

app = FastAPI()

def prepare_for_nil_prediction(x):
    c = x['candidates']

    is_nil = False
    features = {}

    if len(c) == 0:
        is_nil = True
        return is_nil, features

    is_cross = 'is_cross' in c[0] and c[0]['is_cross']

    features = {}
    if not is_cross:
        # bi only
        features['max_bi'] = c[0]['score']
    else:
        # cross
        if 'bi_score' in c[0]:
            features['max_bi'] = c[0]['bi_score']
        features['max_cross'] = c[0]['score']

    features['mention'] = x[mention]
    features['title'] = c[0]['title']
    features['topcandidates'] = c

    return is_nil, features

@app.post('/api/pipeline')
async def run(input: Input):
    # NER

    res_ner = requests.post(
        args.baseurl + ner, json={'text': input.text})

    if res_ner.ok:
        sentences = res_ner.json()['sentences']
        entities = res_ner.json()['ents']
    else:
        print('NER error')
        print(res_ner)
        raise Exception('NER error')

    print('Found {} entities:'.format(len(entities)))
    print('Persons:', sum(1 for i in entities if i['ner_type'] == 'PER'))
    print('Locations:', sum(1 for i in entities if i['ner_type'] == 'LOC'))
    print('Organizations:', sum(1 for i in entities if i['ner_type'] == 'ORG'))
    print('Miscellaneous:', sum(1 for i in entities if i['ner_type'] == 'MISC'))

    data = pd.DataFrame(entities)



    dates = data.query('ner_type == "DATE"').copy()
    print('Found {} dates'.format(dates.shape[0]))

    data = data.query('ner_type != "DATE"').copy()
    print('Found {} non-dates'.format(data.shape[0]))

    # ## Entity Linking

    # ### Encoding
    print('Encoding entities...')
    res_biencoder = requests.post(
        args.baseurl + biencoder_mention,
            json=data[[
                'mention',
                'context_left',
                'context_right'
                ]].to_dict(orient='records'))

    if res_biencoder.ok:
        data['encoding'] = res_biencoder.json()['encodings']
    else:
        print('Biencoder ERROR')
        print(res_biencoder)
        raise Exception('Biencoder ERROR')

    print('Encoded {} entities.'.format(data.shape[0]))

    # ### Retrieval
    print('retrieval')
    body = {
        'encodings': data['encoding'].values.tolist(),
        'top_k': 10
    }
    res_indexer = requests.post(
        args.baseurl + indexer_search, json=body)

    if res_indexer.ok:
        candidates = res_indexer.json()
    else:
        print('ERROR with the indexer.')
        print(res_indexer)
        print(res_indexer.json())
        raise Exception('Indexer ERROR')

    if len(candidates) == 0 or len(candidates[0]) == 0:
        print('No candidates received.')

    data['candidates'] = candidates

    data[['is_nil', 'nil_features']] = data.apply(prepare_for_nil_prediction, axis=1, result_type='expand')

    # ## NIL prediction
    print('nil prediction')
    # prepare fields (default NIL)
    data['nil_score'] = np.zeros(data.shape[0])

    not_yet_nil = data.query('is_nil == False')

    if not_yet_nil.shape[0] > 0:
        res_nilpredictor = requests.post(
            args.baseurl + nilpredictor, json=not_yet_nil['nil_features'].values.tolist())
        if res_nilpredictor.ok:
            # TODO use cross if available
            nil_scores_bi = np.array(res_nilpredictor.json()['nil_score_bi'])
            nil_scores_cross = np.array(res_nilpredictor.json()['nil_score_bi'])
        else:
            print('ERROR during NIL prediction')
            print(res_nilpredictor)
            print(res_nilpredictor.json())

    data.loc[not_yet_nil.index, 'nil_score'] = nil_scores_bi
    data.loc[not_yet_nil.index, 'nil_score_cross'] = nil_scores_cross

    # if below threshold --> is NIL
    data['is_nil'] = data['nil_score'].apply(lambda x: x < nil_threshold)


    print('Estimated {} entities as NOT NIL'.format(data.eval('is_nil == False').sum()))
    print('Estimated {} entities as NIL'.format(data.eval('is_nil == True').sum()))

    data['top_title'] = data['candidates'].apply(lambda x: x[0]['title'])

    # ## Entity Clustering
    print('clustering')
    nil_mentions = data.query('is_nil == True')

    if nil_mentions.shape[0] == 0:
        # no NIL: skipping clustering
        print('Skipping clustering')
        clusters = []
    else:
        res_nilcluster = requests.post(
            args.baseurl + nilcluster, json={
                'ids': nil_mentions.index.tolist(),
                'mentions': nil_mentions[mention].values.tolist(),
                'encodings': nil_mentions['encoding'].values.tolist()
            })

        if not res_nilcluster.ok:
            print('NIL cluster ERROR')
            raise Exception('NIL cluster ERROR')
        else:
            print('OK')

        clusters = pd.DataFrame(res_nilcluster.json())

        clusters = clusters.sort_values(by='nelements', ascending=False)

    if input.populate and clusters:
        # populate with new entities
        print('Populating rw index with new entities')

        # inject url into cluster (most frequent one) # even NIL mentions still have the gold url
        def _helper(x):
            x = x['mentions_id']
            modes = data.loc[x, 'wikiId'].mode()
            if len(modes) > 1:
                mode = None
            else:
                mode = modes[0]
            return {'mode': mode, 'modes': modes}

        clusters[['mode', 'modes']] = clusters.apply(_helper, axis=1, result_type='expand')

        data_new = clusters[['title', 'center']].rename(columns={'center': 'encoding', 'mode': 'wikipedia_id'})
        new_indexed = requests.post(
            args.baseurl + indexer_add, json=data_new.to_dict(orient='records'))

        if not new_indexed.ok:
            print('error adding new entities')
            raise Exception('Populate ERROR')
        else:
            new_indexed = new_indexed.json()
            clusters['index_id'] = new_indexed['ids']
            clusters['index_indexer'] = new_indexed['indexer']
            prev_clusters = prev_clusters.append(clusters[['index_id', 'mode', 'modes']], ignore_index=True)
            prev_clusters.set_index('index_id', drop=False, inplace=True)

    data[['top_title', 'top_wikipedia_id']] = data.apply(lambda x: {
            'top_title': x['candidates'][0]['title'],
            'top_wikipedia_id': x['candidates'][0]['wikipedia_id']
        }, axis=1, result_type='expand')

    data_n_dates = pd.concat([data, dates]).sort_values(by='start_pos')

    # remove intersection entities # TODO improve
    prev = 0
    to_del = []
    for i,row in data_n_dates.iterrows():
        if prev > row['start_pos']:
            to_del.append(i)
        prev = row['end_pos']

    data_n_dates = data_n_dates[~data_n_dates.index.isin(to_del)]
    print('deleted {} rows from data_n_dates'.format(len(to_del)))

    data_n_dates['top_url'] = data_n_dates['candidates'].apply(lambda x: x[0]['url'] if isinstance(x, list) else None)

    data_n_dates['label'] = data_n_dates['label'].fillna("unknown")
    data_n_dates['label_id'] = data_n_dates['label_id'].fillna(-1)
    data_n_dates['context_left'] = data_n_dates['context_left'].fillna("")
    data_n_dates['context_right'] = data_n_dates['context_right'].fillna("")
    data_n_dates['mention'] = data_n_dates['mention'].fillna("")
    data_n_dates['start_pos'] = data_n_dates['start_pos'].fillna(0)
    data_n_dates['end_pos'] = data_n_dates['end_pos'].fillna(0)
    data_n_dates['sent_idx'] = data_n_dates['sent_idx'].fillna(0)
    data_n_dates['ner_type'] = data_n_dates['ner_type'].fillna("")
    if 'normalized_date' in data_n_dates.columns:
        data_n_dates['normalized_date'] = data_n_dates['normalized_date'].fillna("")
    else:
        data_n_dates['normalized_date'] = ""
    data_n_dates['encoding'] = data_n_dates['encoding'].fillna("")
    data_n_dates['candidates'] = data_n_dates['candidates'].fillna("")
    data_n_dates['is_nil'] = data_n_dates['is_nil'].fillna(False)
    data_n_dates['nil_features'] = data_n_dates['nil_features'].fillna("")
    data_n_dates['nil_score'] = data_n_dates['nil_score'].fillna(1)
    data_n_dates['nil_score_cross'] = data_n_dates['nil_score_cross'].fillna(1)
    data_n_dates['top_title'] = data_n_dates['top_title'].fillna("")
    data_n_dates['top_wikipedia_id'] = data_n_dates['top_wikipedia_id'].fillna(-1)
    data_n_dates['top_url'] = data_n_dates['top_url'].fillna("")

    outjson = data_n_dates[['context_left', 'context_right', 'mention',
                         'start_pos', 'end_pos', 'sent_idx',
                         'ner_type', 'normalized_date', 'candidates',
                         'top_title', 'top_wikipedia_id', 'top_url']].to_dict(orient='records')

    if input.save:
        # TODO save in mongo
        pass

    # TODO return also cluster?

    return outjson

if __name__ == '__main__':
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