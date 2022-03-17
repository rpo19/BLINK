import click
from tqdm import tqdm
import requests
import re
import numpy as np
import pandas as pd
import json
import sys
import os
import base64
from Packages.TimeEvolving import Cluster
from pprint import pprint

### TODO move all them behind a single proxy and set configurable addresses
biencoder = 'http://localhost:30300/api/blink/biencoder' # mention # entity
biencoder_mention = f'{biencoder}/mention'
biencoder_entity = f'{biencoder}/entity'
crossencoder = 'http://localhost:30302/api/blink/crossencoder'
indexer = 'http://localhost:30301/api/indexer' # search # add
indexer_search = f'{indexer}/search'
indexer_add = f'{indexer}/add'
indexer_reset = f'{indexer}/reset/rw'
nilpredictor = 'http://localhost:30303/api/nilprediction'
nilcluster = 'http://localhost:30305/api/nilcluster'
###

### TODO make configurable
context_right = 'right_context_text'
context_left = 'left_context_text'
mention = 'word'
###

# wikipedia_id = mode(cluster.wikipedia_ids)
added_entities = pd.DataFrame()
# clusters with multiple modes
prev_clusters = pd.DataFrame()

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

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

def get_new_cand(x):
            url = x['y_wikiurl_dump']
            curid = x['wikiId']
            candidates = x['candidates']
            score = candidates[0]['score'] + 1
            _cand = {
                'wikipedia_id': curid,
                'title': 'Marwari language',
                'url': url,
                'indexer': -1,
                'score': score,
                'hitl': True
            }
            candidates.insert(0, _cand)
            return candidates

def run_batch(batch, add_correct, hitl, no_add, save_path, reset,
        biencoder=biencoder,
        biencoder_mention=biencoder_mention,
        biencoder_entity=biencoder_entity,
        crossencoder=crossencoder,
        indexer=indexer,
        indexer_search=indexer_search,
        indexer_add=indexer_add,
        nilpredictor=nilpredictor,
        nilcluster=nilcluster):

    global added_entities
    global prev_clusters

    print('Run batch', batch)

    if reset:
        print('Resetting RW index...')
        res_reset = requests.post(indexer_reset, data={})

        if res_reset.ok:
            print('Reset done.')
        else:
            print('ERROR while resetting!')
            sys.exit(1)

    print('Loading batch...')
    data = pd.read_json(batch, lines=True)

    # ## Entity Linking

    # ### Encoding
    print('Encoding entities...')
    res_biencoder = requests.post(biencoder_mention,
            json=data.rename(columns={
                mention: 'mention',
                context_left: 'context_left',
                context_right: 'context_right'
                }).to_dict(orient='records'))

    if res_biencoder.ok:
        data['encoding'] = res_biencoder.json()['encodings']
    else:
        print('Biencoder ERROR')
        print(res_biencoder)
        sys.exit(1)

    print('Encoded {} entities.'.format(data.shape[0]))

    # ### Retrieval
    print('retrieval')
    body = {
        'encodings': data['encoding'].values.tolist(),
        'top_k': 10
    }
    res_indexer = requests.post(indexer_search, json=body)

    if res_indexer.ok:
        candidates = res_indexer.json()
    else:
        print('ERROR with the indexer.')
        print(res_indexer)
        print(res_indexer.json())

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
        res_nilpredictor = requests.post(nilpredictor, json=not_yet_nil['nil_features'].values.tolist())
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

    nil_threshold = 0.5
    # if below threshold --> is NIL
    data['is_nil'] = data['nil_score'].apply(lambda x: x < nil_threshold)

    ## fix uncertain threshold hitl
    th_low = 0.4
    th_high = 0.6

    if hitl:
        print('Simulating HITL...')
        to_correct = data.query('nil_score <= 0.6 and nil_score >= 0.4')
        to_correct_nil = to_correct.query('NIL')
        to_correct_not_nil = to_correct.query('~NIL')

        data.loc[to_correct_nil.index, 'is_nil'] = True
        data.loc[to_correct_not_nil.index, 'is_nil'] = False
        data.loc[to_correct_not_nil.index, 'candidates'] = to_correct_not_nil.apply(get_new_cand, axis=1)

    print('Estimated {} entities as NOT NIL'.format(data.eval('is_nil == False').sum()))
    print('Estimated {} entities as NIL'.format(data.eval('is_nil == True').sum()))

    data['top_title'] = data['candidates'].apply(lambda x: x[0]['title'])

    # necessary for evaluation
    prev_added_entities = added_entities.copy()

    added_entities = pd.concat([added_entities, pd.DataFrame(data.query('is_nil')['wikiId'].unique(), columns=['wikiId'])]).drop_duplicates()
    added_entities.set_index('wikiId', drop=False, inplace=True)

    # ## Entity Clustering
    print('clustering')
    nil_mentions = data.query('is_nil == True')

    res_nilcluster = requests.post(nilcluster, json={
            'ids': nil_mentions.index.tolist(),
            'mentions': nil_mentions[mention].values.tolist(),
            'encodings': nil_mentions['encoding'].values.tolist()
        })

    if not res_nilcluster.ok:
        print('NIL cluster ERROR')
    else:
        print('OK')

    clusters = pd.DataFrame(res_nilcluster.json())

    clusters = clusters.sort_values(by='nelements', ascending=False)

    if not no_add and not add_correct:
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
        new_indexed = requests.post(indexer_add, json=data_new.to_dict(orient='records'))

        if not new_indexed.ok:
            print('error adding new entities')
        else:
            new_indexed = new_indexed.json()
            clusters['index_id'] = new_indexed['ids']
            clusters['index_indexer'] = new_indexed['indexer']
            prev_clusters = prev_clusters.append(clusters[['index_id', 'mode', 'modes']], ignore_index=True)
            prev_clusters.set_index('index_id', drop=False, inplace=True)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        batch_basename = os.path.splitext(os.path.basename(batch))[0]
        outdata = os.path.join(save_path, '{}_outdata.pickle'.format(batch_basename))
        data.to_pickle(outdata)
        outclusters = os.path.join(save_path, '{}_outclusters.pickle'.format(batch_basename))
        clusters.to_pickle(outclusters)

    if add_correct:
        # correct clusters
        print('correct clusters')
        ## get gold NILs, correctly cluster them and then add
        # title nelements mentions_id mentions center

        correct_clusters = pd.DataFrame(columns=['title', 'nelements', 'mentions_id', 'mentions', 'center', 'original_url', 'original_id'])
        for k,v in data.query('NIL').groupby('wikiId').groups.items():
            df_mentions = data.iloc[v]
            df_mentions['embedding'] = df_mentions['encoding'].apply(vector_decode)
            c = Cluster()
            for i, row in df_mentions.iterrows():
                c.add_element(mention=row[mention], entity='entity',
                                                encodings=row['embedding'], mentions_id=i)
            title = c.get_title()
            center = c.get_center()
            center = vector_encode(center)
            correct_clusters = correct_clusters.append({
                'original_url': k,
                'original_id': int(k.split('=')[1]),
                'mentions_id': v.tolist(),
                'nelements': len(v),
                'mentions': df_mentions[mention].tolist(),
                'title': title,
                'center': center
            }, ignore_index=True)

            # populate with new entities
            print('Populating rw index with correct entities')

            data_new = correct_clusters[['title', 'center', 'original_id']].rename(columns={'center': 'encoding', 'original_id': 'wikipedia_id'})
            new_indexed = requests.post(indexer_add, json=data_new.to_dict(orient='records'))

            if not new_indexed.ok:
                print('error adding new entities')
            else:
                new_indexed = new_indexed.json()
                correct_clusters['index_id'] = new_indexed['ids']
                correct_clusters['index_indexer'] = new_indexed['indexer']

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                batch_basename = os.path.splitext(os.path.basename(batch))[0]
                outcorrect_clusters = os.path.join(save_path, '{}_outcorrect_clusters.pickle'.format(batch_basename))
                correct_clusters.to_pickle(outcorrect_clusters)

    # Evaluation
    data['top_wikiId'] = data.apply(lambda x: x['candidates'][0]['wikipedia_id'], axis=1)
    # TODO use also top_title?
    report = {}
    report['batch'] = batch
    ## Linking
    ### Consider also previously added entities ?
    report['no_correct_links'] = data.eval('top_wikiId == wikiId').sum()
    report['no_correct_links_normalized'] = report['no_correct_links'] / data.shape[0]
    # no nil
    report['no_correct_links_no_nil'] = data.eval('~NIL and top_wikiId == wikiId').sum()
    report['no_correct_links_no_nil_normalized'] = report['no_correct_links_no_nil'] / data.eval('~NIL').sum()
    # considers nil
    report['no_correct_links_overall'] = data.eval('( NIL and is_nil ) or ( top_wikiId == wikiId and ~is_nil )').sum()
    report['no_correct_links_overall_normalized'] = report['no_correct_links_overall'] / data.shape[0]
    data_not_nil = data.query('~NIL')
    if data_not_nil.shape[0] > 0:
        # how many are correct among the not NIL ones (gold)
        report['no_correct_links_not_nil'] = data_not_nil.eval('top_wikiId == wikiId').sum()
        report['no_correct_links_not_nil_normalized'] = report['no_correct_links_not_nil'] / data_not_nil.shape[0]
    else:
        report['no_correct_links_not_nil'] = -1
        report['no_correct_links_not_nil_normalized'] = -1
    # TODO consider clusters without a mode ?? # actually if the cluster has a single mode we add it and we can evaluate
    ## NIL prediction (consider added entities)
    ### NIL mentions referring to previously added entity should be not NIL now
    ### --> previoulsy identified NIL entities are expected to be linked
    newly_added = data.join(prev_added_entities, how='inner', on='wikiId', rsuffix='_y')
    report['no_correct_links_newly_added'] = newly_added.eval('top_wikiId == wikiId').sum()
    report['no_correct_links_newly_added_normalized'] = report['no_correct_links_newly_added'] / newly_added.shape[0]
    #### consider the ones linked to a cluster with no id (with multiple modes)
    multi_modes = newly_added[newly_added['top_wikiId'].isna()]
    if multi_modes.shape[0] > 0:
        multi_modes['top_linking_id'] = multi_modes.apply(lambda x: x['candidates'][0]['id'], axis=1)
        multi_modes = multi_modes.join(prev_clusters, how='left', on='top_linking_id')
        report['no_approximately_corrected_links'] = multi_modes.apply(lambda x: x['wikiId'] in x['modes']).sum()
        report['no_approximately_corrected_links_normalized'] = report['no_approximately_corrected_links'] / multi_modes.shape[0]
    else:
        report['no_approximately_corrected_links'] = -1
        report['no_approximately_corrected_links_normalized'] = -1

    ## NIL prediction
    report['nil_tp'] = data.eval('NIL and is_nil').sum()
    report['nil_fp'] = data.eval('~NIL and is_nil').sum()
    report['nil_tn'] = data.eval('~NIL and ~is_nil').sum()
    report['nil_fn'] = data.eval('NIL and ~is_nil').sum()
    report['nil_precision'] = report['nil_tp'] / (report['nil_tp'] + report['nil_fp'])
    report['nil_recall'] = report['nil_tp'] / (report['nil_tp'] + report['nil_fn'])
    report['nil_f1'] = 2 * report['nil_precision'] * report['nil_recall'] / (report['nil_precision'] + report['nil_recall'])

    ### NIL prediction mitigated
    ### consider correct also when linking is not correct but is_nil
    # NIL or wrong linking --> true positive
    report['nil_mitigated_tp'] = data.eval('( NIL or top_wikiId != wikiId ) and is_nil').sum()
    # not nil and correct linking --> false positive
    report['nil_mitigated_fp'] = data.eval('~NIL and top_wikiId == wikiId and is_nil').sum()
    report['nil_mitigated_tn'] = data.eval('~NIL and top_wikiId == wikiId and ~is_nil').sum()
    report['nil_mitigated_fn'] = data.eval('( NIL or top_wikiId != wikiId ) and ~is_nil').sum()
    report['nil_mitigated_precision'] = report['nil_mitigated_tp'] / (report['nil_mitigated_tp'] + report['nil_mitigated_fp'])
    report['nil_mitigated_recall'] = report['nil_mitigated_tp'] / (report['nil_mitigated_tp'] + report['nil_mitigated_fn'])
    report['nil_mitigated_f1'] = 2 * report['nil_mitigated_precision'] * report['nil_mitigated_recall'] / (report['nil_mitigated_precision'] + report['nil_mitigated_recall'])

    ## NIL clustering

    ### print output
    pprint(report)

@click.command()
@click.option('--add-correct', default=False, help='Populate the KB with gold entities.')
@click.option('--hitl', default=False, help='Simulate the HITL which corrects linking and NIL prediction results when the scores are uncertain.')
@click.option('--no-add', default=False, help='Do not add new entities to the KB.')
# @click.option('--cross', default=False, help='Use also the crossencoder (implies --no-add).')
@click.option('--save-path', default=None, type=str, help='Folder in which to save data.')
@click.option('--reset', default=True, help='Reset the RW index before starting.')
@click.argument('batches', nargs=-1)
def main(add_correct, hitl, no_add, save_path, reset, batches):
    for batch in tqdm(batches):
        run_batch(batch, add_correct, hitl, no_add, save_path, reset)

if __name__ == '__main__':
    main()