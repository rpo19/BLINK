from ast import expr_context
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
import bcubed
from sklearn.metrics import classification_report
import pandas as pd
import statistics
import textdistance

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
context_right = 'context_right'
context_left = 'context_left'
mention = 'mention'
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

def _bc_get_stats(x, remove_correct=False, scores_col='scores', nns_col='nns', labels_col='labels', top_k=100):
    scores = x[scores_col]
    nns = x[nns_col]
    if isinstance(scores, str):
        scores = np.array(json.loads(scores))
    if isinstance(nns, str):
        nns = np.array(json.loads(nns))

    assert len(scores) == len(nns)
    scores = scores.copy()

    sort_scores_i = np.argsort(scores)[::-1]
    scores = np.array(scores)
    scores = scores[sort_scores_i][:top_k]

    nns = nns.copy()
    nns = np.array(nns)
    nns = nns[sort_scores_i][:top_k]

    correct = None
    if x[labels_col] in nns:
        # found correct entity
        i_correct = list(nns).index(x[labels_col])
        correct = scores[i_correct]

    _stats = {
        "correct": correct,
        "max": max(scores),
        "second": sorted(scores, reverse=True)[1],
        "min": min(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "stdev": statistics.stdev(scores)
    }
    return _stats


def _bi_get_stats(x, remove_correct=False, top_k=100):
    return _bc_get_stats(x, remove_correct=remove_correct, scores_col='scores', top_k=top_k)

def prepare_for_nil_prediction_train(df):
    df['top_id'] = df['candidates'].apply(lambda x: x[0]['wikipedia_id'])
    df['top_title'] = df['candidates'].apply(lambda x: x[0]['title'])
    df[['scores', 'nns']] = df.apply(lambda x: {'scores': [i['score'] for i in x['candidates'] if i['wikipedia_id'] > 0], 'nns': [i['wikipedia_id'] for i in x['candidates'] if i['wikipedia_id'] > 0]}, result_type='expand', axis=1)
    #df['nns'] = df['candidates'].apply(lambda x: [i['wikipedia_id'] for i in x])
    df['labels'] = df.eval('~NIL and Wikipedia_ID == top_id').astype(int)

    stats = df.apply(_bi_get_stats, axis=1, result_type='expand')
    df[stats.columns] = stats

    levenshtein = textdistance.Levenshtein(qval=None)
    jaccard = textdistance.Jaccard(qval=None)
    df['levenshtein'] = df.apply(lambda x: levenshtein.normalized_similarity(x['mention'].lower(), x['top_title'].lower()), axis=1)
    df['jaccard'] = df.apply(lambda x: jaccard.normalized_similarity(x['mention'].lower(), x['top_title'].lower()), axis=1)

    return df

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

def run_batch(batch, data, add_correct, hitl, no_add, save_path, prepare_for_nil_prediction_train_flag,
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

    # ## Entity Linking

    # ### Encoding
    print('Encoding entities...')
    res_biencoder = requests.post(biencoder_mention,
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

    if prepare_for_nil_prediction_train_flag:
        data = prepare_for_nil_prediction_train(data)

        os.makedirs(save_path, exist_ok=True)
        batch_basename = os.path.splitext(os.path.basename(batch))[0]
        outdata = os.path.join(save_path, '{}_outdata.pickle'.format(batch_basename))
        data.to_pickle(outdata)

        return {}


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

    # TODO consider correcting added_entities?
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
    report['size'] = data.shape[0]
    ## Linking
    def eval_linking_helper(x):
        candidate_ids = [i['wikipedia_id'] for i in x['candidates']]
        try:
            return candidate_ids.index(x['wikiId']) + 1 # starting from recall@1
        except ValueError:
            return -1

    not_nil = data.query('~NIL').copy()

    not_nil['linking_found_at'] = not_nil.apply(eval_linking_helper, axis=1)

    for i in [1,2,3,5,10,30,100]:
        report[f'linking_recall@{i}'] = not_nil.eval(f'linking_found_at > 0 and linking_found_at <= {i}').sum() / not_nil.shape[0]

    ## NIL prediction
    # report['nil_tp'] = data.eval('NIL and is_nil').sum()
    # report['nil_fp'] = data.eval('~NIL and is_nil').sum()
    # report['nil_tn'] = data.eval('~NIL and ~is_nil').sum()
    # report['nil_fn'] = data.eval('NIL and ~is_nil').sum()
    # report['nil_precision'] = report['nil_tp'] / (report['nil_tp'] + report['nil_fp'])
    # report['nil_recall'] = report['nil_tp'] / (report['nil_tp'] + report['nil_fn'])
    # report['nil_f1'] = 2 * report['nil_precision'] * report['nil_recall'] / (report['nil_precision'] + report['nil_recall'])

    # report['not_nil_precision'] = report['nil_tn'] / (report['nil_tn'] + report['nil_fn']) # negatives are not-nil
    # report['not_nil_recall'] = report['nil_tn'] / (report['nil_tn'] + report['nil_fn']) # negatives are not-nil
    # report['not_nil_f1'] = 2 * report['not_nil_precision'] * report['not_nil_recall'] / (report['not_nil_precision'] + report['not_nil_recall'])

    # # TODO not-nil avg macro, weighted,...
    # report['nil_pred_macro_precision'] = (report['nil_precision'] + report['not_nil_precision']) / 2
    # report['nil_pred_macro_recall'] = (report['nil_recall'] + report['not_nil_recall']) / 2
    # report['nil_pred_macro_f1'] = (report['nil_f1'] + report['not_nil_f1']) / 2

    report['nil_prediction'] = classification_report(data['NIL'], data['is_nil'], output_dict=True)

    ### NIL prediction mitigated
    ### consider correct also when linking is not correct but is_nil
    data['NIL_mitigated'] = data.eval('NIL or top_wikiId != wikiId')
    report['nil_prediction_mitigated'] = classification_report(data['NIL_mitigated'], data['is_nil'], output_dict=True)
    # # NIL or wrong linking --> true positive
    # report['nil_mitigated_tp'] = data.eval('( NIL or top_wikiId != wikiId ) and is_nil').sum()
    # # not nil and correct linking --> false positive
    # report['nil_mitigated_fp'] = data.eval('~NIL and top_wikiId == wikiId and is_nil').sum()
    # report['nil_mitigated_tn'] = data.eval('~NIL and top_wikiId == wikiId and ~is_nil').sum()
    # report['nil_mitigated_fn'] = data.eval('( NIL or top_wikiId != wikiId ) and ~is_nil').sum()
    # report['nil_mitigated_precision'] = report['nil_mitigated_tp'] / (report['nil_mitigated_tp'] + report['nil_mitigated_fp'])
    # report['nil_mitigated_recall'] = report['nil_mitigated_tp'] / (report['nil_mitigated_tp'] + report['nil_mitigated_fn'])
    # report['nil_mitigated_f1'] = 2 * report['nil_mitigated_precision'] * report['nil_mitigated_recall'] / (report['nil_mitigated_precision'] + report['nil_mitigated_recall'])

    ## NIL clustering
    exploded_clusters = clusters.explode(['mentions_id', 'mentions'])
    merged = data.merge(exploded_clusters, left_index=True, right_on='mentions_id')

    # https://github.com/hhromic/python-bcubed
    keys = [str(x) for x in exploded_clusters['mentions_id']]
    values = [set([str(x)]) for x in exploded_clusters.index]
    cdict = dict(zip(keys, values))

    keysGold = [str(x) for x in merged['mentions_id']]
    # valuesGold= [set([x]) for x in merged['y_wikiurl_dump']]
    valuesGold= [set([x]) for x in merged['wikiId']]
    ldict = dict(zip(keysGold, valuesGold))
    report['nil_clustering_bcubed_precision'] = bcubed.precision(cdict, ldict)
    report['nil_clustering_bcubed_recall'] = bcubed.recall(cdict, ldict)

    # Overall

    overall_correct = 0
    # not nil are corrected when linked to the correct entity and labeled as not-NIL
    overall_to_link_correct = data.eval('~NIL and ~is_nil and wikiId == top_wikiId').sum()
    report['overall_to_link_correct'] = overall_to_link_correct / (data.eval('~NIL').sum() + sys.float_info.min)
    overall_correct += overall_to_link_correct
    # nil not yet added to the kb are correct if labeled as NIL
    should_be_nil = data[data['NIL'] & ~data['wikiId'].isin(prev_added_entities.index)]
    should_be_nil_correct = should_be_nil.eval('is_nil').sum()
    report['should_be_nil_correct'] = should_be_nil_correct / (should_be_nil.shape[0] + sys.float_info.min)
    overall_correct += should_be_nil_correct
    # nil previously added should be linked to the prev added entity (and not nil)
    should_be_linked_to_prev_added = data[data['NIL'] & data['wikiId'].isin(prev_added_entities.index)]
    should_be_linked_to_prev_added_total = should_be_linked_to_prev_added.shape[0]
    should_be_linked_to_prev_added = should_be_linked_to_prev_added.query('~is_nil').copy()
    ## check if linked to a cluster containing at least half coherent mentions
    ### check if the wikiId matches the majority of the wikiIds in the cluster?
    should_be_linked_to_prev_added_correct = 0
    if should_be_linked_to_prev_added.shape[0] > 0:
        should_be_linked_to_prev_added[['top_candidate_id', 'top_candidate_indexer']] = \
            should_be_linked_to_prev_added.apply(\
                lambda x: {
                    'top_candidate_id': x['candidates'][0]['id'],
                    'top_candidate_indexer': x['candidates'][0]['indexer']
                }, axis=1, result_type = 'expand')

        index_indexer = clusters.iloc[0]['index_indexer']
        assert clusters.eval(f'index_indexer == {index_indexer}').all()

        # filter only the ones with the correct indexer
        should_be_linked_to_prev_added = should_be_linked_to_prev_added.query(f'top_candidate_indexer == {index_indexer}')

        # check they are linked correctly
        should_be_linked_to_prev_added = should_be_linked_to_prev_added.merge(prev_clusters, left_on = 'top_candidate_id', right_index = True)

        if should_be_linked_to_prev_added.shape[0] > 0:
            # the majority of the cluster is correct
            should_be_linked_to_prev_added_correct = should_be_linked_to_prev_added.eval('wikiId == mode').sum()
            # half of the cluster is correct
            def helper_half_correct(row):
                return len(row['modes']) == 2 and row['wikiId'] in row['modes']
            should_be_linked_to_prev_added_correct += \
                should_be_linked_to_prev_added.apply(helper_half_correct, axis=1).sum()
            overall_correct += should_be_linked_to_prev_added_correct

    report['should_be_linked_to_prev_added_correct'] = should_be_linked_to_prev_added_correct / (should_be_linked_to_prev_added_total + sys.float_info.min)

    report['overall_correct'] = overall_correct
    report['overall_accuracy'] = overall_correct / data.shape[0]

    #     ### Consider also previously added entities ?
    # report['no_correct_links'] = data.eval('top_wikiId == wikiId').sum()
    # report['no_correct_links_normalized'] = report['no_correct_links'] / data.shape[0]
    # # no nil
    # report['no_correct_links_no_nil'] = data.eval('~NIL and top_wikiId == wikiId').sum()
    # report['no_correct_links_no_nil_normalized'] = report['no_correct_links_no_nil'] / data.eval('~NIL').sum()
    # # considers nil
    # report['no_correct_links_overall'] = data.eval('( NIL and is_nil ) or ( top_wikiId == wikiId and ~is_nil )').sum()
    # report['no_correct_links_overall_normalized'] = report['no_correct_links_overall'] / data.shape[0]
    # data_not_nil = data.query('~NIL')
    # if data_not_nil.shape[0] > 0:
    #     # how many are correct among the not NIL ones (gold)
    #     report['no_correct_links_not_nil'] = data_not_nil.eval('top_wikiId == wikiId').sum()
    #     report['no_correct_links_not_nil_normalized'] = report['no_correct_links_not_nil'] / data_not_nil.shape[0]
    # else:
    #     report['no_correct_links_not_nil'] = -1
    #     report['no_correct_links_not_nil_normalized'] = -1
    # # TODO consider clusters without a mode ?? # actually if the cluster has a single mode we add it and we can evaluate
    # # TODO recall@k

    # ## NIL prediction (consider added entities)
    # ### NIL mentions referring to previously added entity should be not NIL now
    # ### --> previoulsy identified NIL entities are expected to be linked
    # newly_added = data.join(prev_added_entities, how='inner', on='wikiId', rsuffix='_y')
    # report['no_correct_links_newly_added'] = newly_added.eval('top_wikiId == wikiId').sum()
    # report['no_correct_links_newly_added_normalized'] = report['no_correct_links_newly_added'] / newly_added.shape[0]
    # #### consider the ones linked to a cluster with no id (with multiple modes)
    # multi_modes = newly_added[newly_added['top_wikiId'].isna()]
    # if multi_modes.shape[0] > 0:
    #     multi_modes['top_linking_id'] = multi_modes.apply(lambda x: x['candidates'][0]['id'], axis=1)
    #     multi_modes = multi_modes.join(prev_clusters, how='left', on='top_linking_id')
    #     report['no_approximately_corrected_links'] = multi_modes.apply(lambda x: x['wikiId'] in x['modes']).sum()
    #     report['no_approximately_corrected_links_normalized'] = report['no_approximately_corrected_links'] / multi_modes.shape[0]
    # else:
    #     report['no_approximately_corrected_links'] = -1
    #     report['no_approximately_corrected_links_normalized'] = -1

    ### print output
    pprint(report)

    return report

def explode_nil(row, column='nil_prediction', label=''):
    x = row[column]
    res = {}
    for k in x['True'].keys():
        res[f'NIL-{label}-{k}'] = x['True'][k]
    for k in x['False'].keys():
        res[f'not-NIL-{label}-{k}'] = x['False'][k]
    return res

@click.command()
@click.option('--add-correct', is_flag=True, default=False, help='Populate the KB with gold entities.')
@click.option('--hitl', is_flag=True, default=False, help='Simulate the HITL which corrects linking and NIL prediction results when the scores are uncertain.')
@click.option('--no-add', is_flag=True, default=False, help='Do not add new entities to the KB.')
# @click.option('--cross', is_flag=True, default=False, help='Use also the crossencoder (implies --no-add).')
@click.option('--save-path', default=None, type=str, help='Folder in which to save data.')
@click.option('--no-reset', is_flag=True, default=False, help='Reset the RW index before starting.')
@click.option('--report', default=None, help='File in which to write the report in JSON.')
@click.option('--no-incremental', is_flag=True, default=False, help='Run the evaluation merging the batches')
@click.option('--prepare-for-nil-pred', is_flag=True, default=False, help='Prepare data for training NIL prediction. Combine with --save-path.')
@click.argument('batches', nargs=-1, required=True)
def main(add_correct, hitl, no_add, save_path, no_reset, report, batches, no_incremental, prepare_for_nil_pred):
    print('Batches', batches)
    outreports = []

    reset = not no_reset

    if prepare_for_nil_pred and (not save_path or report is not None):
        print('--prepare-for-nil-prediction requires --save-path and no --report')
        sys.exit(1)

    # check batch files exist
    for batch in batches:
        assert os.path.isfile(batch)

    # reset kbp
    if reset:
        print('Resetting RW index...')
        res_reset = requests.post(indexer_reset, data={})

        if res_reset.ok:
            print('Reset done.')
        else:
            print('ERROR while resetting!')
            sys.exit(1)

    if no_incremental:
        print('*** NO INCREMENTAL ***')
        print('Loading and combining batches')
        datas = list(map(lambda x: pd.read_json(x, lines=True), batches))
        data = pd.concat(datas, ignore_index=True)
        outreport = run_batch("no_incremental", data, add_correct, hitl, no_add, save_path, prepare_for_nil_pred)
        outreports.append(outreport)
    else:
        for batch in tqdm(batches):
            print('Loading batch...', batch)
            data = pd.read_json(batch, lines=True)
            outreport = run_batch(batch, data, add_correct, hitl, no_add, save_path, prepare_for_nil_pred)
            outreports.append(outreport)

    if report:
        report_df = pd.DataFrame(outreports)
        temp_df = report_df.apply(lambda x: explode_nil(x, 'nil_prediction'), result_type='expand', axis=1)
        report_df[temp_df.columns] = temp_df
        temp_df = report_df.apply(lambda x: explode_nil(x, 'nil_prediction_mitigated', 'mitigated'), result_type='expand', axis=1)
        report_df[temp_df.columns] = temp_df
        report_df = report_df.drop(columns=['nil_prediction', 'nil_prediction_mitigated'])
        if not no_incremental:
            incremental_overall = report_df.mean(numeric_only=True)
            incremental_overall['batch'] = 'incremental_overall'
            incremental_overall['overall_correct'] = report_df['overall_correct'].sum()
            incremental_overall['size'] = report_df['size'].sum()
            incremental_overall['overall_accuracy'] = incremental_overall['overall_correct'] / incremental_overall['size']
            incremental_overall

            report_df = report_df.append(incremental_overall, ignore_index=True)

        report_df.to_csv(report)

if __name__ == '__main__':
    main()
