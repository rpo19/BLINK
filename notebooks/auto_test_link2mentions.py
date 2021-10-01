import sys

sys.path.append('..')
import os
import seaborn as sns
from matplotlib import pyplot as plt
import textdistance
import statistics
import pickle
import numpy as np
from sklearn_extra.cluster import KMedoids
from blink.indexer.faiss_indexer import DenseFlatIndexer
import pandas as pd

overall_df = pd.DataFrame()
overall_df_oracle = pd.DataFrame()


# -

sns.set(style='ticks', palette='Set2')
sns.despine()


def GetMedoid(vX): return KMedoids(
    n_clusters=1).fit(np.stack(vX)).cluster_centers_


def eval_with_nil(df, name, wiki, test, save_to_path):
    print(name, 'on', test, ', wiki:', wiki)
    # accuracy
    overall_correct = df.query('(nil_p >= 0.5 and wikipedia_id == wikipedia_id_link)'
                            ' or (nil_p < 0.5 and nil_gold == 0)').shape[0]/df.shape[0]
    # accuracy + nil corrects nel
    nil_when_link_fails = df.query('(nil_p >= 0.5 and wikipedia_id == wikipedia_id_link)'
                                ' or (nil_p < 0.5 and wikipedia_id != wikipedia_id_link)').shape[0]/df.shape[0]
    # 1 recall
    correctly_identified_as_not_nil = df.query('nil_p >= 0.5 and wikipedia_id == wikipedia_id_link').shape[0] / \
        df.query('wikipedia_id == wikipedia_id_link').shape[0]
    # 1 precision
    _precision1 = df.query('nil_p >= 0.5 and wikipedia_id == wikipedia_id_link').shape[0] / \
        df.query('nil_p >= 0.5').shape[0]
    # 0 recall    
    correctly_identified_as_nil = df.query('nil_p < 0.5 and nil_gold==0').shape[0] / \
        df.query('nil_gold==0').shape[0]
    # 0 precision
    _precision0 = df.query('nil_p < 0.5 and nil_gold==0').shape[0] / \
        df.query('nil_p < 0.5').shape[0]

    eval_df = pd.DataFrame([overall_correct,
                            nil_when_link_fails,
                            correctly_identified_as_not_nil,
                            _precision1,
                            correctly_identified_as_nil,
                            _precision0,
                            ], index=[
        'overall_accuracy',
        'overall_accuracy_and_nil_corrects_nel',
        '1-recall',
        '1-precision',
        '0-recall',
        '0-precision'], columns=[
            f'{name}_test_{test}_index_{wiki}'
        ])

    res_df = eval_df.transpose().copy()

    print(pd.DataFrame(df['nil_gold'].value_counts()).to_markdown())
    print()
    print(pd.DataFrame(np.round(df['nil_p']).value_counts()).to_markdown())
    print()

    print(eval_df.to_markdown())
    print()
    print(eval_df.to_latex())

    with open(save_to_path+f'/{name}_test_{test}_index_{wiki}_report.txt', 'w') as fd:
        print(name, 'on', test, ', wiki:', wiki, file=fd)
        print(pd.DataFrame(df['nil_gold'].value_counts()
                        ).to_markdown(), file=fd)
        print("", file=fd)
        print(pd.DataFrame(
            np.round(df['nil_p']).value_counts()).to_markdown(), file=fd)
        print("", file=fd)

        print(eval_df.to_markdown(), file=fd)
        print("", file=fd)
        print(eval_df.to_latex(), file=fd)

    overall_correct = df.query('(nil_p >= 0.75 and wikipedia_id == wikipedia_id_link)'
                            ' or (nil_p < 0.25 and nil_gold == 0)').shape[0]/df.query(
        'nil_p >= 0.75 or nil_p <= 0.25').shape[0]
    nil_when_link_fails = df.query('(nil_p >= 0.75 and wikipedia_id == wikipedia_id_link)'
                                ' or (nil_p < 0.25 and wikipedia_id != wikipedia_id_link)').shape[0]/df.query(
        'nil_p >= 0.75 or nil_p <= 0.25').shape[0]
    correctly_identified_as_not_nil = df.query('nil_p >= 0.75 and wikipedia_id == wikipedia_id_link').shape[0] / \
        df.query('nil_p >= 0.75').shape[0]
    try:
        correctly_identified_as_nil = df.query('nil_p < 0.25 and nil_gold==0').shape[0] / \
            df.query('nil_p < 0.25').shape[0]
    except:
        correctly_identified_as_nil = 0

    eval_df = pd.DataFrame([overall_correct,
                            nil_when_link_fails,
                            correctly_identified_as_not_nil,
                            correctly_identified_as_nil,
                            ], index=[
        'overall_accuracy_oracle',
        'overall_accuracy_and_nil_corrects_nel_oracle',
        '1-precision',
        '0-precision', ],
        columns = [
            f'{name}_test_{test}_index_{wiki}_oracle'
        ])


    res_df_oracle = eval_df.transpose().copy()

    print('-'*10, 'ORACLE', '-'*10)
    print(eval_df.to_markdown())
    print()
    print(eval_df.to_latex())

    with open(save_to_path+f'/{name}_test_{test}_index_{wiki}_oracle_report.txt', 'w') as fd:
        print(name, 'on', test, ', oracle, wiki:', wiki, file=fd)
        print('-'*10, 'ORACLE', '-'*10, file=fd)
        print(eval_df.to_markdown(), file=fd)
        print("", file=fd)
        print(eval_df.to_latex(), file=fd)

    plt.figure()
    pd.DataFrame(pd.DataFrame({
        'pred': df['nil_p'],
        'pred_round': np.round(df['nil_p']),
        'gold': df['nil_gold']
    })).plot(kind='kde')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(save_to_path, name+f'_test_{test}_index_{wiki}_kde.png'))
    plt.close()

    plt.figure()
    correct = df[df['nil_p'].round() == df['nil_gold']].rename(
        columns={
            'nil_p': 'correct'
        })[['correct']]
    ax = correct.plot(kind='kde', legend=True)
    plt.tight_layout(pad=0.5)

    errors = df[df['nil_p'].round() != df['nil_gold']].rename(
        columns={
            'nil_p': 'errors'
        })[['errors']]
    errors.plot(kind='kde', legend=True, ax=ax)
    plt.tight_layout(pad=0.5)
    # plt.savefig(os.path.join(outpath, title+'_kde_errors.png'))
    plt.savefig(os.path.join(save_to_path, name+f'_test_{test}_index_{wiki}_kde_correct_errors.png'))
    plt.close()

    plt.close('all')

    return res_df, res_df_oracle


train_df = pd.read_json(
    '../output/AIDA-YAGO2_train_encodings.jsonl', lines=True)
testa_df = pd.read_json(
    '../output/AIDA-YAGO2_testa_encodings.jsonl', lines=True)
testb_df = pd.read_json(
    '../output/AIDA-YAGO2_testb_encodings.jsonl', lines=True)

# # with NIL

train_ner = pd.read_json('AIDA-YAGO2_train_ner.jsonl', lines=True)

testa_ner = pd.read_json('AIDA-YAGO2_testa_ner.jsonl', lines=True)

testb_ner = pd.read_json('AIDA-YAGO2_testb_ner.jsonl', lines=True)

onehot_t = train_ner.apply(lambda x: {
    'ner_per': x['ner'] == 'PER',
    'ner_loc': x['ner'] == 'LOC',
    'ner_org': x['ner'] == 'ORG',
    'ner_misc': x['ner'] == 'MISC'
}, axis=1, result_type='expand')

train_df[onehot_t.columns] = onehot_t.astype(int)

train_df['mention'] = train_ner['mention']

train_df.columns

# medoid
to_index = pd.DataFrame(train_df.query('wikipedia_id > 0').groupby('wikipedia_id')['encoding'].apply(
    lambda x: GetMedoid(x)[0]))

# index all
#to_index = pd.DataFrame(train_df.query('wikipedia_id > 0')[['wikipedia_id', 'encoding']])
#to_index = to_index.set_index('wikipedia_id')

to_index = to_index.join(pd.DataFrame(train_df.query('wikipedia_id > 0').groupby('wikipedia_id')[['ner_per', 'ner_loc',
                                                                                                  'ner_org', 'ner_misc']].mean()))

to_index = to_index.join(pd.DataFrame(train_df.query('wikipedia_id > 0').groupby('wikipedia_id')['mention'].apply(
    lambda x: x.value_counts().index[0])))

to_index = to_index.join(pd.DataFrame(train_df.query(
    'wikipedia_id > 0').groupby('wikipedia_id')['label'].first()))

to_index

to_index = to_index.sample(frac=1)  # shuffle
to_index['index'] = range(to_index.shape[0])
to_index['wikipedia_id'] = to_index.index
to_index = to_index.set_index('index')
to_index

######

# ### types stuff

wiki_types = pd.read_csv('aida_ner_type_v2.csv', index_col=0)

wiki_types = wiki_types.rename(columns={
    'per': 'wiki_per',
    'loc': 'wiki_loc',
    'org': 'wiki_org',
    'misc': 'wiki_misc'
})

wiki_types[['wiki_per', 'wiki_loc', 'wiki_org', 'wiki_misc']] = wiki_types[[
    'wiki_per', 'wiki_loc', 'wiki_org', 'wiki_misc']].astype(int)

wiki_types = wiki_types.set_index('id')

# medoid wiki
to_index_wiki = pd.DataFrame(train_df.query('wikipedia_id > 0').groupby('wikipedia_id')['encoding'].apply(
    lambda x: GetMedoid(x)[0]))

to_index_wiki = to_index_wiki.join(pd.DataFrame(train_df.query('wikipedia_id > 0').groupby('wikipedia_id')[['ner_per', 'ner_loc',
                                                                                                            'ner_org', 'ner_misc']].mean()))

to_index_wiki = to_index_wiki.join(pd.DataFrame(train_df.query('wikipedia_id > 0').groupby('wikipedia_id')['mention'].apply(
    lambda x: x.value_counts().index[0])))

to_index_wiki = to_index_wiki.join(pd.DataFrame(train_df.query(
    'wikipedia_id > 0').groupby('wikipedia_id')['label'].first()))


to_index_wiki['wikipedia_id'] = to_index_wiki.index

to_index_wiki = to_index_wiki.set_index('label')

to_index_wiki


to_index_wiki = to_index_wiki.join(wiki_types)

to_index_wiki = to_index_wiki[to_index_wiki['wiki_per'].notna()]

to_index_wiki['index'] = range(to_index_wiki.shape[0])
to_index_wiki = to_index_wiki.set_index('index')

to_index_wiki

# ### end types stuff

# +
#to_index = to_index_wiki

# 1024 dimensions, 50000 default as BLINK
index_1 = DenseFlatIndexer(1024, 50000)
index_1.index.ntotal

# +
# index in batch of 100 mentions
for i in range(100, to_index.shape[0], 100):
    #print(i-100, i)
    index_1.index_data(
        np.stack(
            to_index.iloc[i-100:i]['encoding'].values
        ).astype('float32'))


# index last batch
index_1.index_data(
    np.stack(
        to_index.iloc[i:to_index.shape[0]]['encoding'].values
    ).astype('float32'))
# -

assert index_1.index.ntotal == to_index.shape[0]

print('index 2', index_1.index.ntotal)

####

# 1024 dimensions, 50000 default as BLINK
index_2 = DenseFlatIndexer(1024, 50000)
index_2.index.ntotal

# +
# index in batch of 100 mentions
for i in range(100, to_index_wiki.shape[0], 100):
    #print(i-100, i)
    index_2.index_data(
        np.stack(
            to_index_wiki.iloc[i-100:i]['encoding'].values
        ).astype('float32'))


# index last batch
index_2.index_data(
    np.stack(
        to_index_wiki.iloc[i:to_index_wiki.shape[0]]['encoding'].values
    ).astype('float32'))
# -

assert index_2.index.ntotal == to_index_wiki.shape[0]
print('index 2', index_2.index.ntotal)

# +
onehot_testa = testa_ner.apply(lambda x: {
    'ner_per': x['ner'] == 'PER',
    'ner_loc': x['ner'] == 'LOC',
    'ner_org': x['ner'] == 'ORG',
    'ner_misc': x['ner'] == 'MISC'
}, axis=1, result_type='expand')

onehot_testb = testb_ner.apply(lambda x: {
    'ner_per': x['ner'] == 'PER',
    'ner_loc': x['ner'] == 'LOC',
    'ner_org': x['ner'] == 'ORG',
    'ner_misc': x['ner'] == 'MISC'
}, axis=1, result_type='expand')
# -

testa_df[onehot_testa.columns] = onehot_testa.astype(int)
testb_df[onehot_testa.columns] = onehot_testb.astype(int)

testa_df['mention'] = testa_ner['mention']
testb_df['mention'] = testb_ner['mention']

testa_df.columns

testb_df.columns

testa_linking_results = index_1.search_knn(
    np.stack(testa_df['encoding'].values).astype('float32'), 100)

# +
#del testa_linking_results
#del testa_linking_results_wiki_id
# -

testb_linking_results = index_1.search_knn(
    np.stack(testb_df['encoding'].values).astype('float32'), 100)

myfun = np.vectorize(lambda x: to_index.iloc[x]['wikipedia_id'])

testa_linking_results_wiki_id = myfun(testa_linking_results[1])

testb_linking_results_wiki_id = myfun(testb_linking_results[1])

# index wiki

testa_linking_results_wiki = index_2.search_knn(
    np.stack(testa_df['encoding'].values).astype('float32'), 100)

testb_linking_results_wiki = index_2.search_knn(
    np.stack(testb_df['encoding'].values).astype('float32'), 100)

myfun_wiki = np.vectorize(lambda x: to_index_wiki.iloc[x]['wikipedia_id'])

testa_linking_results_wiki_id_wiki = myfun_wiki(testa_linking_results_wiki[1])

testb_linking_results_wiki_id_wiki = myfun_wiki(testb_linking_results_wiki[1])

print('nel done...')

# ls ../models/nil_pred_new/bi

# +
# load nil predictor

all_features_map = {
    'aida_under_bi_max': [
        'max',
    ],
    'aida_under_bi_max_stdev4': [
        'max',
        'stdev4',
    ],
    'aida_under_bi_max_jaccard': [
        'max',
        'jaccard',
    ],
    'aida_under_bi_max_jaccard_ner_avg': [
        'max',
        'jaccard',
        'ner_per',
        'ner_loc',
        'ner_org',
        'ner_misc',
        'cand_ner_per',
        'cand_ner_loc',
        'cand_ner_org',
        'cand_ner_misc',
    ],
    'aida_under_bi_max_ner_wiki_jaccard': [
        'max',
        'jaccard',
        'ner_per',
        'ner_loc',
        'ner_org',
        'ner_misc',
        'wiki_per',
        'wiki_loc',
        'wiki_org',
        'wiki_misc',
    ],
    'aida_under_bi_max_ner': [
        'max',
        'ner_per',
        'ner_loc',
        'ner_org',
        'ner_misc',
        'cand_ner_per',
        'cand_ner_loc',
        'cand_ner_org',
        'cand_ner_misc',
    ],
    'aida_under_bi_max_ner_wiki': [
        'max',
        'ner_per',
        'ner_loc',
        'ner_org',
        'ner_misc',
        'wiki_per',
        'wiki_loc',
        'wiki_org',
        'wiki_misc',
    ],
    'aida_under_bi_max_hamming': [
        'max',
        'hamming',
    ],
    'aida_under_bi_max_ner_wiki_hamming': [
        'max',
        'hamming',
        'ner_per',
        'ner_loc',
        'ner_org',
        'ner_misc',
        'wiki_per',
        'wiki_loc',
        'wiki_org',
        'wiki_misc',
    ],
    'aida_under_bi_max_ner_avg_hamming': [
        'max',
        'hamming',
        'ner_per',
        'ner_loc',
        'ner_org',
        'ner_misc',
        'cand_ner_per',
        'cand_ner_loc',
        'cand_ner_org',
        'cand_ner_misc',
    ],
    'aida_under_bi_max_stdev4_hamming': [
        'max',
        'stdev4',
        'hamming',
    ],
    'aida_under_bi_max_ner_wiki_stdev4': [
        'max',
        'stdev4',
        'ner_per',
        'ner_loc',
        'ner_org',
        'ner_misc',
        'wiki_per',
        'wiki_loc',
        'wiki_org',
        'wiki_misc',
    ],
    'aida_under_bi_max_ner_avg_stdev4': [
        'max',
        'stdev4',
        'ner_per',
        'ner_loc',
        'ner_org',
        'ner_misc',
        'cand_ner_per',
        'cand_ner_loc',
        'cand_ner_org',
        'cand_ner_misc',
    ],

}


# #model_name = 'aida_under_bi_max'
# #model_name = 'aida_under_bi_max_stdev4'
# #model_name = 'aida_under_bi_max_jaccard'
# #model_name = 'aida_under_bi_max_jaccard_ner_avg'
# model_name = 'aida_under_bi_max_ner_wiki_jaccard'
# #model_name = 'aida_under_bi_max_ner'
# #model_name = 'aida_under_bi_max_ner_wiki'
# #model_name = 'aida_under_bi_max_hamming'
# #model_name = 'aida_under_bi_max_ner_wiki_hamming'
# #model_name = 'aida_under_bi_max_ner_avg_hamming'
# #model_name = 'aida_under_bi_max_stdev4_hamming'
# #model_name = 'aida_under_bi_max_ner_wiki_stdev4'
# #model_name = 'aida_under_bi_max_ner_avg_stdev4'

if len(sys.argv) >=2:
    only = sys.argv[1].split(',')
    all_features_map = {k:v for k,v in all_features_map.items() if k in only}
    print('Only:', list(all_features_map.keys()))

for model_name, nil_features in all_features_map.items():
    index_type = 'wikitp' if 'wiki_per' in nil_features else 'normal'
    print(model_name, 'on', index_type)
    with open(f'../models/nil_pred_new/bi/{model_name}_model.pickle', 'rb') as fd:
        nil_clf = pickle.load(fd)
# -

# +
# calc features
# max
# stdev4
# hamming

# test a



    save_to_path = 'linking2mentions_test_auto'

    try:

        for test in ['testa', 'testb']:
            testa = test == 'testa'
            if testa:
                test_df = testa_df
            else:
                test_df = testb_df

            if 'wiki_per' in nil_features:
                # use wiki index 
                _to_index = to_index_wiki

                if test == 'testa':
                    _linking_results = testa_linking_results_wiki
                    _linking_results_wiki_id = testa_linking_results_wiki_id_wiki
                elif test == 'testb':
                    _linking_results = testb_linking_results_wiki
                    _linking_results_wiki_id = testb_linking_results_wiki_id_wiki

            else:
                _to_index = to_index

                if test == 'testa':
                    _linking_results = testa_linking_results
                    _linking_results_wiki_id = testa_linking_results_wiki_id
                elif test == 'testb':
                    _linking_results = testb_linking_results
                    _linking_results_wiki_id = testb_linking_results_wiki_id


            if 'wiki_per' in nil_features:
                X_nil = pd.DataFrame({
                    'max': [i[0] for i in _linking_results[0]],
                    'stdev4': [statistics.stdev(i[:4].tolist()) for i in _linking_results[0]],
                    'mention': [_to_index.iloc[i[0]]['mention'] for i in _linking_results[1]],
                    'cand_ner_per': [_to_index.iloc[i[0]]['ner_per'] for i in _linking_results[1]],
                    'cand_ner_loc': [_to_index.iloc[i[0]]['ner_loc'] for i in _linking_results[1]],
                    'cand_ner_org': [_to_index.iloc[i[0]]['ner_org'] for i in _linking_results[1]],
                    'cand_ner_misc': [_to_index.iloc[i[0]]['ner_misc'] for i in _linking_results[1]],
                    'wiki_per': [_to_index.iloc[i[0]]['wiki_per'] for i in _linking_results[1]],
                    'wiki_loc': [_to_index.iloc[i[0]]['wiki_loc'] for i in _linking_results[1]],
                    'wiki_org': [_to_index.iloc[i[0]]['wiki_org'] for i in _linking_results[1]],
                    'wiki_misc': [_to_index.iloc[i[0]]['wiki_misc'] for i in _linking_results[1]],
                })
            else:
                X_nil = pd.DataFrame({
                    'max': [i[0] for i in _linking_results[0]],
                    'stdev4': [statistics.stdev(i[:4].tolist()) for i in _linking_results[0]],
                    'mention': [_to_index.iloc[i[0]]['mention'] for i in _linking_results[1]],
                    'cand_ner_per': [_to_index.iloc[i[0]]['ner_per'] for i in _linking_results[1]],
                    'cand_ner_loc': [_to_index.iloc[i[0]]['ner_loc'] for i in _linking_results[1]],
                    'cand_ner_org': [_to_index.iloc[i[0]]['ner_org'] for i in _linking_results[1]],
                    'cand_ner_misc': [_to_index.iloc[i[0]]['ner_misc'] for i in _linking_results[1]],
                })

            X_nil['mention_to_link'] = test_df['mention']

            hamming_i = textdistance.Hamming(qval=None)
            X_nil['hamming'] = X_nil.apply(lambda x: hamming_i.normalized_similarity(
                x['mention_to_link'], x['mention']), axis=1)

            jaccard_i = textdistance.Jaccard(qval=None)
            X_nil['jaccard'] = X_nil.apply(lambda x: jaccard_i.normalized_similarity(
                x['mention_to_link'], x['mention']), axis=1)

            X_nil[['ner_per', 'ner_loc', 'ner_org', 'ner_misc']
                ] = test_df[['ner_per', 'ner_loc', 'ner_org', 'ner_misc']]

            y_nil = np.array(
                [i for _, i in nil_clf.predict_proba(X_nil[nil_features].values)])


            eval_test_nil = pd.DataFrame()
            eval_test_nil['wikipedia_id'] = test_df['wikipedia_id']
            eval_test_nil['wikipedia_id_link'] = [i[0]
                                                for i in _linking_results_wiki_id]
            eval_test_nil['nil_gold'] = eval_test_nil['wikipedia_id'].isin(
                _to_index['wikipedia_id']).astype(int)
            eval_test_nil['nil_p'] = y_nil

            is_wiki = 'wikitp' if 'wiki_per' in nil_features else 'normal'

            res_df, res_df_oracle = eval_with_nil(eval_test_nil, model_name, is_wiki, test, save_to_path)

            overall_df = pd.concat([overall_df, res_df])
            overall_df_oracle = pd.concat([overall_df_oracle, res_df_oracle])

    except:
        overall_df.to_csv(save_to_path+'/overall_df.csv')
        overall_df_oracle.to_csv(save_to_path+'/overall_df_oracle.csv')
        raise
    
    overall_df.to_csv(save_to_path+'/overall_df.csv')
    overall_df_oracle.to_csv(save_to_path+'/overall_df_oracle.csv')
