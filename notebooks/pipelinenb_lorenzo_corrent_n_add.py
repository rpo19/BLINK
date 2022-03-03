# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import requests
import re
import numpy as np
import pandas as pd
import json
import sys
import os
import base64
from sklearn_extra.cluster import KMedoids

biencoder = 'http://localhost:30300/api/blink/biencoder' # mention # entity
biencoder_mention = f'{biencoder}/mention'
biencoder_entity = f'{biencoder}/entity'
crossencoder = 'http://localhost:30302/api/blink/crossencoder'
indexer = 'http://localhost:30301/api/indexer' # search # add
indexer_search = f'{indexer}/search'
indexer_add = f'{indexer}/add'
nilpredictor = 'http://localhost:30303/api/nilprediction'
nilcluster = 'http://localhost:30305/api/nilcluster'

infile = sys.argv[1]

data = pd.read_json(infile, lines=True)

data = data.rename(columns={'right_context_text': 'context_right', 'left_context_text': 'context_left', 'word': 'mention'})

# ## Entity Linking

# ### Encoding

res_biencoder = requests.post(biencoder_mention, json=data.to_dict(orient='records'))

if res_biencoder.ok:
    data['encoding'] = res_biencoder.json()['encodings']
else:
    print('Biencoder ERROR')
    print(res_biencoder)

print('Encoded {} entities.'.format(data.shape[0]))

data.head()

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

data.head()

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

    features['mention'] = x['mention']
    features['title'] = c[0]['title']

    return is_nil, features


data[['is_nil', 'nil_features']] = data.apply(prepare_for_nil_prediction, axis=1, result_type='expand')

data.head()

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

#data.loc[not_yet_nil.index, 'nil_score_bi'] = nil_scores_bi
#data.loc[not_yet_nil.index, 'nil_score'] = nil_scores_cross

data.loc[not_yet_nil.index, 'nil_score'] = nil_scores_bi
data.loc[not_yet_nil.index, 'nil_score_cross'] = nil_scores_cross

nil_threshold = 0.5
# if below threshold --> is NIL
data['is_nil'] = data['nil_score'].apply(lambda x: x < nil_threshold)

data.head()

print('Estimated {} entities as NOT NIL'.format(data.eval('is_nil == False').sum()))
print('Estimated {} entities as NIL'.format(data.eval('is_nil == True').sum()))

data['top_title'] = data['candidates'].apply(lambda x: x[0]['title'])

# not NIL
data.query('is_nil == False')[['mention', 'top_title']].head()

# ## Entity Clustering
print('clustering')
nil_mentions = data.query('is_nil == True')

res_nilcluster = requests.post(nilcluster, json={
        'ids': nil_mentions.index.tolist(),
        'mentions': nil_mentions['mention'].values.tolist(),
        'encodings': nil_mentions['encoding'].values.tolist()
    })

if not res_nilcluster.ok:
    print('NIL cluster ERROR')
else:
    print('OK')

clusters = pd.DataFrame(res_nilcluster.json())

clusters = clusters.sort_values(by='nelements', ascending=False)

# +
# TODO considero i tipi nel clustering
# -

clusters.head()
#clusters

#clusters['nelements'].plot(kind='hist', bins=20)

print('Found {} clusters out of {} NIL mentions.'.format(clusters.shape[0], nil_mentions.shape[0]))
print('saving')

outdata = './output_test/{}_outdata.pickle'.format(os.path.splitext(os.path.basename(infile))[0])
data.to_pickle(outdata)

outclusters ='./output_test/{}_outclusters.pickle'.format(os.path.splitext(os.path.basename(infile))[0])
clusters.to_pickle(outclusters)

# correct clusters
print('correct clusters')
## get gold NILs, correctly cluster them and then add
# title nelements mentions_id mentions center
def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

correct_clusters = pd.DataFrame(columns=['title', 'nelements', 'mentions_id', 'mentions', 'center', 'original_url', 'original_id'])
for k,v in data.query('NIL').groupby('y_wikiurl_dump').groups.items():
    df_mentions = data.iloc[v]
    title = df_mentions['mention'].value_counts().index[0]
    center = KMedoids(n_clusters=1).fit(np.stack(df_mentions['encoding'].apply(vector_decode).to_numpy())).cluster_centers_
    center = vector_encode(center)
    correct_clusters = correct_clusters.append({
        'original_url': k,
	'original_id': int(k.split('=')[1]),
        'mentions_id': v.tolist(),
        'nelements': len(v),
        'mentions': df_mentions['mention'].tolist(),
        'title': title,
        'center': center
    }, ignore_index=True)

# populate with new entities
print('Populating rw index with new entities')

data_new = correct_clusters[['title', 'center', 'original_id']].rename(columns={'center': 'encoding', 'original_id': 'wikipedia_id'})
new_indexed = requests.post(indexer_add, json=data_new.to_dict(orient='records'))

if not new_indexed.ok:
    print('error adding new entities')
else:
    new_indexed = new_indexed.json()
    correct_clusters['index_id'] = new_indexed['ids']
    correct_clusters['index_indexer'] = new_indexed['indexer']

outcorrect_clusters ='./output_test/{}_outcorrect_clusters.pickle'.format(os.path.splitext(os.path.basename(infile))[0])
correct_clusters.to_pickle(outcorrect_clusters)
