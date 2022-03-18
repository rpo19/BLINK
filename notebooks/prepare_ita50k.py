# %%
import pandas as pd
import requests
from tqdm import trange,tqdm
import textdistance
# %%
#train_path = '../data/ita/train50k.jsonl.gz'
train_path = '../data/ita/test.jsonl.gz'
outpath = 'nil_pred_ita_dataset/test.pickle'
# %%
train_df = pd.read_json(train_path, lines=True, compression='gzip')
# %%
batchsize=1000
res_biencoder = []
for batch in trange(0,train_df.shape[0], batchsize):
    train_batch = train_df.iloc[batch: batch + batchsize]
    res_biencoder.append(requests.post('http://127.0.0.1/ita/api/blink/biencoder/mention',
        json=train_batch[['context_left', 'mention', 'context_right']].to_dict(orient='records')))
# %%
assert all(res for res in res_biencoder)

# %%
x = 0
for batch in trange(0,train_df.shape[0], batchsize):
    train_df.loc[batch: batch + batchsize-1, 'encoding'] = \
        res_biencoder[x].json()['encodings']
    x += 1
# %%
assert train_df['encoding'].isna().sum() == 0
# %%
batchsize=200
res_indexer = []
for batch in trange(0,train_df.shape[0], batchsize):
    train_batch = train_df.iloc[batch: batch + batchsize]
    body = {
        'encodings': train_batch['encoding'].values.tolist(),
        'top_k': 10
    }
    res_indexer.append(requests.post('http://127.0.0.1/ita/api/indexer/search',
        json=body))

# %%
assert all(res for res in res_indexer)
# %%
candidates = []
for res_batch in tqdm(res_indexer):
    candidates.extend(res_batch.json())
train_df['candidates'] = candidates
# %%
train_df['candidate_title'] = train_df['candidates'].apply(lambda x: x[0]['title'])
# %%
train_df['candidate_score'] = train_df['candidates'].apply(lambda x: x[0]['score'])

# %%
levenshtein = textdistance.Levenshtein(qval=None)
jaccard = textdistance.Jaccard(qval=None)
# %%
train_df['levenshtein'] = train_df.apply(lambda x: levenshtein.normalized_similarity(x['mention'].lower(), x['candidate_title'].lower()), axis=1)
# %%
train_df['jaccard'] = train_df.apply(lambda x: jaccard.normalized_similarity(x['mention'].lower(), x['candidate_title'].lower()), axis=1)

# %%
train_df.to_pickle(outpath)
# %%

# %%

# %%
# forgot y
# moved datasets

# %%
train_df = pd.read_pickle('~/git/BLINK/data/ita/nil/train.pickle')
valid_df = pd.read_pickle('~/git/BLINK/data/ita/nil/valid.pickle')
test_df = pd.read_pickle('~/git/BLINK/data/ita/nil/test.pickle')
# %%
train_df['candidate_id'] = train_df['candidates'].apply(lambda x: x[0]['wikipedia_id'])
# %%
valid_df['candidate_id'] = valid_df['candidates'].apply(lambda x: x[0]['wikipedia_id'])

# %%
test_df['candidate_id'] = test_df['candidates'].apply(lambda x: x[0]['wikipedia_id'])

# %%
train_df['y'] = train_df.eval('label_id == candidate_id').astype(int)
# %%
valid_df['y'] = valid_df.eval('label_id == candidate_id').astype(int)

# %%
test_df['y'] = test_df.eval('label_id == candidate_id').astype(int)

# %%
train_df.to_pickle('~/git/BLINK/data/ita/nil/train.pickle')
# %%
valid_df.to_pickle('~/git/BLINK/data/ita/nil/valid.pickle')

# %%
test_df.to_pickle('~/git/BLINK/data/ita/nil/test.pickle')
# %%
