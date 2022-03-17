# %%
import numpy as np
import pandas as pd
import json
import statistics
import textdistance
import sys
# %%
#filename = 'notebooks/output_test_train_data0/data0_outdata.pickle'
#filename = 'notebooks/output_test_dev_data0/data0_outdata.pickle'
# filename = 'notebooks/output_test/train0_outdata.pickle'
filename = sys.argv[1]
# %%
df = pd.read_pickle(filename)
# %%
df
# %%
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

# %%
df['top_id'] = df['candidates'].apply(lambda x: x[0]['wikipedia_id'])
df['top_title'] = df['candidates'].apply(lambda x: x[0]['title'])
df[['scores', 'nns']] = df.apply(lambda x: {'scores': [i['score'] for i in x['candidates'] if i['wikipedia_id'] > 0], 'nns': [i['wikipedia_id'] for i in x['candidates'] if i['wikipedia_id'] > 0]}, result_type='expand', axis=1)
#df['nns'] = df['candidates'].apply(lambda x: [i['wikipedia_id'] for i in x])
df['labels'] = df.eval('~NIL and wikiId == top_id').astype(int)
# %%
stats = df.apply(_bi_get_stats, axis=1, result_type='expand')
# %%
df[stats.columns] = stats
# %%
levenshtein = textdistance.Levenshtein(qval=None)
jaccard = textdistance.Jaccard(qval=None)
# %%
df['levenshtein'] = df.apply(lambda x: levenshtein.normalized_similarity(x['mention'].lower(), x['top_title'].lower()), axis=1)
# %%
df['jaccard'] = df.apply(lambda x: jaccard.normalized_similarity(x['mention'].lower(), x['top_title'].lower()), axis=1)
# %%
df.to_pickle(filename+'_mod')
# %%
