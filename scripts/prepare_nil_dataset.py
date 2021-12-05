import numpy as np
import pandas as pd
import textdistance
import json
import os
import statistics
import pickle

scores_path = './data/scores'
datasets_path = './data/BLINK_benchmark'

dataset_output_path = './data/nil_dataset.pickle'


datasets = [
    ('AIDA-YAGO2_testa_ner', 'AIDA-YAGO2_testa_scores'),
    ('AIDA-YAGO2_testb_ner', 'AIDA-YAGO2_testb_scores'),
    ('AIDA-YAGO2_train_ner', 'AIDA-YAGO2_train_scores'),
    ('ace2004_questions', 'ace2004_questions_scores'),
    ('aquaint_questions', 'aquaint_questions_scores'),
    ('clueweb_questions', 'clueweb_questions_scores'),
    ('msnbc_questions', 'msnbc_questions_scores'),
    ('wnedwiki_questions', 'wnedwiki_questions_scores'),
]


def _eval_line(x, scores_col='scores'):
    assert len(x[scores_col]) == len(x.nns)
    scores = x[scores_col].copy()
    correct = -1
    if x.labels in x.nns:
        # found correct entity
        i_correct = x.nns.index(x.labels)
        # correct is position of the correct entity according to the estimated score
        # correct = 0 means the best scored entity is the correct one
        # correct = -1 means the correct entity is not in the top k
        correct = np.argsort(x[scores_col])[::-1].tolist().index(i_correct)

    return correct


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


def _cross_get_stats(x, remove_correct=False, top_k=100):
    return _bc_get_stats(x, remove_correct=remove_correct, scores_col='unsorted_scores', top_k=top_k)


def _load_scores(bi_scores, cross_scores, basepath=None):
    if basepath is not None:
        bi_scores = os.path.join(basepath, bi_scores)
        cross_scores = os.path.join(basepath, cross_scores)

    bi_df = pd.read_json(bi_scores)

    assert (bi_df['labels'].apply(lambda x: len(x)) != 1).sum() == 0
    bi_df['labels'] = bi_df['labels'].apply(lambda x: x[0])

    cross_df = pd.read_json(cross_scores)

    assert (cross_df['labels'].apply(lambda x: len(x)) != 1).sum() == 0
    cross_df['labels'] = cross_df['labels'].apply(lambda x: x[0])

    assert all(bi_df['labels'] == cross_df['labels'])

    return bi_df, cross_df


def myf(x):
    x['cross_labels_title'] = id2title[x['cross_labels']
                                       ] if x['cross_labels'] != -1 else 'NIL'
    x['cross_best_candidate_title'] = id2title[x['cross_best_candidate']]

    x['bi_labels_title'] = id2title[x['bi_labels']
                                    ] if x['bi_labels'] != -1 else 'NIL'
    x['bi_best_candidate_title'] = id2title[x['bi_best_candidate']]

    return x

def _best_candidate(scores, nns, nil_score=None, nil_threshold=0.5):
    if nil_score is not None and nil_score < nil_threshold:
        # identified as NIL
        return -1
    else:
        return nns[np.argmax(scores)]

# load id2title
id2title_path = './data/id2title.pickle'
print('Loading id2title from {}'.format(id2title_path))
if os.path.isfile(id2title_path):
    with open(id2title_path, 'rb') as fd:
        id2title = pickle.load(fd)
else:
    raise Exception('{} not found! Generate it with `python blink/main_dense.py --save-id2title`.'.format(id2title_path))



whole = pd.DataFrame()  # the entire dataset
for d_data, d_score in datasets:
    df_src = d_data
    d_score = os.path.join(scores_path, d_score)
    d_data = os.path.join(datasets_path, d_data)

    print(f'Processing scores for {d_score}...')
    bi_path = f'{d_score}_bi.jsonl'
    cross_path = f'{d_score}_cross.jsonl'
    bi_df, cross_df = _load_scores(bi_path, cross_path)

    assert bi_df.shape[0] == cross_df.shape[0]

    bi_df['recall@'] = bi_df.apply(lambda x: _eval_line(x, 'scores'), axis=1)
    bi_df['best_candidate'] = bi_df.apply(
        lambda x: _best_candidate(x['scores'], x.nns), axis=1)

    cross_df['recall@'] = cross_df.apply(
        lambda x: _eval_line(x, 'unsorted_scores'), axis=1)
    cross_df['best_candidate'] = cross_df.apply(
        lambda x: _best_candidate(x['unsorted_scores'], x.nns), axis=1)

    d_source = pd.read_json(f'{d_data}.jsonl', lines=True)

    assert d_source.shape[0] == bi_df.shape[0]
    assert d_source.shape[0] == cross_df.shape[0]

    d_source[['bi_'+c for c in bi_df.columns]] = bi_df
    d_source[['cross_'+c for c in cross_df.columns]] = cross_df

    d_source['src'] = df_src
    d_source['src_i'] = d_source.index

    # calculate all the stats we need
    _all_k = list(range(2, 10)) + list(range(10, 105, 5))
    counter = 0
    for k in _all_k:
        counter += 1
        print('\r{}/{}'.format(counter, len(_all_k)), end='')
        _stats_n_bi = d_source.apply(
            lambda x: _bc_get_stats(x,
                                    scores_col='bi_scores',
                                    nns_col='bi_nns',
                                    labels_col='bi_labels',
                                    top_k=k
                                    ),
            axis=1, result_type='expand')
        d_source[[f'bi_stats_{k}_'+c for c in _stats_n_bi.columns]] = _stats_n_bi

        _stats_n_cross = d_source.apply(
            lambda x: _bc_get_stats(x,
                                    scores_col='cross_unsorted_scores',
                                    nns_col='cross_nns',
                                    labels_col='cross_labels',
                                    top_k=k
                                    ),
            axis=1, result_type='expand')
        d_source[[f'cross_stats_{k}_'+c for c in _stats_n_cross.columns]] = _stats_n_cross

    print()

    whole = pd.concat([whole, d_source])

# save original index
whole['i'] = whole.index

print('Getting titles...')
whole = whole.apply(myf, axis=1)

# feature selection
# calc all text distances
# # get the list from devlop-local:train_new.Rmd with qval=None

dist_dict = {
    "Hamming": textdistance.Hamming(qval=None),
    "Mlipns": textdistance.MLIPNS(qval=None),
    "Levenshtein": textdistance.Levenshtein(qval=None),
    "DamerauLevenshtein": textdistance.DamerauLevenshtein(qval=None),
    "JaroWinkler": textdistance.JaroWinkler(qval=None),
    "StrCmp95": textdistance.StrCmp95(),
    "NeedlemanWunsch": textdistance.NeedlemanWunsch(qval=None),
    "Gotoh": textdistance.Gotoh(qval=None),
    "SmithWaterman": textdistance.SmithWaterman(qval=None),
    "Jaccard": textdistance.Jaccard(qval=None),
    "Sorensen": textdistance.Sorensen(qval=None),
    "Tversky": textdistance.Tversky(qval=None),
    "Overlap": textdistance.Overlap(qval=None),
    "Tanimoto": textdistance.Tanimoto(qval=None),
    "Cosine": textdistance.Cosine(qval=None),
    "MongeElkan": textdistance.MongeElkan(qval=None),
    "Bag": textdistance.Bag(qval=None),
    "LCSSeq": textdistance.LCSSeq(qval=None),
    "LCSStr": textdistance.LCSStr(qval=None),
    "Editex": textdistance.Editex()
}

print('Text distances cross...')
crossbi='cross'
for i in range(0, len(dist_dict.keys())):
    diststring = list(dist_dict.keys())[i]
    dist = dist_dict[diststring]
    print('\r{}/{}'.format(i, len(dist_dict.keys())), end='')
    whole[crossbi + '_'+ diststring.lower()] = whole.apply(
        lambda x: dist.normalized_similarity(x['mention'].lower(), x[f'{crossbi}_best_candidate_title'].lower()),
        axis=1
    )

print()
print('Text distances bi...')
crossbi='bi'
for i in range(0, len(dist_dict.keys())):
    diststring = list(dist_dict.keys())[i]
    dist = dist_dict[diststring]
    print('\r{}/{}'.format(i, len(dist_dict.keys())), end='')
    whole[crossbi + '_'+ diststring.lower()] = whole.apply(
        lambda x: dist.normalized_similarity(x['mention'].lower(), x[f'{crossbi}_best_candidate_title'].lower()),
        axis=1
    )
print()

print('One-hot encoding NER types...')
whole['ner_per'] = whole.eval('ner == "PER"').astype(float)
whole['ner_loc'] = whole.eval('ner == "LOC"').astype(float)
whole['ner_org'] = whole.eval('ner == "ORG"').astype(float)
whole['ner_misc'] = whole.eval('ner == "MISC"').astype(float)

print('Calculating target y...')
whole['y_bi'] = whole.eval('bi_labels != -1 and bi_labels == bi_best_candidate').astype(int)
whole['y_cross'] = whole.eval('cross_labels != -1 and cross_labels == cross_best_candidate').astype(int)

print('Saving dataset at {}...'.format(dataset_output_path))
whole.to_pickle(dataset_output_path)
