import pandas as pd
import json
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
import os
import statistics

import click

def _bi_get_stats(x):
    assert len(x.scores) == len(x.nns)
    uncorrect = x.scores.copy()
    if x.labels in x.nns:
        # found correct entity
        i_correct = x.nns.index(x.labels)
        correct = x.scores[i_correct]
        del uncorrect[i_correct]
    else:
        # not found correct entity
        correct = None
    _stats = {
        "correct": correct,
        "max": max(uncorrect),
        "min": min(uncorrect),
        "mean": statistics.mean(uncorrect),
        "median": statistics.median(uncorrect),
        "stdev": statistics.stdev(uncorrect),
    }
    return _stats

def _cross_get_stats(x):
    assert len(x.unsorted_scores) == len(x.nns)
    uncorrect = x.unsorted_scores.copy()
    if x.labels in x.nns:
        # found correct entity
        i_correct = x.nns.index(x.labels)
        correct = x.unsorted_scores[i_correct]
        del uncorrect[i_correct]
    else:
        # not found correct entity
        correct = None
    _stats = {
        "correct": correct,
        "max": max(uncorrect),
        "min": min(uncorrect),
        "mean": statistics.mean(uncorrect),
        "median": statistics.median(uncorrect),
        "stdev": statistics.stdev(uncorrect),
    }
    return _stats

def _load_scores(score_path):
    bi_scores = sorted(glob(os.path.join(score_path, '*_bi.jsonl')))
    cross_scores = sorted(glob(os.path.join(score_path, '*_cross.jsonl')))

    bi_df = pd.DataFrame()
    for score_file in bi_scores:
        bi_df = pd.concat([bi_df, pd.read_json(score_file)])

    assert (bi_df['labels'].apply(lambda x: len(x)) != 1).sum() == 0
    bi_df['labels'] = bi_df['labels'].apply(lambda x: x[0])

    bi_stats = bi_df.apply(_bi_get_stats, axis=1, result_type='expand')

    cross_df = pd.DataFrame()
    for score_file in cross_scores:
        cross_df = pd.concat([cross_df, pd.read_json(score_file)])

    assert (cross_df['labels'].apply(lambda x: len(x)) != 1).sum() == 0
    cross_df['labels'] = cross_df['labels'].apply(lambda x: x[0])

    cross_stats = cross_df.apply(_cross_get_stats, axis=1, result_type='expand')

    assert all(bi_df['labels'] == cross_df['labels'])

    combined_stats = bi_stats.copy()
    combined_stats.columns = [c+'_bi' for c in combined_stats.columns]
    combined_stats[[c+'_cross' for c in cross_stats.columns]] = cross_stats

    combined_stats['idx'] = combined_stats.index

    return bi_stats, cross_stats, combined_stats

def _load_nil(nil_path):
    bi_scores_nil = sorted(glob(os.path.join(nil_path, '*_bi.jsonl')))
    cross_scores_nil = sorted(glob(os.path.join(nil_path, '*_cross.jsonl')))

    bi_df_nil = pd.DataFrame()
    for score_file in bi_scores_nil:
        bi_df_nil = pd.concat([bi_df_nil, pd.read_json(score_file)])
        
    assert (bi_df_nil['labels'].apply(lambda x: len(x)) != 1).sum() == 0
    bi_df_nil['labels'] = bi_df_nil['labels'].apply(lambda x: x[0])

    bi_stats_nil = bi_df_nil.apply(_bi_get_stats, axis=1, result_type='expand')

    assert bi_stats_nil['correct'].notna().sum() == 0

    cross_df_nil = pd.DataFrame()
    for score_file in cross_scores_nil:
        cross_df_nil = pd.concat([cross_df_nil, pd.read_json(score_file)])
        
    assert (cross_df_nil['labels'].apply(lambda x: len(x)) != 1).sum() == 0
    cross_df_nil['labels'] = cross_df_nil['labels'].apply(lambda x: x[0])

    cross_stats_nil = cross_df_nil.apply(_cross_get_stats, axis=1, result_type='expand')

    combined_stats_nil = bi_stats_nil.copy()
    combined_stats_nil.columns = [c+'_bi' for c in combined_stats_nil.columns]
    combined_stats_nil[[c+'_cross' for c in cross_stats_nil.columns]] = cross_stats_nil

    combined_stats_nil['idx'] = combined_stats_nil.index

    return bi_stats_nil, cross_stats_nil, combined_stats_nil

def _bi_cross_errors(combined_stats):
    print('bi errors', combined_stats.query('correct_bi < max_bi').count()[0])
    print('cross errors', combined_stats.query('correct_cross < max_cross').count()[0])
    print('cross >> bi', combined_stats.query('correct_bi < max_bi and correct_cross > max_cross').count()[0])
    print('bi >> cross', combined_stats.query('correct_cross < max_cross and correct_bi > max_bi').count()[0])
    assert all(combined_stats['correct_bi'].isna() == combined_stats['correct_cross'].isna())

def _create_dataset(combined_stats, combined_stats_nil):
    combined_positives = combined_stats[combined_stats['correct_bi'].notna()].query('correct_bi > max_bi and correct_cross > max_cross')
    combined_positives = combined_positives[['correct_bi', 'correct_cross', 'idx']]
    combined_positives.columns = ['x_bi', 'x_cross', 'idx']
    combined_positives['y'] = [1] * combined_positives.shape[0]

    _correct_eq_max = combined_stats[combined_stats['correct_bi'].notna()].query('correct_bi == max_bi').count()[0]
    if _correct_eq_max > 0:
        print('bi: correct == max. why?', _correct_eq_max)

    _correct_eq_max = combined_stats[combined_stats['correct_cross'].notna()].query('correct_cross == max_cross').count()[0]
    if _correct_eq_max > 0:
        print('cross: correct == max. why?', _correct_eq_max)

    # the ones failed or "hard positives"
    combined_hard_positives = combined_stats[combined_stats['correct_bi'].notna()].query('correct_bi < max_bi and correct_cross < max_cross')
    combined_hard_positives = combined_hard_positives[['correct_bi', 'correct_cross', 'idx']]
    combined_hard_positives.columns = ['x_bi', 'x_cross', 'idx']
    combined_hard_positives['y'] = [1] * combined_hard_positives.shape[0]

    # the ones failed but negatives
    combined_hard_negatives = combined_stats[combined_stats['correct_bi'].notna()].query('correct_bi < max_bi and correct_cross < max_cross')
    combined_hard_negatives = combined_hard_negatives[['max_bi', 'max_cross', 'idx']]
    combined_hard_negatives.columns = ['x_bi', 'x_cross', 'idx']
    combined_hard_negatives['y'] = [0] * combined_hard_negatives.shape[0]

    # neg not found by models
    combined_nf_negatives = combined_stats[combined_stats['correct_bi'].isna()]
    combined_nf_negatives = combined_nf_negatives[['max_bi', 'max_cross', 'idx']]
    combined_nf_negatives.columns = ['x_bi', 'x_cross', 'idx']
    combined_nf_negatives['y'] = [0] * combined_nf_negatives.shape[0]

    # nil negatives
    combined_nil_negatives = combined_stats_nil[['max_bi', 'max_cross', 'idx']].copy()
    combined_nil_negatives.columns = ['x_bi', 'x_cross', 'idx']
    combined_nil_negatives['y'] = [0] * combined_nil_negatives.shape[0]

    combined_dataset = pd.concat([combined_positives, combined_nf_negatives, combined_nil_negatives])

    train, test, _ = _train_test(combined_dataset)
    print('dataset shape (train-test):', train.shape, test.shape)
    print('dataset y distr (train-test)\n:', train['y'].value_counts(), test['y'].value_counts())

    combined_dataset_hard = pd.concat([combined_hard_positives, combined_hard_negatives])

    train_hard_only, test_hard_only, _ = _train_test(combined_dataset_hard)

    train_hard = pd.concat([train, train_hard_only])
    test_hard = pd.concat([test, test_hard_only])

    print('dataset hard shape (train-test):', train_hard.shape, test_hard.shape)
    print('dataset hard y distr (train-test)\n:', train_hard['y'].value_counts(), test_hard['y'].value_counts())

    return train, test, train_hard, test_hard

def _train_test(dataset, create_validation=False):
    train, test = train_test_split(dataset, test_size=0.33, random_state=42)
    validation = None
    if create_validation:
        train, validation = train_test_split(train, test_size=0.2, random_state=42)
    return train, test, validation

def _augment(dataset, combined_stats):
    n_augment = (lambda x: abs(x[0]-x[1]))(dataset['y'].value_counts())

    _temp_stats = combined_stats.iloc[dataset['idx']]
    pos_to_sample = _temp_stats[_temp_stats['correct_bi'].notna()].query('correct_bi > max_bi and correct_cross > max_cross')

    aug_negatives = pos_to_sample.iloc[sample_without_replacement(pos_to_sample.shape[0], n_augment, random_state=42)]
    aug_negatives = aug_negatives[['max_bi', 'max_cross', 'idx']]
    aug_negatives.columns = ['x_bi', 'x_cross', 'idx']
    aug_negatives['y'] = [0] * aug_negatives.shape[0]
    print('augment shape', aug_negatives.shape)

    dataset_aug = pd.concat([dataset, aug_negatives])
    print('augmented shape', dataset_aug.shape)
    print('augmented distr:\n', dataset_aug['y'].value_counts())

    return dataset_aug

def _load_datasets(path):
    datasets = {}
    for d in ['train', 'test', 'train_hard', 'test_hard', 'train_aug', 'train_hard_aug']:
        datasets[d] = pd.read_csv(os.path.join(path, d))
    return datasets


@click.command()
@click.option('--score-path', required=True, type=str, help='path in which to find score files')
@click.option('--nil-path', required=True, type=str, help='path in which to find nil score files')
@click.option('--out-path', required=True, type=str, help='output path in which to save dataset')
def main(score_path, nil_path, out_path):
    _,_,combined_stats = _load_scores(score_path)
    print('score shape', combined_stats.shape)
    _,_,combined_stats_nil = _load_nil(nil_path)
    print('nil shape', combined_stats_nil.shape)

    _bi_cross_errors(combined_stats)

    train, test, train_hard, test_hard = _create_dataset(combined_stats, combined_stats_nil)

    train_aug = _augment(train, combined_stats)
    train_hard_aug = _augment(train_hard, combined_stats)

    # shuffle
    train = train.sample(frac=1, random_state=42)
    test = test.sample(frac=1, random_state=42)
    train_hard = train_hard.sample(frac=1, random_state=42)
    test_hard = test_hard.sample(frac=1, random_state=42)
    train_aug = train_aug.sample(frac=1, random_state=42)
    train_hard_aug = train_hard_aug.sample(frac=1, random_state=42)

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    train.to_csv(os.path.join(out_path, 'train'), index = False)
    test.to_csv(os.path.join(out_path, 'test'), index = False)
    train_hard.to_csv(os.path.join(out_path, 'train_hard'), index = False)
    test_hard.to_csv(os.path.join(out_path, 'test_hard'), index = False)
    train_aug.to_csv(os.path.join(out_path, 'train_aug'), index = False)
    train_hard_aug.to_csv(os.path.join(out_path, 'train_hard_aug'), index = False)

if __name__ == '__main__':
    main()