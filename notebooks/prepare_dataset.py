import pandas as pd
import json
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
import os
import statistics

import click

bi_higher_is_better = True

# TODO consider removing top_n to create easier augmented instances
def _bi_get_stats(x, remove_correct = False):
    global bi_higher_is_better
    assert len(x.scores) == len(x.nns)
    scores = x.scores.copy()
    correct = None
    if x.labels in x.nns:
        # found correct entity
        i_correct = x.nns.index(x.labels)
        if remove_correct:
            del scores[i_correct]
        else:
            correct = scores[i_correct]

    _stats = {
        "correct": correct,
        "max": max(scores),
        "second": sorted(scores, reverse=bi_higher_is_better)[1],
        "min": min(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "stdev": statistics.stdev(scores)
    }
    return _stats

def _cross_get_stats(x, remove_correct=False):
    assert len(x.unsorted_scores) == len(x.nns)
    scores = x.unsorted_scores.copy()
    correct = None
    if x.labels in x.nns:
        # found correct entity
        i_correct = x.nns.index(x.labels)
        if remove_correct:
            del scores[i_correct]
        else:
            correct = scores[i_correct]

    _stats = {
        "correct": correct,
        "max": max(scores),
        "second": sorted(scores, reverse=True)[1],
        "min": min(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "stdev": statistics.stdev(scores),
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

    return bi_stats, cross_stats, combined_stats, bi_df, cross_df

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

    return bi_stats_nil, cross_stats_nil, combined_stats_nil, bi_df_nil, cross_df_nil

def _bi_cross_errors(combined_stats):
    global bi_higher_is_better
    print('bi errors', combined_stats.query('correct_bi {}_bi'.format('< max' if bi_higher_is_better else '> min')).count()[0])
    print('cross errors', combined_stats.query('correct_cross < max_cross').count()[0])
    print('cross >> bi', combined_stats.query('correct_bi {}_bi and correct_cross >= max_cross'.format('< max' if bi_higher_is_better else '> min')).count()[0])
    print('bi >> cross', combined_stats.query('correct_cross < max_cross and correct_bi {}_bi'.format('>= max' if bi_higher_is_better else '<= min')).count()[0])
    assert all(combined_stats['correct_bi'].isna() == combined_stats['correct_cross'].isna())

def _create_dataset(combined_stats, combined_stats_nil = None):
    global bi_higher_is_better
    print('bi_h', bi_higher_is_better)
    combined_positives = combined_stats[combined_stats['correct_bi'].notna()]\
        .query('correct_bi {}_bi and correct_cross >= max_cross'.format('>= max' if bi_higher_is_better else '<= min'))
    combined_positives['y'] = [1] * combined_positives.shape[0]

    # the ones failed or "hard positives"
    combined_hard_positives = combined_stats[combined_stats['correct_bi'].notna()]\
        .query('correct_bi {}_bi and correct_cross < max_cross'.format('< max' if bi_higher_is_better else '> min'))
    combined_hard_positives['y'] = [1] * combined_hard_positives.shape[0]

    # the ones failed but negatives
    combined_hard_negatives = combined_stats[combined_stats['correct_bi'].notna()]\
        .query('correct_bi {}_bi and correct_cross < max_cross'.format('< max' if bi_higher_is_better else '> min'))
    combined_hard_negatives['y'] = [0] * combined_hard_negatives.shape[0]

    # neg not found by models
    combined_nf_negatives = combined_stats[combined_stats['correct_bi'].isna()].copy()
    combined_nf_negatives['y'] = [0] * combined_nf_negatives.shape[0]

    # nil negatives
    if combined_stats_nil is not None:
        combined_nil_negatives = combined_stats_nil.copy()
        combined_nil_negatives['y'] = [0] * combined_nil_negatives.shape[0]
        combined_nf_negatives = pd.concat([combined_nf_negatives, combined_nil_negatives])

    combined_dataset = pd.concat([combined_positives, combined_nf_negatives])

    train, test, _ = _train_test(combined_dataset)
    print('dataset shape (train-test):', train.shape, test.shape)
    print('dataset y distr (train):')
    print(pd.DataFrame(train['y'].value_counts()).to_markdown())
    print('dataset y distr (test):')
    print(pd.DataFrame(test['y'].value_counts()).to_markdown())

    combined_dataset_hard = pd.concat([combined_hard_positives, combined_hard_negatives])

    train_hard_only, test_hard_only, _ = _train_test(combined_dataset_hard)

    train_hard = pd.concat([train, train_hard_only])
    test_hard = pd.concat([test, test_hard_only])

    print('dataset hard shape (train-test):', train_hard.shape, test_hard.shape)
    print('dataset hard y distr (train):')
    print(pd.DataFrame(train_hard['y'].value_counts()).to_markdown())
    print('dataset hard y distr (test):')
    print(pd.DataFrame(test_hard['y'].value_counts()).to_markdown())

    assert (train['y'] == 1).sum() > 0
    assert (test['y'] == 1).sum() > 0
    assert (train_hard['y'] == 1).sum() > 0
    assert (train_hard['y'] == 1).sum() > 0

    assert (train['y'] == 0).sum() > 0
    assert (test['y'] == 0).sum() > 0
    assert (train_hard['y'] == 0).sum() > 0
    assert (train_hard['y'] == 0).sum() > 0

    return train, test, train_hard, test_hard

def _train_test(dataset, create_validation=False):
    train, test = train_test_split(dataset, test_size=0.33, random_state=42)
    validation = None
    if create_validation:
        train, validation = train_test_split(train, test_size=0.2, random_state=42)
    return train, test, validation

def _augment(dataset, combined_stats, bi_df, cross_df):
    n_augment = (lambda x: abs(x[0]-x[1]))(dataset['y'].value_counts())

    _temp_stats = combined_stats.iloc[dataset['idx']]
    pos_to_sample = _temp_stats[_temp_stats['correct_bi'].notna()].query('correct_bi {}_bi and correct_cross >= max_cross'.format('>= max' if bi_higher_is_better else '<= min'))

    aug_negatives_idx = sample_without_replacement(pos_to_sample.shape[0], n_augment, random_state=42)

    bi2aug = bi_df.iloc[aug_negatives_idx]
    cross2aug = cross_df.iloc[aug_negatives_idx]

    bi_aug_stats = bi2aug.apply(lambda x: _bi_get_stats(x, remove_correct=True), axis=1, result_type='expand')
    cross_aug_stats = cross2aug.apply(lambda x: _cross_get_stats(x, remove_correct=True), axis=1, result_type='expand')

    combined_aug_stats = bi_aug_stats.copy()
    combined_aug_stats.columns = [c+'_bi' for c in combined_aug_stats.columns]
    combined_aug_stats[[c+'_cross' for c in cross_aug_stats.columns]] = cross_aug_stats

    combined_aug_stats['idx'] = combined_aug_stats.index

    combined_aug_stats['y'] = [0] * combined_aug_stats.shape[0]
    print('augment shape', combined_aug_stats.shape)

    assert combined_aug_stats['correct_bi'].notna().sum() == 0
    assert combined_aug_stats['correct_cross'].notna().sum() == 0

    dataset_aug = pd.concat([dataset, combined_aug_stats])
    print('augmented shape', dataset_aug.shape)
    print('augmented distr:')
    print(pd.DataFrame(dataset_aug['y'].value_counts()).to_markdown())

    return dataset_aug

def _load_datasets(path):
    datasets = {}
    for d in ['train', 'test', 'train_hard', 'test_hard', 'train_aug', 'train_hard_aug', 'test_aug', 'test_hard_aug']:
        datasets[d] = pd.read_csv(os.path.join(path, d))
    return datasets


@click.command()
@click.option('--score-path', required=True, type=str, help='path in which to find score files')
@click.option('--nil-path', required=False, default=None, type=str, help='path in which to find nil score files')
@click.option('--out-path', required=True, type=str, help='output path in which to save dataset')
@click.option('--bi-lower-is-better', required=False, default=False, is_flag=True, help='whether the correct bi score should be lower than uncorrect ones. e.g. when the score is the NN distance.')
def main(score_path, nil_path, out_path, bi_lower_is_better):

    global bi_higher_is_better
    bi_higher_is_better = not bi_lower_is_better

    _,_,combined_stats, bi_df, cross_df = _load_scores(score_path)
    print('score shape', combined_stats.shape)

    if nil_path is not None:
        _,_,combined_stats_nil,_,_ = _load_nil(nil_path)
        print('nil shape', combined_stats_nil.shape)
    else:
        combined_stats_nil = None

    _bi_cross_errors(combined_stats)

    train, test, train_hard, test_hard = _create_dataset(combined_stats, combined_stats_nil)

    print('augment train')
    train_aug = _augment(train, combined_stats, bi_df, cross_df)
    print('augment train_hard')
    train_hard_aug = _augment(train_hard, combined_stats, bi_df, cross_df)

    print('augment test')
    test_aug = _augment(test, combined_stats, bi_df, cross_df)
    print('augment train_hard')
    test_hard_aug = _augment(test_hard, combined_stats, bi_df, cross_df)

    assert train['max_bi'].isna().sum() == 0
    assert train['max_cross'].isna().sum() == 0
    assert train_hard['max_bi'].isna().sum() == 0
    assert train_hard['max_cross'].isna().sum() == 0
    assert test['max_bi'].isna().sum() == 0
    assert test['max_cross'].isna().sum() == 0
    assert test_hard['max_bi'].isna().sum() == 0
    assert test_hard['max_cross'].isna().sum() == 0
    assert train['y'].isna().sum() == 0
    assert train_hard['y'].isna().sum() == 0
    assert test['y'].isna().sum() == 0
    assert test_hard['y'].isna().sum() == 0
    assert train_aug['max_bi'].isna().sum() == 0
    assert train_aug['max_cross'].isna().sum() == 0
    assert train_hard_aug['max_bi'].isna().sum() == 0
    assert train_hard_aug['max_cross'].isna().sum() == 0
    assert test_aug['max_bi'].isna().sum() == 0
    assert test_aug['max_cross'].isna().sum() == 0
    assert test_hard_aug['max_bi'].isna().sum() == 0
    assert test_hard_aug['max_cross'].isna().sum() == 0
    assert test_aug['y'].isna().sum() == 0
    assert test_hard_aug['y'].isna().sum() == 0

    # shuffle
    train = train.sample(frac=1, random_state=42)
    test = test.sample(frac=1, random_state=42)
    train_hard = train_hard.sample(frac=1, random_state=42)
    test_hard = test_hard.sample(frac=1, random_state=42)
    train_aug = train_aug.sample(frac=1, random_state=42)
    train_hard_aug = train_hard_aug.sample(frac=1, random_state=42)
    test_aug = test_aug.sample(frac=1, random_state=42)
    test_hard_aug = test_hard_aug.sample(frac=1, random_state=42)


    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    train.to_csv(os.path.join(out_path, 'train'), index = False)
    test.to_csv(os.path.join(out_path, 'test'), index = False)
    train_hard.to_csv(os.path.join(out_path, 'train_hard'), index = False)
    test_hard.to_csv(os.path.join(out_path, 'test_hard'), index = False)
    train_aug.to_csv(os.path.join(out_path, 'train_aug'), index = False)
    train_hard_aug.to_csv(os.path.join(out_path, 'train_hard_aug'), index = False)
    test_aug.to_csv(os.path.join(out_path, 'test_aug'), index = False)
    test_hard_aug.to_csv(os.path.join(out_path, 'test_hard_aug'), index = False)

if __name__ == '__main__':
    main()