import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

import pickle
from sklearn.metrics import classification_report, roc_curve

import os

import matplotlib.pyplot as plt
import seaborn as sns

import sys

only = None
if len(sys.argv) >= 2:
    only = sys.argv[1].split(',')
    print('Only', only)

sns.set(style='ticks', palette='Set2')
sns.despine()

def myplot(y, y_pred, y_pred_round, title, outpath):
    plt.figure()
    df = pd.DataFrame({
        'y': y,
        'y_pred': y_pred,
        'y_pred_rnd': y_pred_round
    })
    # kde
    df[['y', 'y_pred']].plot(kind='kde', title=title+'_kde')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(outpath, title+'_kde.png'))
    plt.close()

    # roc curve
    fpr, tpr, thresholds = roc_curve(df['y'], df['y_pred'])
    plt.figure()
    pd.DataFrame({
        'tpr': tpr,
        'fpr': fpr,
        'thresholds': thresholds
    }).plot(x='fpr', y='tpr', title=title+'_roc')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(outpath, title+'_roc.png'))
    plt.close()

    # correct
    correct = df.query('y == y_pred_rnd')['y_pred'].rename('correct')
    plt.figure()
    correct.plot(kind='kde', title=title+'_kde', legend=True)
    plt.tight_layout(pad=0.5)
    # plt.savefig(os.path.join(outpath, title+'_kde_correct.png'))
    # plt.close()

    # errors
    errors = df.query('y != y_pred_rnd')['y_pred'].rename('errors')
    # plt.figure()
    # errors.plot(kind='density', title=title+'errors kde')
    errors.plot(kind='density', title=title+'kde', legend=True)
    plt.tight_layout(pad=0.5)
    # plt.savefig(os.path.join(outpath, title+'_kde_errors.png'))
    plt.savefig(os.path.join(outpath, title+'_kde_correct_errors.png'))
    plt.close()

    plt.close('all')



outpath = 'train_new_out'
if not os.path.isdir(outpath):
    os.mkdir(outpath)

print('loading dataset...')
# dataset = pd.read_csv('whole4wikitypes.csv', index_col=0)
# dataset = pd.read_csv('whole_df5_dists_topk.csv', index_col=0)
dataset = pd.read_csv('whole6.csv', index_col=0)
print('loaded...')

tasks = [
    {
        'name': 'aida_all10',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'no',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'aida_all100',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'no',
        'features':  [
                'cross_stats_100_max',
                'cross_stats_100_mean',
                'cross_stats_100_median',
                'cross_stats_100_stdev',
                'bi_stats_100_max',
                'bi_stats_100_mean',
                'bi_stats_100_median',
                'bi_stats_100_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_max_ner',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_max_jaccard',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_sim_jaccard',
                'bi_sim_jaccard',
            ]
    },
    {
        'name': 'aida_under_all10_max_hamming',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'aida_under_all10_stats',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
            ]
    },
    # {
    #     'name': 'aida_under_all4_stats',
    #     'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
    #     'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
    #     'sampling': 'undersample',
    #     'features':  [
    #             'cross_stats_10_max',
    #             'cross_stats_4_mean',
    #             'cross_stats_4_median',
    #             'cross_stats_4_stdev',
    #             'bi_stats_10_max',
    #             'bi_stats_4_mean',
    #             'bi_stats_4_median',
    #             'bi_stats_4_stdev',
    #         ]
    # },
    {
        'name': 'aida_under_all10_max_stdev',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_stdev',
            ]
    },
    {
        'name': 'aida_under_all10_max',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
            ]
    },
    {
        'name': 'aida_under_cross10_max',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
            ]
    },
    {
        'name': 'aida_under_all10_max_stdev4',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_4_stdev',
                'bi_stats_10_max',
                'bi_stats_4_stdev',
            ]
    },
    {
        'name': 'aida_under_all10_max_stdev4_hamming',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_4_stdev',
                'bi_stats_10_max',
                'bi_stats_4_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'aida_under_cross10',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'cross_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'aida_all10_ner_correct',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'no',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_correct_cross',
                'avg_ner_loc_correct_cross',
                'avg_ner_org_correct_cross',
                'avg_ner_misc_correct_cross',
            ]
    },
    {
        'name': 'aida_under_all10_with_ner_correct',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_correct_cross',
                'avg_ner_loc_correct_cross',
                'avg_ner_org_correct_cross',
                'avg_ner_misc_correct_cross',
            ]
    },
    {
        'name': 'aida_under_all10_max_ner_correct',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_correct_cross',
                'avg_ner_loc_correct_cross',
                'avg_ner_org_correct_cross',
                'avg_ner_misc_correct_cross',
            ]
    },
    {
        'name': 'aida_under_all10_with_ner_wiki',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_max_ner_wiki',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_max_stdev4_ner_wiki',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_stats_4_stdev',
                'bi_stats_4_stdev',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross',
            ]
    },
    {
        'name': 'aida_all10_max_stdev4_ner_wiki',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'no',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_stats_4_stdev',
                'bi_stats_4_stdev',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_max_hamming_ner_wiki',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_max_stdev4_hamming_ner_wiki',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_stats_4_stdev',
                'bi_stats_4_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_max_stats_hamming_ner_wiki',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_stats_ner_wiki',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_ner_wiki_but_dst',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_but_stats',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_ner_wiki_but_stats',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross'
            ]
    },
    {
        'name': 'aida_under_all10_ner_wiki_jaccard_but_stats',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_sim_jaccard',
                'bi_sim_jaccard',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross'
            ]
    },
    {
        'name': 'aida_under_all10_but_dst',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'aida_under_all10_but_ner_avg_ner',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'aida_under_all10_but_avg_ner',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
            ]
    },
    {
        'name': 'aida_under_all10_but_ner',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'aida_under_bi_max',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
            ]
    },
    {
        'name': 'aida_under_bi_max_stdev4',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_4_stdev',
            ]
    },
    {
        'name': 'aida_under_bi_max_stdev4_hamming',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_4_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'aida_under_bi_max_stdev',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_stdev',
            ]
    },
    {
        'name': 'aida_under_bi_max_hamming',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_bi',
                'avg_ner_loc_bi',
                'avg_ner_org_bi',
                'avg_ner_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner_wiki',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_bi',
                'wiki_loc_bi',
                'wiki_org_bi',
                'wiki_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner_wiki_stdev4',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_4_stdev',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_bi',
                'wiki_loc_bi',
                'wiki_org_bi',
                'wiki_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner_avg_stdev4',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_4_stdev',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_bi',
                'avg_ner_loc_bi',
                'avg_ner_org_bi',
                'avg_ner_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner_wiki_stdev4_hamming',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_4_stdev',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_bi',
                'wiki_loc_bi',
                'wiki_org_bi',
                'wiki_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner_wiki_hamming',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_bi',
                'wiki_loc_bi',
                'wiki_org_bi',
                'wiki_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner_avg_hamming',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_bi',
                'avg_ner_loc_bi',
                'avg_ner_org_bi',
                'avg_ner_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner_wiki_jaccard',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_sim_jaccard',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_bi',
                'wiki_loc_bi',
                'wiki_org_bi',
                'wiki_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_jaccard',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_sim_jaccard',
            ]
    },
    {
        'name': 'aida_under_bi_max_jaccard_ner_avg',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_sim_jaccard',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_bi',
                'avg_ner_loc_bi',
                'avg_ner_org_bi',
                'avg_ner_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner_wiki_stats',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_bi',
                'wiki_loc_bi',
                'wiki_org_bi',
                'wiki_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_max_ner_wiki_stats_hamming',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_bi',
                'wiki_loc_bi',
                'wiki_org_bi',
                'wiki_misc_bi',
            ]
    },
    {
        'name': 'aida_under_bi_stats',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
            ]
    },
    {
        'name': 'alldata_all10',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'no',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'alldata_under_all10',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'alldata_under_all10_stats',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
            ]
    },
    {
        'name': 'alldata_under_all10_max_stdev',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_stdev',
            ]
    },
    {
        'name': 'alldata_under_all10_max_stdev4',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_4_stdev',
                'bi_stats_10_max',
                'bi_stats_4_stdev',
            ]
    },
    {
        'name': 'alldata_under_all10_but_stats',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'alldata_under_all10_ner_wiki_but_stats',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_cross',
                'wiki_loc_cross',
                'wiki_org_cross',
                'wiki_misc_cross'
            ]
    },
    {
        'name': 'alldata_under_cross10',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'cross_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_cross',
                'avg_ner_loc_cross',
                'avg_ner_org_cross',
                'avg_ner_misc_cross',
            ]
    },
    {
        'name': 'alldata_under_bi10',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_bi',
                'avg_ner_loc_bi',
                'avg_ner_org_bi',
                'avg_ner_misc_bi',
            ]
    },
    {
        'name': 'alldata_under_bi10_max',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
            ]
    },
    {
        'name': 'alldata_under_bi10_max_dst',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_max_dst_ner',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_bi',
                'avg_ner_loc_bi',
                'avg_ner_org_bi',
                'avg_ner_misc_bi',
            ]
    },
    {
        'name': 'alldata_under_bi10_max_dst_ner_wiki',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'wiki_per_bi',
                'wiki_loc_bi',
                'wiki_org_bi',
                'wiki_misc_bi',
            ]
    },
    {
        'name': 'alldata_under_bi10_max_dst_ner_stats',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': 0.33,
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
                'ner_per',
                'ner_loc',
                'ner_org',
                'ner_misc',
                'avg_ner_per_bi',
                'avg_ner_loc_bi',
                'avg_ner_org_bi',
                'avg_ner_misc_bi',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_tested_on_all',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_stats_tested_on_all',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_tested_on_all',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_stats_tested_on_all',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_tested_on_aida_train',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_stats_tested_on_aida_train',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_tested_on_aida_train',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_stats_tested_on_aida_train',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_tested_on_aida_testa',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_stats_tested_on_aida_testa',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_tested_on_aida_testa',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_stats_tested_on_aida_testa',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_tested_on_aida_testb',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_stats_tested_on_aida_testb',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_tested_on_aida_testb',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_stats_tested_on_aida_testb',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_tested_on_clueweb',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/clueweb_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_stats_tested_on_clueweb',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/clueweb_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_tested_on_clueweb',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/clueweb_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_stats_tested_on_clueweb',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/clueweb_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_tested_on_wnedwiki',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/wnedwiki_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_stats_tested_on_wnedwiki',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/wnedwiki_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_tested_on_wnedwiki',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/wnedwiki_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_stats_tested_on_wnedwiki',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/wnedwiki_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_tested_on_aquaint',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/aquaint_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_stats_tested_on_aquaint',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/aquaint_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_nut_ner_tested_on_aquaint',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/aquaint_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_stats_tested_on_aquaint',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/aquaint_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_tested_on_msnbc',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/msnbc_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_stats_tested_on_msnbc',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/msnbc_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_tested_on_msnbc',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/msnbc_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_stats_tested_on_msnbc',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/msnbc_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_tested_on_ace2004',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/ace2004_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_all10_but_ner_stats_tested_on_ace2004',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/ace2004_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'bi_stats_10_max',
                'cross_hamming',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_tested_on_ace2004',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/ace2004_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_mean',
                'bi_stats_10_median',
                'bi_stats_10_stdev',
                'bi_hamming',
            ]
    },
    {
        'name': 'alldata_under_bi10_but_ner_stats_tested_on_ace2004',
        'train': [
            'dataset_and_preds/AIDA-YAGO2_train.csv',
            'dataset_and_preds/AIDA-YAGO2_testa.csv',
            'dataset_and_preds/AIDA-YAGO2_testb.csv',
            'dataset_and_preds/clueweb_questions.csv',
            'dataset_and_preds/wnedwiki_questions.csv',
            'dataset_and_preds/aquaint_questions.csv',
            'dataset_and_preds/msnbc_questions.csv',
            'dataset_and_preds/ace2004_questions.csv'
        ],
        'test': [
            'dataset_and_preds/ace2004_questions.csv',
        ],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
]

#['max', 'stats', 'dst', 'ner', 'avg_ner', 'avg_ner_correct']

# assert no duplicates
vc = pd.DataFrame([task['name'] for task in tasks]).value_counts()
if not (vc <= 1).all():
    print('!' * 30)
    print('Duplicates:')
    print('!' * 30)
    print(vc[vc > 1])
    raise Exception('duplicate task!')

csv_report = pd.DataFrame()

if only is not None:
    tasks = [t for t in tasks if t['name'] in only]

for task in tasks:
    print('-'*30)
    print(task['name'])

    train_df = dataset[dataset['src'].isin(task['train'])]
    if isinstance(task['test'], list):
        test_df = dataset[dataset['src'].isin(task['test'])]
    elif isinstance(task['test'], float):
        train_df, test_df = train_test_split(train_df, test_size = task['test'], random_state = 1234)
    else:
        raise Exception()

    train_df_shape_original = train_df.shape[0]
    test_df_shape_original = test_df.shape[0]

    train_df = train_df[train_df[task['features']].notna().all(axis=1)]
    test_df = test_df[test_df[task['features']].notna().all(axis=1)]

    train_df_shape_notna = train_df.shape[0]
    test_df_shape_notna = test_df.shape[0]

    if task['sampling'] == 'undersample':
        print('undersampling...')

        train_df_0 = train_df.query('y == 0')
        train_df_1 = train_df.query('y == 1')

        train_df_1 = train_df_1.sample(frac=1).iloc[:train_df_0.shape[0]]
        train_df = pd.concat([train_df_0, train_df_1]).sample(frac=1)

    elif task['sampling'] == 'no':
        pass
    else:
        raise Exception()

    train_df_shape_actual = train_df.shape[0]
    test_df_shape_actual = test_df.shape[0]

    df_size_report = pd.DataFrame({
        'train': [train_df_shape_original, train_df_shape_notna, train_df_shape_actual],
        'test': [test_df_shape_original, test_df_shape_notna, test_df_shape_actual]
    }, index=['original', 'notna', 'actual']).to_markdown()
    print(df_size_report)

    print(pd.DataFrame(train_df['y'].value_counts()).to_markdown())

    X_train = train_df[task['features']].values
    y_train = train_df['y'].values

    X_test = test_df[task['features']].values
    y_test = test_df['y'].values

    # model
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=1234, max_iter=200)
    )

    clf.fit(X_train, y_train)

    y_pred = np.array(list(map(lambda x: x[1], clf.predict_proba(X_test))))
    y_pred_round = np.round(y_pred)

    test_df['y_pred_round'] = y_pred_round
    test_df['y_pred'] = y_pred

    bi_baseline = test_df.query('bi_labels == bi_best_candidate or Wikipedia_title == bi_best_candidate_title').shape[0]
    cross_baseline = test_df.query('cross_labels == cross_best_candidate or Wikipedia_title == cross_best_candidate_title').shape[0]

    bi_acc = test_df.query('(y_pred_round == 1 and (bi_labels == bi_best_candidate or Wikipedia_title == bi_best_candidate_title)) or (bi_labels == -1 and y_pred_round == 0)').shape[0]
    cross_acc = test_df.query('(y_pred_round == 1 and (cross_labels == cross_best_candidate or Wikipedia_title == cross_best_candidate_title)) or (cross_labels == -1 and y_pred_round == 0)').shape[0]

    _classification_report = classification_report(y_test, y_pred_round)

    # oracle corrects in [0.25, 0.75]
    # TODO maybe look for a better way to get them (e.g. correct-error kde intersections ?)
    tl = 0.25
    th = 0.75
    oracle_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_round': y_pred_round
    })
    oracle_original_shape = oracle_df.shape[0]
    oracle_df = oracle_df.query(f'y_pred <= {tl} or y_pred >= {th}')

    _classification_report_oracle = classification_report(oracle_df['y_test'], oracle_df['y_pred_round'])

    test_df_oracle = test_df.query(f'y_pred <= {tl} or y_pred >= {th}')

    bi_acc_oracle = test_df_oracle.query('(y_pred_round == 1 and (bi_labels == bi_best_candidate or Wikipedia_title == bi_best_candidate_title)) or (bi_labels == -1 and y_pred_round == 0)').shape[0]
    cross_acc_oracle = test_df_oracle.query('(y_pred_round == 1 and (cross_labels == cross_best_candidate or Wikipedia_title == cross_best_candidate_title)) or (cross_labels == -1 and y_pred_round == 0)').shape[0]

    csv_report = csv_report.append({
        'name': task['name'],
        'bi_baseline': bi_baseline / test_df_shape_actual,
        'cross_baseline': cross_baseline / test_df_shape_actual,
        'bi_acc': bi_acc / test_df_shape_actual,
        'cross_acc': cross_acc / test_df_shape_actual,
        'bi_acc_adjusted': bi_acc / test_df_shape_original,
        'cross_acc_adjusted': cross_acc / test_df_shape_original,
        '0-f1': f1_score(y_test, y_pred_round, pos_label=0),
        '1-f1': f1_score(y_test, y_pred_round, pos_label=1),
        'oracle_ratio': 1 - (oracle_df.shape[0] / oracle_original_shape),
        'bi_acc_oracle': bi_acc_oracle / test_df_oracle.shape[0],
        'cross_acc_oracle': cross_acc_oracle / test_df_oracle.shape[0],
        '0-f1-oracle': f1_score(oracle_df['y_test'], oracle_df['y_pred_round'], pos_label=0),
        '1-f1-oracle': f1_score(oracle_df['y_test'], oracle_df['y_pred_round'], pos_label=1),
    }, ignore_index=True)

    print(_classification_report)

    print('-- Performances over test set:', task['test'], '--')
    print('Bi baseline:', bi_baseline / test_df_shape_actual)
    print('Cross baseline:', cross_baseline / test_df_shape_actual)
    print('Bi acc:', bi_acc / test_df_shape_actual)
    print('Cross acc:', cross_acc / test_df_shape_actual)
    print('Bi acc adjusted:', bi_acc / test_df_shape_original)
    print('Cross acc adjusted:', cross_acc / test_df_shape_original)

    print(f'-- Oracle HITL evaluation when y_pred in [{tl}, {th}]')
    print('Ratio to human validator:', 1 - (oracle_df.shape[0] / oracle_original_shape))
    print(_classification_report_oracle)

    print('Bi acc oracle:', bi_acc_oracle / test_df_oracle.shape[0])
    print('Cross acc oracle:', cross_acc_oracle / test_df_oracle.shape[0])


    with open(os.path.join(outpath, task['name']+'_report.txt'), 'w') as fd:
        print(pd.DataFrame(train_df['y'].value_counts()).to_markdown(), file=fd)
        print(df_size_report, file=fd)
        print(_classification_report, file=fd)

        print('-- Performances over test set:', task['test'], '--', file=fd)
        print('Bi baseline:', bi_baseline / test_df_shape_actual, file=fd)
        print('Cross baseline:', cross_baseline / test_df_shape_actual, file=fd)
        print('Bi acc:', bi_acc / test_df_shape_actual, file=fd)
        print('Cross acc:', cross_acc / test_df_shape_actual, file=fd)
        print('Bi acc adjusted:', bi_acc / test_df_shape_original, file=fd)
        print('Cross acc adjusted:', cross_acc / test_df_shape_original, file=fd)

        print(f'-- Oracle HITL evaluation when y_pred in [{tl}, {th}]', file=fd)
        print('Ratio to human validator:', oracle_df.shape[0] / oracle_original_shape, file=fd)
        print(_classification_report_oracle, file=fd)
        print('Bi acc oracle:', bi_acc_oracle / test_df_oracle.shape[0], file=fd)
        print('Cross acc oracle:', cross_acc_oracle / test_df_oracle.shape[0], file=fd)



    with open(os.path.join(outpath, task['name']+'_model.pickle'), 'wb') as fd:
        pickle.dump(clf, fd)

    myplot(y_test, y_pred, y_pred_round, task['name'], outpath)

    print('-'*30)

if only is not None:
    csv_report_old = pd.read_csv(os.path.join(outpath, 'train_new_summary.csv'), index_col=0)
    csv_report_old = csv_report_old[~csv_report_old['name'].isin(csv_report['name'].unique())]
    csv_report = pd.concat([csv_report_old, csv_report])

csv_report.to_csv(os.path.join(outpath, 'train_new_summary.csv'))