import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import sklearn

import os

import matplotlib.pyplot as plt
import seaborn as sns
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
    correct = df.query('y == y_pred_rnd')['y_pred']
    plt.figure()
    correct.plot(kind='kde', title=title+'correct kde')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(outpath, title+'_kde_correct.png'))
    plt.close()

    # errors
    errors = df.query('y != y_pred_rnd')['y_pred']
    plt.figure()
    errors.plot(kind='density', title=title+'errors kde')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(outpath, title+'_kde_errors.png'))
    plt.close()

    plt.close('all')



outpath = 'train_new_out'
if not os.path.isdir(outpath):
    os.mkdir(outpath)

print('loading dataset...')
dataset = pd.read_csv('whole3_ner_avg_also_correct2.csv', index_col=0)
print('loaded...')

tasks = [
    {
        'name': 'aida_all10',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'no',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'cross_stats_100_min',
                'cross_stats_100_mean',
                'cross_stats_100_median',
                'cross_stats_100_stdev',
                'bi_stats_100_max',
                'bi_stats_100_min',
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
        'name': 'aida_under_all10',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
        'name': 'aida_under_cross10',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
        'name': 'aida_under_all10_ner_correct',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
        'name': 'aida_under_all10_but_dst',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'cross_stats_10_max',
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
        'name': 'aida_under_bi_max_dst',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_hamming',
            ]
    },
    {
        'name': 'aida_under_bi_max_dst_ner',
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
        'name': 'aida_under_bi_dst_ner_stats',
        'train': ['dataset_and_preds/AIDA-YAGO2_train.csv'],
        'test': ['dataset_and_preds/AIDA-YAGO2_testa.csv', 'dataset_and_preds/AIDA-YAGO2_testb.csv'],
        'sampling': 'undersample',
        'features':  [
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'cross_stats_10_min',
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
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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
                'cross_stats_10_min',
                'cross_stats_10_mean',
                'cross_stats_10_median',
                'cross_stats_10_stdev',
                'bi_stats_10_max',
                'bi_stats_10_min',
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
                'bi_stats_10_min',
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

    print(_classification_report)

    print('-- Performances over test set:', task['test'], '--')
    print('Bi baseline:', bi_baseline / test_df_shape_actual)
    print('Cross baseline:', cross_baseline / test_df_shape_actual)
    print('Bi acc:', bi_acc / test_df_shape_actual)
    print('Cross acc:', cross_acc / test_df_shape_actual)
    print('Bi acc adjusted:', bi_acc / test_df_shape_original)
    print('Cross acc adjusted:', cross_acc / test_df_shape_original)

    print(f'-- Oracle HITL evaluation when y_pred in [{tl}, {th}]')
    print('Ratio to human validator:', oracle_df.shape[0] / oracle_original_shape)
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
        print('Ratio to human validator:', oracle_df / oracle_original_shape, file=fd)
        print(_classification_report_oracle, file=fd)
        print('Bi acc oracle:', bi_acc_oracle / test_df_oracle.shape[0], file=fd)
        print('Cross acc oracle:', cross_acc_oracle / test_df_oracle.shape[0], file=fd)



    with open(os.path.join(outpath, task['name']+'_model.pickle'), 'wb') as fd:
        pickle.dump(clf, fd)

    myplot(y_test, y_pred, y_pred_round, task['name'], outpath)

    print('-'*30)