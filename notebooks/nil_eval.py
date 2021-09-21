import click
from prepare_dataset import _load_datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import itertools
import re
import math
import os

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pickle
_save_models = False
_save_models_path = None


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train data


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class binaryClassification(nn.Module):
    def __init__(self, n):
        super(binaryClassification, self).__init__()
        self.fc1 = nn.Linear(n, 2)
        self.fc2 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = nn.Sigmoid(x)

        return x


def _fbeta(b, PPV, TPR):
    f = (1+b**2) * PPV * TPR / (b**2*PPV + TPR + sys.float_info.min)
    return f


def _eval(df, y_pred, y='y', title=None, threshold=0.5):
    tot = df.shape[0]
    TP = df.query(f'y==1 and {y_pred} >= {threshold}').count()[0]
    TN = df.query(f'y==0 and {y_pred} < {threshold}').count()[0]
    FP = df.query(f'y==0 and {y_pred} >= {threshold}').count()[0]
    FN = df.query(f'y==1 and {y_pred} < {threshold}').count()[0]
    assert tot == TP + TN + FP + FN
    ACC = (TP + TN) / (sys.float_info.min + tot)
    TPR = TP / (sys.float_info.min + TP + FN)
    TNR = TN / (sys.float_info.min + TN + FP)
    FNR = 1 - TPR
    PPV = TP / (sys.float_info.min + TP + FP)
    NPV = TN / (sys.float_info.min + TN + FN)
    FPR = 1 - TNR
    # FDR = 1 - PPV
    # FOR = 1 - NPV
    F1 = _fbeta(1, PPV, TPR)
    F2 = _fbeta(2, PPV, TPR)
    F05 = _fbeta(0.5, PPV, TPR)
    FN1 = _fbeta(1, NPV, TNR)
    FN2 = _fbeta(2, NPV, TNR)
    FN05 = _fbeta(1, NPV, TNR)
    MCC = (TP*TN - FP*FN) / (sys.float_info.min +
                             math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

    return pd.DataFrame({
        "ACC": [ACC],
        "MCC": [MCC],
        "F1": [F1],
        "F2": [F2],
        "F05": [F05],
        "FN1": [FN1],
        "FN2": [FN2],
        "FN05": [FN05],
        "TPR": [TPR],
        "TNR": [TNR],
        "FPR": [FPR],
        "FNR": [FNR],
        "PPV": [PPV],
        "NPV": [NPV],
        # "FDR": [FDR],
        # "FOR": [FOR],
    }, index=[y_pred if title is None else title])


def _cm(df, y_pred, y='y', threshold=0.5):
    TP = df.query(f'y==1 and {y_pred} >= {threshold}').count()[0]
    FP = df.query(f'y==0 and {y_pred} >= {threshold}').count()[0]
    TN = df.query(f'y==0 and {y_pred} < {threshold}').count()[0]
    FN = df.query(f'y==1 and {y_pred} < {threshold}').count()[0]
    cmdf = pd.DataFrame([
        {"T": TP, "F": FP},
        {"T": FN, "F": TN}
    ], index=['T', 'F'])
    cmdf.index.name = 'pred\gold'
    return cmdf


def _cmn(df, y_pred, y='y'):
    cmdf = _cm(df, y_pred, y)
    cmdf['T'] = cmdf['T']/((df['y'] == 1).sum())
    cmdf['F'] = cmdf['F']/((df['y'] == 0).sum())
    return cmdf


def _lr_single(datasets, x):
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        d = datasets[k]
        assert d[x].isna().sum() == 0
        mdl = LinearRegression().fit(d[x].values.reshape(-1, 1), d['y'])

        if _save_models:
            with open(os.path.join(_save_models_path, f'lr_{x}+{k}.pkl'), 'wb') as fd:
                pickle.dump(mdl, fd)

        for tk in ['test', 'test_hard']:
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            _temp['y_pred'] = mdl.predict(t[x].values.reshape(-1, 1))

            yield _eval(_temp, 'y_pred', title=f'lr_{x}+{k}+{tk}')


def _lr_double(datasets, x1, x2):
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        d = datasets[k]
        mdl = LinearRegression().fit(d[[x1, x2]].values.reshape(-1, 2), d['y'])

        if _save_models:
            with open(os.path.join(_save_models_path, f'lr_{x1}+{x2}+{k}.pkl'), 'wb') as fd:
                pickle.dump(mdl, fd)

        for tk in ['test', 'test_hard']:
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            _temp['y_pred'] = mdl.predict(t[[x1, x2]].values.reshape(-1, 2))

            yield _eval(_temp, 'y_pred', title=f'lr_{x1}+{x2}+{k}+{tk}')


def _lr_all(datasets, _filter=None, title='all'):
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        d = datasets[k]
        cols = [c for c in d.columns
                if c not in ['y', 'idx']
                and not c.startswith('correct')
                and (_filter is None or re.match(_filter, c))]
        #print('_lr_all selected columns:', cols)
        mdl = LinearRegression().fit(
            d[cols].values.reshape(-1, len(cols)), d['y'])

        if _save_models:
            with open(os.path.join(_save_models_path, f'lr_{title}+{k}.pkl'), 'wb') as fd:
                pickle.dump(mdl, fd)

        for tk in ['test', 'test_hard', 'test_aug', 'test_hard_aug']:
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            _temp['y_pred'] = mdl.predict(
                t[cols].values.reshape(-1, len(cols)))

            yield _eval(_temp, 'y_pred', title=f'lr_{title}+{k}+{tk}')


def _svc_cross_max(datasets, title='svc_cross_max'):
    print(title)
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        print('training on...', k)
        d = datasets[k]
        clf = make_pipeline(StandardScaler(),
                            SVC(random_state=42, tol=1e-5, max_iter=100000, probability=True))
        X = d[['max_cross']].values
        y = d['y'].values
        clf.fit(X, y)

        if _save_models:
            with open(os.path.join(_save_models_path, f'{title}+{k}.pkl'), 'wb') as fd:
                pickle.dump(clf, fd)

        for tk in ['test', 'test_hard', 'test_aug', 'test_hard_aug']:
            print('testing on...', tk)
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            tX = t[['max_cross']].values
            _temp['y_pred'] = clf.predict(tX)

            yield _eval(_temp, 'y_pred', title=f'{title}+{k}+{tk}')


def _svc_max(datasets, title='svc_max'):
    print(title)
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        print('training on...', k)
        d = datasets[k]
        clf = make_pipeline(StandardScaler(),
                            SVC(random_state=42, tol=1e-5, max_iter=100000, probability=True))
        X = d[['max_bi', 'max_cross']].values
        y = d['y'].values
        clf.fit(X, y)

        if _save_models:
            with open(os.path.join(_save_models_path, f'{title}+{k}.pkl'), 'wb') as fd:
                pickle.dump(clf, fd)

        for tk in ['test', 'test_hard', 'test_aug', 'test_hard_aug']:
            print('testing on...', tk)
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            tX = t[['max_bi', 'max_cross']].values
            _temp['y_pred'] = clf.predict(tX)

            yield _eval(_temp, 'y_pred', title=f'{title}+{k}+{tk}')


def _svc_cross(datasets, title='svc_cross'):
    print(title)
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        print('training on...', k)
        d = datasets[k]
        clf = make_pipeline(StandardScaler(),
                            SVC(random_state=42, tol=1e-5, max_iter=100000, probability=True))
        X = d[['max_cross', 'second_cross', 'min_cross',
               'mean_cross', 'median_cross', 'stdev_cross']].values
        y = d['y'].values
        clf.fit(X, y)

        if _save_models:
            with open(os.path.join(_save_models_path, f'{title}+{k}.pkl'), 'wb') as fd:
                pickle.dump(clf, fd)

        for tk in ['test', 'test_hard', 'test_aug', 'test_hard_aug']:
            print('testing on...', tk)
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            tX = t[['max_cross', 'second_cross', 'min_cross',
                    'mean_cross', 'median_cross', 'stdev_cross']].values
            _temp['y_pred'] = clf.predict(tX)

            yield _eval(_temp, 'y_pred', title=f'{title}+{k}+{tk}')


def _svc_bi(datasets, title='svc_bi'):
    print(title)
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        print('training on...', k)
        d = datasets[k]
        clf = make_pipeline(StandardScaler(),
                            SVC(random_state=42, tol=1e-5, max_iter=100000, probability=True))
        X = d[['max_bi', 'second_bi', 'min_bi', 'mean_bi', 'median_bi',
               'stdev_bi']].values
        y = d['y'].values
        clf.fit(X, y)

        if _save_models:
            with open(os.path.join(_save_models_path, f'{title}+{k}.pkl'), 'wb') as fd:
                pickle.dump(clf, fd)

        for tk in ['test', 'test_hard', 'test_aug', 'test_hard_aug']:
            print('testing on...', tk)
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            tX = t[['max_bi', 'second_bi', 'min_bi', 'mean_bi', 'median_bi',
                    'stdev_bi']].values
            _temp['y_pred'] = clf.predict(tX)

            yield _eval(_temp, 'y_pred', title=f'{title}+{k}+{tk}')


def _svc_all(datasets, title='svc_all'):
    print(title)
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        print('training on...', k)
        d = datasets[k]
        clf = make_pipeline(StandardScaler(),
                            SVC(random_state=42, tol=1e-5, max_iter=100000, probability=True))
        X = d[['max_bi', 'second_bi', 'min_bi', 'mean_bi', 'median_bi',
               'stdev_bi', 'max_cross', 'second_cross', 'min_cross',
               'mean_cross', 'median_cross', 'stdev_cross']].values
        y = d['y'].values
        clf.fit(X, y)

        if _save_models:
            with open(os.path.join(_save_models_path, f'{title}+{k}.pkl'), 'wb') as fd:
                pickle.dump(clf, fd)

        for tk in ['test', 'test_hard', 'test_aug', 'test_hard_aug']:
            print('testing on...', tk)
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            tX = t[['max_bi', 'second_bi', 'min_bi', 'mean_bi', 'median_bi',
                    'stdev_bi', 'max_cross', 'second_cross', 'min_cross',
                    'mean_cross', 'median_cross', 'stdev_cross']].values
            _temp['y_pred'] = clf.predict(tX)

            yield _eval(_temp, 'y_pred', title=f'{title}+{k}+{tk}')


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def _nrl(datasets, title='nrl', features=['max_cross'], EPOCHS=100, BATCH_SIZE=64, LEARNING_RATE=0.0001):
    print('nrl-{} ...'.format('+'.join(features)))
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        print('train on', k)
        d = datasets[k]
        assert d[features].isna().sum().sum() == 0
        scaler = StandardScaler()

        X_train = d[features].values
        y_train = d['y'].values

        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_validation)

        X_validation = torch.FloatTensor(X_validation)
        y_validation = torch.FloatTensor(y_validation.reshape(-1, 1))

        model = binaryClassification(len(features))
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_data = trainData(torch.FloatTensor(X_train),
                        torch.FloatTensor(y_train))
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        for e in range(1, EPOCHS+1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch.unsqueeze(1))

                with torch.no_grad():
                    y_pred_validation = model(X_validation)
                    val_loss = float(F.binary_cross_entropy_with_logits(y_pred_validation, y_validation))

                acc = binary_acc(y_pred, y_batch.unsqueeze(1))

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            if e % 5 == 0:
                print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Val Loss: {val_loss:.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

        if _save_models:
            with open(os.path.join(_save_models_path, 'stdscaler+{}+{}.pkl'.format('+'.join(features), k)), 'wb') as fd:
                pickle.dump(scaler, fd)
            torch.save(model.state_dict(), os.path.join(_save_models_path, '{}+{}+{}.torch'.format(title, '+'.join(features), k)))

        for tk in ['test', 'test_hard']:
            print('test on', tk)
            t = datasets[tk]

            X_test = t[features].values
            X_test = scaler.transform(X_test)

            y_test = t['y']

            test_data = testData(torch.FloatTensor(X_test))
            test_loader = DataLoader(dataset=test_data, batch_size=1)

            y_pred_list = []
            model.eval()
            with torch.no_grad():
                for X_batch in test_loader:
                    X_batch = X_batch.to(device)
                    y_test_pred = model(X_batch)
                    y_test_pred = torch.sigmoid(y_test_pred)
                    y_pred_tag = torch.round(y_test_pred)
                    y_pred_list.append(y_pred_tag.cpu().numpy())
            y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

            _temp = pd.DataFrame()
            _temp['y'] = y_test
            _temp['y_pred'] = y_pred_list

            yield _eval(_temp, 'y_pred', title='{}+{}+{}+{}'.format(title, '+'.join(features), k, tk))

def _print_results(results_df, sort=None, grep=None, reverse=False):
    to_print = results_df.copy()
    if grep:
        to_print = to_print.loc[[
            i for i in to_print.index if re.match(grep, i)]]
    if sort:
        to_print = to_print.sort_values(by=sort, ascending=reverse)
    to_print = to_print.round(decimals=3)
    print(to_print.to_markdown())


@click.command()
@click.option('-d', '--data-path', required=False, type=str, help='path in which to find datasets')
@click.option('-r', '--results', required=False, type=str, default=None, help='whether and where to save results')
@click.option('-s', '--sort', required=False, type=str, default=None, help='sort by metric')
@click.option('--reverse', required=False, default=False, is_flag=True, help='reverse sort order')
@click.option('-f', '--fast', required=False, default=False, is_flag=True, help='do not calculate but load from --results and show')
@click.option('-g', '--grep', required=False, default=None, type=str, help='regex to filter index')
@click.option('--save', required=False, default=None, type=str, help='save models into this folder')
def main(data_path, results, sort, fast, grep, reverse, save):

    if fast and results is None:
        raise Exception('--fast requires --results to load dataset')

    if fast:
        results_df = pd.read_csv(results, index_col=0)
        _print_results(results_df, sort, grep, reverse)
        sys.exit(0)

    if save is not None:
        global _save_models
        _save_models = True

        global _save_models_path
        _save_models_path = save

        if not os.path.isdir(_save_models_path):
            os.mkdir(_save_models_path)

    if not data_path:
        raise Exception('--data-path not set!')

    datasets = _load_datasets(data_path)

    # train + train_hard + train_aug + train_hard_aug (4)

    # test + test_hard (2)

    results_df = pd.DataFrame()
    for _df in itertools.chain(
        # _lr_single(datasets, 'max_bi'),
        # _lr_single(datasets, 'max_cross'),
        # _lr_double(datasets, 'max_bi', 'max_cross'),
        # _lr_all(datasets, '.*_bi', 'all_bi'),
        # _lr_all(datasets, '.*_cross', 'all_cross'),
        # _lr_all(datasets),
        # _svc_all(datasets),
        # _svc_bi(datasets),
        # _svc_cross(datasets),
        # _svc_max(datasets),
        # _svc_cross_max(datasets),
        _nrl(datasets, features=['max_cross']),
        _nrl(datasets, features=['max_bi']),
        _nrl(datasets, features=['max_bi', 'max_cross']),
        _nrl(datasets, features=['max_bi',
                       'second_bi',
                       'min_bi',
                       'mean_bi',
                       'median_bi',
                       'stdev_bi']),
        # _nrl(datasets, features=['max_cross',
        #                'second_cross',
        #                'min_cross',
        #                'mean_cross',
        #                'median_cross',
        #                'stdev_cross']),
        _nrl(datasets, features=['max_bi',
                       'second_bi',
                       'min_bi',
                       'mean_bi',
                       'median_bi',
                       'stdev_bi',
                       'max_cross',
                       'second_cross',
                       'min_cross',
                       'mean_cross',
                       'median_cross',
                       'stdev_cross'])
    ):
        results_df = pd.concat([results_df, _df])

    _print_results(results_df, sort, grep, reverse)

    if results is not None:
        results_df.to_csv(results, index=True)


if __name__ == '__main__':
    main()
