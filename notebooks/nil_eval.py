from prepare_dataset import _load_datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import itertools

import click

def _eval(df, y_pred, y='y', title=None, threshold=0.5):
    ACC = df.query(f'(y==1 and {y_pred}>={threshold}) or (y==0 and {y_pred}<{threshold})').count()[0] / (sys.float_info.min + df.shape[0])
    TPR = df.query(f'y==1 and {y_pred} >= {threshold}').count()[0] / (sys.float_info.min + df.query(f'y>={threshold}').shape[0])
    TNR = df.query(f'y==0 and {y_pred} < {threshold}').count()[0] / (sys.float_info.min + df.query(f'y<{threshold}').shape[0])
    FNR = 1 - TPR
    PPV = df.query(f'y==1 and {y_pred} >= {threshold}').count()[0] / (sys.float_info.min + df.query(f'{y_pred}>={threshold}').shape[0])
    NPV = df.query(f'y==0 and {y_pred} < {threshold}').count()[0] / (sys.float_info.min + df.query(f'{y_pred}<{threshold}').shape[0])
    FPR = 1 - TNR
    FDR = 1 - PPV
    FOR = 1 - NPV
    F1 = 2*(PPV*TPR)/(PPV+TPR)
    return pd.DataFrame({
        "ACC": [ACC],
        "TPR": [TPR],
        "TNR": [TNR],
        "FNR": [FNR],
        "PPV": [PPV],
        "NPV": [NPV],
        "FPR": [FPR],
        "FDR": [FDR],
        "FOR": [FOR],
        "F1": [F1],        
    }, index = [y_pred if title is None else title])

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
        for tk in ['test', 'test_hard']:
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            _temp['y_pred'] = mdl.predict(t[x].values.reshape(-1, 1))

            yield _eval(_temp, 'y_pred', title=f'{x}+{k}+{tk}')

def _lr_double(datasets, x1, x2):
    for k in ['train', 'train_hard', 'train_aug', 'train_hard_aug']:
        d = datasets[k]
        mdl = LinearRegression().fit(d[[x1, x2]].values.reshape(-1, 2), d['y'])

        for tk in ['test', 'test_hard']:
            t = datasets[tk]
            _temp = pd.DataFrame()
            _temp['y'] = t['y']
            _temp['y_pred'] = mdl.predict(t[[x1, x2]].values.reshape(-1, 2))

            yield _eval(_temp, 'y_pred', title=f'{x1}+{x2}+{k}+{tk}')

@click.command()
@click.option('--data-path', required=True, type=str, help='path in which to find datasets')
@click.option('--save-results', required=False, type=str, default=None, help='whether and where to save results')
def main(data_path, save_results):
    datasets = _load_datasets(data_path)
    
    # train + train_hard + train_aug + train_hard_aug (4)

    # test + test_hard (2)

    results = pd.DataFrame()
    for _df in itertools.chain(
        _lr_single(datasets, 'max_bi'),
        _lr_single(datasets, 'max_cross'),
        _lr_double(datasets, 'max_bi', 'max_cross'),
        ):
        results = pd.concat([results, _df])

    print(results.to_markdown())

    if save_results is not None:
        results.to_csv(save_results, index=True) 

if __name__ == '__main__':
    main()