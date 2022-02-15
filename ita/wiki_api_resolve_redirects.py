import requests
from tqdm.contrib.concurrent import process_map
from urllib.parse import quote
import pandas as pd
import os

"""
Resolves redirects.
Probably worth to use sql dump. TODO
"""

def get_id(title, title2norm, title2pageid):
    if title in title2norm:
        title = title2norm[title]
    if title in title2pageid:
        return title2pageid[title]
    else:
        return -1

def wiki_href2id(href):
    ids = []
    
    href_string = '|'.join([h for h in href if h != -1 ])
    #href_string = quote(href_string)
    url = 'https://it.wikipedia.org/w/api.php?action=query&pageids={}&format=json&redirects'.format(href_string)
    res = requests.get(url)
    pageid = None
    uncertain = 0
    title2norm = {}
    title2pageid = {}
    if res.ok:
        js = res.json()
        if 'query' in js and 'pages' in js['query']:
            if 'normalized' in js['query']:
                for i in js['query']['normalized']:
                    title2norm[i['from']] = i['to']
            
            for i,v in js['query']['pages'].items():
                if 'pageid' in v:
                    title2pageid[v['title']] = v['pageid']
          
    for hr in href:
        if hr is None:
            ids.append(-1)
        else:
            ids.append(get_id(hr, title2norm, title2pageid))

    return ids

def process_batch(batch):
    idx = batch.index
    outfile = 'wikirefid/id_{}.pickle'.format(idx[0])
    if not os.path.isfile(outfile):
        vals = batch['href'].values
        refids = wiki_href2id(vals, {})
        res_df = pd.DataFrame(index=idx, data={'refid':refids})
        res_df.to_pickle(outfile, protocol=4)
        #return res_df

if __name__ == '__main__':
    print('loading mentions...')
    mentions_df = pd.read_pickle('mentions.pickle')
    print('mentions loaded.')
    batch_size = 20

    print('creating batches...')
    batches = [mentions_df[i:i+batch_size] for i in range(0,mentions_df.shape[0],batch_size)]

    print('start...')
    process_map(process_batch, batches, max_workers=20, chunksize=20)
    print('done.')
