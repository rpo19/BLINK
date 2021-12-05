#!/usr/bin/env python
# coding: utf-8

# In[115]:


import os
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import time
import pickle
from bs4 import BeautifulSoup

# In[94]:
print('Loading persons...')
person_path = './data/wikidata_type_person.pickle'
if os.path.isfile(person_path):
    with open(person_path, 'rb') as fd:
        person = pickle.load(fd)
else:
    person = []
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # person
    sparql.setQuery('SELECT ?x { ?x wdt:P279* wd:Q215627 }')
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    for result in results["results"]["bindings"]:
        person.append(result["x"]["value"])

    with open(person_path, 'wb') as fd:
        pickle.dump(person, fd)

print(len(person))


# In[93]:


print('Loading organization...')
organization_path = './data/wikidata_type_organization.pickle'
if os.path.isfile(organization_path):
    with open(organization_path, 'rb') as fd:
        organization = pickle.load(fd)
else:
    organization = []
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # person
    sparql.setQuery('SELECT ?x { ?x wdt:P279* wd:Q43229 }')
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    for result in results["results"]["bindings"]:
        organization.append(result["x"]["value"])

    with open(organization_path, 'wb') as fd:
        pickle.dump(organization, fd)

print(len(organization))


# In[91]:


print('Loading location...')
location_path = './data/wikidata_type_location.pickle'
if os.path.isfile(location_path):
    with open(location_path, 'rb') as fd:
        location = pickle.load(fd)
else:
    location = []
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # person
    sparql.setQuery('SELECT ?x { ?x wdt:P279* wd:Q2221906 }')
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    for result in results["results"]["bindings"]:
        location.append(result["x"]["value"])

    with open(location_path, 'wb') as fd:
        pickle.dump(location, fd)

print(len(location))


# In[85]:


def getwd(title):
    res = requests.get(f'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={title}&format=json')
    temp = res.json()['query']['pages']
    assert len(list(temp.keys())) == 1
    wd = temp[list(temp.keys())[0]]['pageprops']['wikibase_item']
    return wd


# In[112]:


def gettype(wd):
    q = f'SELECT ?x {{ <http://www.wikidata.org/entity/{wd}> wdt:P31 ?x. }}'

    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    per = False
    loc = False
    org = False
    misc = False
    for result in results["results"]["bindings"]:
        if result["x"]["value"] in person:
            per = True
        if result["x"]["value"] in location:
            loc = True
        if result["x"]["value"] in organization:
            org = True
    misc = not per and not loc and not org

    assert per or loc or org or misc

    return (per, loc, org, misc)

def getwikidataid(wikipediaurl):
    res = requests.get(wikipediaurl)
    html = BeautifulSoup(res.text, 'html.parser')
    for l in html.find_all('a', href=True):
        if l.text == 'Wikidata item':
            return l.get('href').split('/')[-1]

    return None


# In[127]:

print('Loading nil dataset')
nil_dataset_path = './data/nil_dataset.pickle'
whole = pd.read_pickle(nil_dataset_path)

## getting unique entities
# load wikipedia_id2local_id
wikipedia_id2local_id_path = './data/wikipedia_id2local_id.pickle'
print('Loading wikipedia_id2local_id from {}'.format(wikipedia_id2local_id_path))
if os.path.isfile(wikipedia_id2local_id_path):
    with open(wikipedia_id2local_id_path, 'rb') as fd:
        wikipedia_id2local_id = pickle.load(fd)
else:
    raise Exception('{} not found! Generate it with `python blink/main_dense.py --save-wikipedia-id2local-id`.'.format(wikipedia_id2local_id_path))

local_id2wikipedia_id = {v:k for k,v in wikipedia_id2local_id.items()}
id2url = {
    v: "https://en.wikipedia.org/wiki?curid=%s" % k
    for k, v in wikipedia_id2local_id.items()
}

# only for aida
aida = whole.query('src.str.contains("AIDA-YAGO2")')

t_df2 = pd.DataFrame(columns=['title', 'id', 'wd', 'per', 'loc', 'org', 'misc'])
t_df2[['title', 'id']] = pd.concat([
    aida[['bi_best_candidate_title', 'bi_best_candidate']].rename(
        columns={'bi_best_candidate_title': 'title', 'bi_best_candidate': 'id'}),
    aida[['cross_best_candidate_title', 'cross_best_candidate']].rename(
        columns={'cross_best_candidate_title': 'title', 'cross_best_candidate': 'id'})])
t_df2 = t_df2.drop_duplicates(subset=['id'])

sleep_time = 1

output_path = './data/aida_ner_types_wikidata.csv'
if os.path.isfile(output_path):
    t_df2 = pd.read_csv(output_path, index_col=0)

print('starting')
missing = t_df2.loc[t_df2['per'].isna()]
counter = 0
for i,row in missing.iterrows():
    counter += 1
    time.sleep(sleep_time)
    try:
        wd = getwikidataid(id2url[row['id']])
        t_df2.loc[i, 'wd'] = wd
    except:
        print()
        print(f'Error getting wikidata id for i={i}')
        time.sleep(5)
        continue
    try:
        t = gettype(wd)
        t_df2.loc[i, ['per', 'loc', 'org', 'misc']] = t
    except:
        print()
        print(f'Error getting type for i={i}')
        time.sleep(5)
        continue
    print('\r{}/{}'.format(counter, missing.shape[0]), end='')

# convert from booleans to floats
t_df2[['per', 'loc', 'org', 'misc']] = t_df2[['per', 'loc', 'org', 'misc']].astype(float)

nanum = t_df2['per'].isna().sum()
print('Finished getting types.')
if nanum > 0:
    print('There are {} missing types, re-run until they are zero.'.format(nanum))
else:
    print('All done.')


print('Saving types to {}...'.format(output_path))
t_df2.to_csv(output_path)

print('Updating {}...'.format(nil_dataset_path))

pre_shape = whole.shape[0]

if 'wiki_per_cross' in whole.columns:
    print('Overwriting existing wiki types from {}...'.format(nil_dataset_path))
    whole = whole.drop(columns=[
        'wiki_per_cross',
        'wiki_loc_cross',
        'wiki_org_cross',
        'wiki_misc_cross',
        'wiki_per_bi',
        'wiki_loc_bi',
        'wiki_org_bi',
        'wiki_misc_bi',
        ])

whole = whole.merge(t_df2.rename(columns={
    'per': 'wiki_per_cross',
    'loc': 'wiki_loc_cross',
    'org': 'wiki_org_cross',
    'misc': 'wiki_misc_cross'
})[['id',
    'wiki_per_cross',
    'wiki_loc_cross',
    'wiki_org_cross',
    'wiki_misc_cross']], how='left', left_on='cross_best_candidate', right_on='id').drop(columns='id')

whole = whole.merge(t_df2.rename(columns={
    'per': 'wiki_per_bi',
    'loc': 'wiki_loc_bi',
    'org': 'wiki_org_bi',
    'misc': 'wiki_misc_bi'
})[['id',
    'wiki_per_bi',
    'wiki_loc_bi',
    'wiki_org_bi',
    'wiki_misc_bi']], how='left', left_on='bi_best_candidate', right_on='id').drop(columns='id')

assert pre_shape == whole.shape[0]

print('Saving updated {}...'.format(nil_dataset_path))
whole.to_pickle(nil_dataset_path)
