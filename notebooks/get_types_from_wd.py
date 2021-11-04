#!/usr/bin/env python
# coding: utf-8

# In[115]:


import os
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import time


# In[16]:


basedir = '../data/BLINK_benchmark'


# In[8]:


datasets = [
    #'AIDA-YAGO2_testa.jsonl',
    'AIDA-YAGO2_testb.jsonl',
    'AIDA-YAGO2_train.jsonl',
    ]

# In[94]:


person = []

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
# person
sparql.setQuery('SELECT ?x { ?x wdt:P279* wd:Q215627 }')
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    person.append(result["x"]["value"])
print(len(person))


# In[93]:


organization = []

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
# person
sparql.setQuery('SELECT ?x { ?x wdt:P279* wd:Q43229 }')
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    organization.append(result["x"]["value"])
print(len(organization))


# In[91]:


location = []

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
# person
sparql.setQuery('SELECT ?x { ?x wdt:P279* wd:Q2221906 }')
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    location.append(result["x"]["value"])
print(len(location))


# In[85]:


def getwd(title):
    res = requests.get(f'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={title}&format=json')
    temp = res.json()['query']['pages']
    assert len(list(temp.keys())) == 1
    wd = temp[list(temp.keys())[0]]['pageprops']['wikibase_item']
    return wd


# In[112]:


def gettype(wikiurl):
    if wikiurl != 'NIL':
        wikiname = wikiurl.split('/')[-1]
        wd = getwd(wikiname)
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
    else:
        return (False, False, False, False)


# In[127]:


for d in datasets:
    dpath = os.path.join(basedir, d)
    print(dpath)
    df = pd.read_json(dpath, lines=True)
    
    i = 0
    i_end = df.shape[0]
    types = pd.DataFrame(columns=['per', 'loc', 'org', 'misc'])
    
    try:
        for i, row in df.iterrows():
        #while i < i_end-1:
            #i += 1
            #print(row['Wikipedia_URL'])
            time.sleep(1)
            #row = df.iloc[i]
            try:
                t = gettype(row['Wikipedia_URL'])
            except KeyError:
                t = (False, False, False, False)
            except:
                try:
                    t = gettype(row['Wikipedia_URL'])
                except KeyError:
                    t = (False, False, False, False)
                except:
                    time.sleep(5)
            types.loc[i] = t
            if i % 100 == 0:
                print('\r{}/100'.format(i/i_end*100), end='')
    except:
        pass
    types.to_csv(f'{d}_ner_type.csv')

