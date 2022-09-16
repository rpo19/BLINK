import requests
from tqdm import tqdm

BASE_URL = 'http://localhost:3001'

documents = requests.get(BASE_URL + '/api/document?limit=99999')
documents = documents.json()['docs']


for doc in tqdm(documents):
  doc_id = doc['id']
  full_doc = requests.get(BASE_URL + '/api/document/' + str(doc_id))
  full_doc = full_doc.json()

  for ann_set_key in full_doc['annotation_sets']:
    if 'trie' in ann_set_key:
      ann_set_id = full_doc['annotation_sets'][ann_set_key]['_id']
      res = requests.delete(BASE_URL + '/api/document/' + str(doc_id) + '/annotation-set/' + str(ann_set_id))

