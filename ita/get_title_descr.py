from html import unescape
from urllib.parse import unquote
import pandas as pd
from bs4 import BeautifulSoup
import bs4
import os
import json
import sys
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def _helper(x):
    if isinstance(x, bs4.element.Tag):
        return x.text
    elif isinstance(x, str):
        return x
    else:
        raise Exception('unexpected type')
        
def parse_text(text, max_tokens = 128):
    text = unescape(text)
    soup = BeautifulSoup(text, 'lxml')
    # remove links
    text = soup.text
    # truncate
    limit = find_nth(text, ' ', max_tokens)
    if limit > 0:
        text = text[:limit]
    return text

def process_file(file_path):
    control_file = file_path + '.parsed.done'
    if not os.path.isfile(control_file):
        df = pd.read_json(file_path, lines=True)

        with open(file_path + '.parsed.jsonl', 'w') as fd:
            for i, row in df.iterrows():
                row['parsed'] = parse_text(row['text'])
                jsline = row.to_json()
                print(jsline, file=fd)

        open(control_file, 'w').close()

if __name__ == '__main__':
    file_list = sys.stdin.read().splitlines()

    res = process_map(process_file, file_list, chunksize=10, max_workers=20)

    print('done.')
