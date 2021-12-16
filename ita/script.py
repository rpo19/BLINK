from html import unescape
from urllib.parse import unquote
import pandas as pd
from bs4 import BeautifulSoup
import bs4
import os
import json
import sys
from multiprocessing import Pool
from tqdm import tqdm

file_path = 'wiki_00'

df = pd.read_json(file_path, lines=True)

t1 = df['text'].values[0]

print(unquote(unescape(t1)))

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
        
def get_mention_context(tag, max_tokens=32):
    mention = tag.text
    href = unquote(tag.get('href'))
    context_left = ''.join(map(_helper, list(tag.previous_siblings)[::-1]))
    context_right = ''.join(map(_helper, tag.next_siblings))
    # truncate at max_tokens
    right_limit = find_nth(context_right, ' ', 32)
    if right_limit > 0:
        context_right = context_right[:right_limit]
    # revert left and search left to right. then revert back
    reverse_context_left = context_left[::-1]
    left_limit = find_nth(reverse_context_left, ' ', 32)
    if left_limit > 0:
        context_left = reverse_context_left[:left_limit][::-1]
    return {'href': href, 'mention': mention, 'context_left': context_left, 'context_right': context_right}

def process_text(text):
    text = unescape(text)
    soup = BeautifulSoup(text)
    tags = soup.find_all('a')
    res = list(map(get_mention_context, tags))
    return res

def process_file(file_path):
    df = pd.read_json(file_path, lines=True)

    with open(os.path.join(file_path, '.jsonl'), 'w') as fd:
        for i, row in df.iterrows():
            for line in process_text(row['text']):
                jsline = json.dumps(line)
                print(jsline, file=fd)

if __name__ == '__main__':
    file_list = sys.stdin.read().splitlines()

    with Pool(20) as p:
        tqdm(p.imap_unordered(process_file, file_list, chunksize=10), total=len(file_list))