from html import unescape
from urllib.parse import unquote
import pandas as pd

file_path = 'wiki_00'

df = pd.read_json(file_path, lines=True)

t1 = df['text'].values[0]

print(unquote(unescape(t1)))