import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Union, List
import spacy
import sys

class Item(BaseModel):
    text: Union[List[str], str]

app = FastAPI()

@app.post('/api/ner')
async def encode_mention(item: Item):
    samples = []
    sentences = []

    if isinstance(item.text, str):
        input_sentences = nlp(item.text).sents
        process_sent = lambda x: (x, x.text)
    else:
        input_sentences = item.text
        process_sent = lambda x: (nlp(x), x)

    for i, sentence in enumerate(input_sentences):
        doc, sentence = process_sent(sentence)

        sentences.append(sentence)

        for ent in doc.ents:
            sample = {
                'label': 'unknown',
                'label_id': -1,
                'context_left': sentence[:ent.start_char],
                'context_right': sentence[ent.end_char:],
                'mention': ent.text,
                'start_pos': ent.start_char,
                'end_pos': ent.end_char,
                'sent_idx': i,
                'ner_type': ent.label_
            }

            samples.append(sample)

    return {
        'ents': samples,
        'sentences': sentences
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30304", help="port to listen at",
    )
    parser.add_argument(
        "--model", type=str, default="en_core_web_sm", help="spacy model to load",
    )

    args = parser.parse_args()

    print('Loading spacy model...')
    # Load spacy model
    try:
        nlp = spacy.load(args.model, exclude=['tok2vec', 'morphologizer', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    except Exception as e:
        print('ERROR.')
        print(e)
        if "Can't find model" in str(e):
            print('Maybe you did not download the model. To download it run ```python -m spacy download $MODEL```.')
        sys.exit(1)
    nlp.enable_pipe('senter')
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
