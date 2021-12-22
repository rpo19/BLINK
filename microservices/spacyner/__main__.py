import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Union, List
import spacy

class Item(BaseModel):
    text: Union[List[str], str]

app = FastAPI()

@app.post('/api/ner')
async def encode_mention(item: Item):
    if isinstance(item.text, str):
        texts = [item.text]
    else:
        texts = item.text
    samples = _annotate(nlp, texts)
    return samples

def _annotate(nlp, input_sentences):
    samples = []
    for sentence in input_sentences:
        doc = nlp(sentence)

        for i, ent in enumerate(doc.ents):
            sample = {
                'label': 'unknown',
                'label_id': -1,
                'context_left': sentence[:ent.start_char],
                'context_right': sentence[ent.end_char:],
                'mention': ent.text,
                'start_pos': ent.start_char,
                'end_pos': ent.end_char,
                'sent_ids': i,
                'ner_type': ent.label_
            }

            samples.append(sample)
    return samples

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
    nlp = spacy.load(args.model, exclude=['tok2vec', 'morphologizer', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
