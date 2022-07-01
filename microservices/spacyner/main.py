import argparse
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
import spacy
from spacy.cli import download as spacy_download

from gatenlp import Document

DEFAULT_TAG='aplha_v0.1.0_spacy'

class Item(BaseModel):
    text: str

app = FastAPI()

@app.post('/api/spacyner')
async def encode_mention(doc: dict = Body(...)):

    doc = Document.from_dict(doc)
    sentence_set = doc.annset(f'sentences_{DEFAULT_TAG}')
    entity_set = doc.annset(f'entities_{DEFAULT_TAG}')

    spacy_out = spacy_pipeline(doc.text)

    # sentences
    for sent in spacy_out.sents:
        # TODO keep track of entities in this sentence?
        sentence_set.add(sent.start_char, sent.end_char, "sentence", {
            "source": "spacy",
            "spacy_model":args.model
        })

    for ent in spacy_out.ents:
        # TODO keep track of sentences
        # sentence_set.overlapping(ent.start_char, ent.end_char)
        feat_to_add = {
            "ner": {
                "type": ent.label_,
                "score": 1.0,
                "source": "spacy",
                "spacy_model": args.model
                }}
        if ent.label_ == 'DATE':
            feat_to_add['linking'] = {
                "skip": True
            }

        entity_set.add(ent.start_char, ent.end_char, ent.label_, feat_to_add)

    if not 'pipeline' in doc.features:
        doc.features['pipeline'] = []
    doc.features['pipeline'].append('spacyner')

    return doc.to_dict()

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
    parser.add_argument(
        "--tag", type=str, default=DEFAULT_TAG, help="AnnotationSet tag",
    )

    args = parser.parse_args()

    print('Loading spacy model...')
    # Load spacy model
    try:
        spacy_pipeline = spacy.load(args.model, exclude=['tok2vec', 'morphologizer', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    except Exception as e:
        spacy_download(args.model)
        spacy_pipeline = spacy.load(args.model, exclude=['tok2vec', 'morphologizer', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    spacy_pipeline.enable_pipe('senter')

    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
