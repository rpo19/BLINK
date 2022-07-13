import argparse
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
from SlidingWindows import SlidingWindowsPipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer

from gatenlp import Document

DEFAULT_TAG='aplha_v0.1.0_electra'

class Item(BaseModel):
    text: str

app = FastAPI()

@app.post('/api/electraner')
async def encode_mention(doc: dict = Body(...)):

    doc = Document.from_dict(doc)
    entity_set = doc.annset(f'entities_{DEFAULT_TAG}')

    out = electra_pipeline(doc.text)

    for ent in out:
        # {'entity_group': 'PER', 'score': 0.9990982711315155, 'word': 'ww', 'start': 17, 'end': 32}
        feat_to_add = {
            "ner": {
                "type": ent['entity_group'],
                "score": ent['score'],
                "source": "electra",
                "model": args.model
                }}

        entity_set.add(ent['start'], ent['end'], ent['entity_group'], feat_to_add)

    if not 'pipeline' in doc.features:
        doc.features['pipeline'] = []
    doc.features['pipeline'].append('electraner')

    return doc.to_dict()

def electra_pipeline(text):
    global sw

    res = sw(text)

    assert len(res) == 1

    return res[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30309", help="port to listen at",
    )
    parser.add_argument(
        "--model", type=str, default="en_core_web_sm", help="model to load",
    )
    parser.add_argument(
        "--tokenizer", type=str, default='dbmdz/electra-base-italian-xxl-cased-discriminator', help="tokenizer",
    )
    parser.add_argument(
        "--tag", type=str, default=DEFAULT_TAG, help="AnnotationSet tag",
    )

    args = parser.parse_args()

    print('Loading model...')
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sw = SlidingWindowsPipeline(model, tokenizer)

    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
